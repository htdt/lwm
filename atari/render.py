import time
import argparse
from tqdm import trange
import torch

from common.load_cfg import load_cfg
from atari import make_vec_envs
from dqn import actor_iter, DQN
from dqn.buffer import Buffer
from repre.w_mse import WMSE
from repre.predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", type=str, default="default")
    parser.add_argument("--env", type=str, default="MontezumaRevenge")
    parser.add_argument("--ri_scale", type=float, default=1)
    p = parser.parse_args()
    cfg = load_cfg(p.cfg)
    cfg.update(vars(p))

    num_env = cfg["agent"]["actors"]
    fstack = cfg["agent"]["frame_stack"]
    env = make_vec_envs(name=cfg["env"], num=1)
    model = DQN(env.action_space.n, fstack, device="cpu")
    wmse = WMSE(None, cfg, device="cpu")
    pred = Predictor(None, wmse.encoder, env.action_space.n, cfg, device="cpu")
    actor = actor_iter(env, model, pred, 0, eps=0)
    obs_shape = env.observation_space.shape
    buffer = Buffer(num_env=1, maxlen=fstack + 1, obs_shape=obs_shape, device="cpu")

    wmse.load(), pred.load()
    cp = torch.load("models/dqn.pt", map_location="cpu")
    model.load_state_dict(cp)
    model.eval()

    for n_iter in trange(2000):
        full_step = buffer.get_recent(fstack + 1, "cpu")
        step, hx, log_a = actor.send(full_step)
        buffer.append(step)
        env.render()
        time.sleep(1 / 30)
