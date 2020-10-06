import os
import time
import torch

from common.load_cfg import load_cfg
from env import make_vec_envs
from dqn import actor_iter, DQN
from dqn.buffer import Buffer
from predictor import Predictor


if __name__ == "__main__":
    cfg = load_cfg("default")
    cfg["env"] = "pol"

    num_env = cfg["agent"]["actors"]
    env = make_vec_envs(
        num=1,
        size=3,
        max_ep_len=cfg["train"]["max_ep_len"],
        seed=10,
    )
    model = DQN(cfg["agent"]["rnn_size"], device="cpu")
    pred = Predictor(None, cfg, device="cpu")
    actor = actor_iter(env, model, pred, 0, eps=0)
    buffer = Buffer(num_env=1, maxlen=2, obs_shape=(4,), device="cpu")

    cp = torch.load("models/dqn.pt", map_location="cpu")
    model.load_state_dict(cp)
    model.eval()
    pred.load()

    for n_iter in range(2000):
        full_step = buffer.get_recent(2, "cpu")
        step, hx, log_a = actor.send(full_step)
        buffer.append(step)
        # env.render()
        os.system("clear")
        env.remotes[0].send(('render', None))
        env.remotes[0].recv()
        time.sleep(1)
