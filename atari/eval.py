import argparse
import torch
import wandb

from dqn.buffer import Buffer
from common.load_cfg import load_cfg
from atari import make_vec_envs
from dqn import actor_iter, DQN
from repre.w_mse import WMSE
from repre.predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", type=str, default="default")
    parser.add_argument("--env", type=str, default="MontezumaRevenge")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ri_scale", type=float, default=1)
    p = parser.parse_args()
    cfg = load_cfg(p.cfg)
    cfg.update(vars(p))
    wandb.init(project="lwm", config=cfg)

    num_env = cfg["agent"]["actors"]
    fstack = cfg["agent"]["frame_stack"]
    envs = make_vec_envs(cfg["env"], num_env, cfg["seed"])

    buffer = Buffer(
        num_env=num_env,
        maxlen=int(cfg["buffer"]["size"] / num_env),
        obs_shape=envs.observation_space.shape,
        device=cfg["buffer"]["device"],
    )
    model = DQN(envs.action_space.n, fstack).cuda().train()
    wmse = WMSE(buffer, cfg)
    pred = Predictor(buffer, wmse.encoder, envs.action_space.n, cfg)
    actor = actor_iter(envs, model, pred, 0, eps=0.001)

    wmse.load(), pred.load()
    cp = torch.load("models/dqn.pt", map_location="cuda")
    model.load_state_dict(cp)
    model.eval()

    while True:
        full_step = buffer.get_recent(fstack + 1)
        step, hx, log = actor.send(full_step)
        buffer.append(step)
        if "reward" in log:
            wandb.log({"final_reward": log["reward"]})
            break

    wandb.save("models/dqn.pt")
    wandb.save("models/w_mse.pt")
    wandb.save("models/predictor.pt")
