import argparse
from tqdm import trange
import wandb
import torch

from dqn.buffer import Buffer
from common.load_cfg import load_cfg
from atari import make_vec_envs
from dqn import actor_iter
from repre.w_mse import WMSE
from repre.inverse_dynamics import IDF
from repre.cpc import CPC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", type=str, default="default")
    parser.add_argument("--env", type=str, default="MontezumaRevenge")
    p = parser.parse_args()
    cfg = load_cfg(p.cfg)
    cfg.update(vars(p))
    wandb.init(project="lwm", config=cfg)

    num_env = cfg["agent"]["actors"]
    fstack = cfg["agent"]["frame_stack"]
    envs = make_vec_envs(cfg["env"], num_env, max_ep_len=cfg["train"]["max_ep_len"])
    num_action = envs.action_space.n

    buffer = Buffer(
        num_env=num_env,
        maxlen=int(cfg["buffer"]["size"] / num_env),
        obs_shape=envs.observation_space.shape,
        device=cfg["buffer"]["device"],
    )
    wmse = WMSE(buffer, cfg)
    idf = IDF(buffer=buffer, num_action=num_action)
    cpc = CPC(buffer=buffer, num_action=num_action)
    actor = actor_iter(envs, None, None, cfg["buffer"]["warmup"], eps=1)

    pretrain = int(cfg["buffer"]["warmup"] / num_env)
    for n_iter in trange(pretrain):
        step, hx, log = next(actor)
        buffer.append(step)

    # batch = 256
    for i in trange(20000):
        cur_log = wmse.train()
        if i % 200 == 0:
            wandb.log(cur_log)
    torch.save(wmse.encoder.state_dict(), "models/conv_wmse.pt")

    # batch = 256
    for i in trange(20000):
        cur_log = idf.train()
        if i % 200 == 0:
            wandb.log(cur_log)
    torch.save(idf.encoder.state_dict(), "models/conv_idf.pt")

    # batch = 32 * 32
    for i in trange(5000):
        cur_log = cpc.train()
        if i % 50 == 0:
            wandb.log(cur_log)
    torch.save(cpc.model.conv.state_dict(), "models/conv_cpc.pt")
