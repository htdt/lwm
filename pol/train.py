import argparse
from tqdm import trange
import torch
import wandb

from dqn.buffer import Buffer
from common.load_cfg import load_cfg
from env import make_vec_envs
from dqn import actor_iter, Learner, DQN
from predictor import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--add_ri", action="store_true")
    p = parser.parse_args()
    cfg = load_cfg("default")
    cfg.update(vars(p))
    cfg["env"] = "pol"
    wandb.init(project="lwm", config=cfg)

    num_env = cfg["agent"]["actors"]
    envs = make_vec_envs(
        num=num_env,
        size=cfg["size"],
        max_ep_len=cfg["train"]["max_ep_len"],
    )
    buffer = Buffer(
        num_env=num_env,
        maxlen=int(cfg["buffer"]["size"] / num_env),
        obs_shape=(4,),
        device=cfg["buffer"]["device"],
    )
    model = DQN(cfg["agent"]["rnn_size"]).cuda().train()
    pred = Predictor(buffer, cfg)
    learner = Learner(model, buffer, pred, cfg)
    eps = cfg["agent"].get("eps")
    actor = actor_iter(envs, model, pred, cfg["buffer"]["warmup"], eps=eps)

    start_train = int(cfg["buffer"]["warmup"] / num_env)
    log_every = cfg["train"]["log_every"]
    train_every = cfg["train"]["learner_every"]

    count = trange(int(cfg["train"]["frames"] / num_env), smoothing=0.05)
    for n_iter in count:
        full_step = buffer.get_recent(2)
        step, hx, log = actor.send(full_step)
        buffer.append(step)

        if n_iter == start_train and cfg["add_ri"]:
            for i in trange(1000):
                cur_log = pred.train()
                if i % 100 == 0:
                    wandb.log(cur_log)
            pred.save()

        if n_iter > start_train and (n_iter + 1) % train_every == 0:
            cur_log = learner.train()
            if (n_iter + 1) % log_every < train_every:
                log.update(cur_log)

        if len(log):
            wandb.log({"frame": n_iter * num_env, **log})

    torch.save(model.state_dict(), "models/dqn.pt")
    pred.save()
