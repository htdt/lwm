import random
from itertools import chain
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.functional import cross_entropy, relu
from common.optim import ParamOptim
from dqn.model import mnih_cnn
from dqn.buffer import Buffer
from dqn.prepare_obs import prepare_obs


@dataclass
class IDF:
    buffer: Buffer
    num_action: int
    emb_size: int = 32
    batch_size: int = 256
    lr: float = 5e-4
    frame_stack: int = 1
    device: str = "cuda"

    def __post_init__(self):
        self.encoder = mnih_cnn(self.frame_stack, self.emb_size)
        self.encoder = self.encoder.to(self.device).train()
        self.clf = nn.Sequential(
            nn.Linear(self.emb_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_action),
        )
        self.clf = self.clf.to(self.device).train()
        params = chain(self.encoder.parameters(), self.clf.parameters())
        self.optim = ParamOptim(lr=self.lr, params=params)

    def train(self):
        # 0 1 2 3 4
        # p p p o o
        #         a

        sample_steps = self.frame_stack + 1
        if len(self.buffer) < self.buffer.maxlen:
            no_prev = set(range(sample_steps))
        else:
            no_prev = set(
                (self.buffer.cursor + i) % self.buffer.maxlen
                for i in range(sample_steps)
            )
        all_idx = list(set(range(len(self.buffer))) - no_prev)
        idx0 = torch.tensor(random.choices(all_idx, k=self.batch_size))
        idx1 = torch.tensor(
            random.choices(range(self.buffer.num_env), k=self.batch_size)
        )
        batch = self.buffer.query(idx0, idx1, sample_steps)
        obs = prepare_obs(batch["obs"], batch["done"], self.frame_stack)
        action = batch["action"][-1, :, 0]

        x0, x1 = self.encoder(obs[0]), self.encoder(obs[1])
        x = torch.cat([x0, x1], dim=-1)
        y = self.clf(relu(x))
        loss_idf = cross_entropy(y, action)
        acc_idf = (y.argmax(-1) == action).float().mean()

        self.optim.step(loss_idf)
        return {"loss_idf": loss_idf, "acc_idf": acc_idf}

    def load(self):
        cp = torch.load("models/idf.pt", map_location=self.device)
        self.encoder.load_state_dict(cp[0])
        self.clf.load_state_dict(cp[1])

    def save(self):
        cp = [self.encoder.state_dict(), self.clf.state_dict()]
        torch.save(cp, "models/idf.pt")
