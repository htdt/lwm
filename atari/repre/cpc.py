import random
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, one_hot, relu

from dqn.model import mnih_cnn
from dqn.prepare_obs import prepare_obs
from common.optim import ParamOptim
from dqn.buffer import Buffer


class CPCModel(nn.Module):
    def __init__(self, num_action, size_emb, size_stack, device="cuda"):
        super(CPCModel, self).__init__()
        self.size_emb = size_emb
        self.size_stack = size_stack
        self.num_action = num_action
        self.device = device

        self.conv = mnih_cnn(size_stack, size_emb)
        self.rnn = nn.GRUCell(num_action + size_emb, 512)
        self.fc = nn.Linear(512, size_emb)

    def forward(self, obs, action, done, hx=None, only_hx=False):
        obs = prepare_obs(obs, done, self.size_stack)
        steps, batch, *img_shape = obs.shape
        obs = obs.view(steps * batch, *img_shape)
        z = self.conv(obs).view(steps, batch, self.size_emb)

        pf = self.size_stack - 1
        mask = (1 - done[pf:]).float()
        a = one_hot(action[:, :, 0], self.num_action).float()
        x = torch.cat([relu(z[:-1]), a], 2)

        steps -= 1
        y = torch.empty(steps, batch, 512, device=self.device)
        for i in range(steps):
            if hx is not None:
                hx *= mask[i]
            y[i] = hx = self.rnn(x[i], hx)
        hx = hx.clone().detach()
        if only_hx:
            return hx

        y = y.view(steps * batch, 512)
        z_pred = self.fc(y).view(steps, batch, self.size_emb)
        return z[1:], z_pred, hx


@dataclass
class CPC:
    buffer: Buffer
    num_action: int
    frame_stack: int = 1
    batch_size: int = 32
    unroll: int = 32
    emb_size: int = 32
    lr: float = 5e-4
    device: str = "cuda"

    def __post_init__(self):
        self.model = CPCModel(self.num_action, self.emb_size, self.frame_stack)
        self.model = self.model.train().to(self.device)
        self.optim = ParamOptim(params=self.model.parameters(), lr=self.lr)
        self.target = torch.arange(self.batch_size * self.unroll).to(self.device)

    def train(self):
        # burnin = 2, fstack = 4, unroll = 2
        # idx 0 1 2 3 4 5 6 7
        # bin p p p b b b
        #             a a
        #             hx
        # rol     p p p o o o
        #                 a a

        sample_steps = self.frame_stack + self.unroll

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

        obs = batch["obs"]
        action = batch["action"][self.frame_stack :]
        done = batch["done"]
        z, z_pred, _ = self.model(obs, action, done)

        size = self.batch_size * self.unroll
        z = z.view(size, self.emb_size)
        z_pred = z_pred.view(size, self.emb_size)
        logits = z @ z_pred.t()
        loss = cross_entropy(logits, self.target)
        acc = (logits.argmax(-1) == self.target).float().mean()
        self.optim.step(loss)
        return {"loss_cpc": loss.item(), "acc_cpc": acc}

    def load(self):
        cp = torch.load("models/cpc.pt", map_location=self.device)
        self.model.load_state_dict(cp)

    def save(self):
        torch.save(self.model.state_dict(), "models/cpc.pt")
