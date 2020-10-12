import random
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from common.optim import ParamOptim


class PredictorModel(nn.Module):
    def __init__(self, rnn_size):
        super(PredictorModel, self).__init__()
        self.rnn_size = rnn_size
        self.encoder = nn.Sequential(nn.Linear(4 + 4, 32), nn.ReLU())
        # self.rnn = nn.GRUCell(32, self.rnn_size)
        self.rnn = nn.GRU(32, self.rnn_size)
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, 4),
            nn.Sigmoid(),
        )

    def forward(self, z, action, done, hx=None):
        unroll, batch, emb_size = z.shape
        a = one_hot(action[:, :, 0], 4).float()
        z = torch.cat([z, a], dim=2)
        z = self.encoder(z.view(unroll * batch, 4 + 4))
        z = z.view(unroll, batch, 32)

        # mask = 1 - done.float()
        # x = torch.empty(unroll, batch, self.rnn_size, device=z.device)
        # for i in range(unroll):
        #     if hx is not None:
        #         hx *= mask[i]
        #     x[i] = hx = self.rnn(z[i], hx)
        # hx = hx.clone().detach()

        x, hx = self.rnn(z, hx)

        x = self.fc(x.view(unroll * batch, self.rnn_size))
        z_pred = x.view(unroll, batch, 4)
        return z_pred, hx


class Predictor:
    def __init__(self, buffer, cfg, device="cuda"):
        self.device = device
        self.buffer = buffer

        self.model = PredictorModel(cfg["agent"]["rnn_size"])
        self.model = self.model.to(device).train()
        lr = cfg["self_sup"]["lr"]
        self.optim = ParamOptim(params=self.model.parameters(), lr=lr)
        self.ri_mean = self.ri_std = None
        self.ri_momentum = cfg["self_sup"]["ri_momentum"]

    def get_error(self, batch, hx=None, update_stats=False):
        z = batch["obs"].float()
        action = batch["action"][1:]
        done = batch["done"][:-1]
        z_pred, hx = self.model(z[:-1], action, done, hx)
        err = (z[1:] - z_pred).pow(2).mean(2)

        ri = err.detach()
        if update_stats:
            if self.ri_mean is None:
                self.ri_mean = ri.mean()
                self.ri_std = ri.std()
            else:
                m = self.ri_momentum
                self.ri_mean = m * self.ri_mean + (1 - m) * ri.mean()
                self.ri_std = m * self.ri_std + (1 - m) * ri.std()
        if self.ri_mean is not None:
            ri = (ri[..., None] - self.ri_mean) / self.ri_std
        else:
            ri = 0
        return err.mean(), ri, hx

    def train(self):
        # this function is used only for pretrain, main training loop is in dqn learner
        batch_size = 64
        sample_steps = 100
        if len(self.buffer) < self.buffer.maxlen:
            no_prev = set(range(sample_steps))
        else:
            no_prev = set(
                (self.buffer.cursor + i) % self.buffer.maxlen
                for i in range(sample_steps)
            )
        all_idx = list(set(range(len(self.buffer))) - no_prev)
        idx0 = torch.tensor(random.choices(all_idx, k=batch_size))
        idx1 = torch.tensor(random.choices(range(self.buffer.num_env), k=batch_size))
        batch = self.buffer.query(idx0, idx1, sample_steps)
        loss = self.get_error(batch, update_stats=True)[0]
        self.optim.step(loss)
        return {"loss_predictor": loss.item()}

    def load(self):
        cp = torch.load("models/predictor.pt", map_location=self.device)
        self.ri_mean, self.ri_std, model = cp
        self.model.load_state_dict(model)

    def save(self):
        data = [self.ri_mean, self.ri_std, self.model.state_dict()]
        torch.save(data, "models/predictor.pt")
