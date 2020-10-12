import random
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from common.optim import ParamOptim
from dqn.prepare_obs import prepare_obs


class PredictorModel(nn.Module):
    def __init__(self, num_action, fstack, emb_size, rnn_size):
        super(PredictorModel, self).__init__()
        self.fstack = fstack
        self.num_action = num_action
        self.rnn_size = rnn_size
        self.emb_fc = nn.Linear(emb_size + num_action, 128)
        self.rnn = nn.GRUCell(128, rnn_size)
        self.fc = nn.Sequential(
            nn.Linear(rnn_size, rnn_size), nn.ReLU(), nn.Linear(rnn_size, emb_size),
        )

    def forward(self, z, action, done, hx=None):
        unroll, batch, emb_size = z.shape
        a = one_hot(action[:, :, 0], self.num_action).float()
        z = torch.cat([z, a], dim=2)
        z = self.emb_fc(z.view(unroll * batch, (emb_size + self.num_action)))
        z = z.view(unroll, batch, 128)

        mask = 1 - done.float()
        x = torch.empty(unroll, batch, self.rnn_size, device=z.device)
        for i in range(unroll):
            if hx is not None:
                hx *= mask[i]
            x[i] = hx = self.rnn(z[i], hx)
        hx = hx.clone().detach()
        x = self.fc(x.view(unroll * batch, self.rnn_size))
        z_pred = x.view(unroll, batch, emb_size)
        return z_pred, hx


class Predictor:
    def __init__(self, buffer, encoder, num_action, cfg, device="cuda"):
        self.device = device
        self.buffer = buffer
        self.encoder = encoder

        self.frame_stack = cfg["w_mse"]["frame_stack"]
        self.emb_size = cfg["w_mse"]["emb_size"]
        self.rnn_size = cfg["w_mse"]["rnn_size"]

        self.model = PredictorModel(
            num_action, self.frame_stack, self.emb_size, self.rnn_size
        )
        self.model = self.model.to(device).train()
        lr = cfg["w_mse"]["lr"]
        self.optim = ParamOptim(params=self.model.parameters(), lr=lr)
        self.ri_mean = self.ri_std = None
        self.ri_momentum = cfg["w_mse"]["ri_momentum"]
        self.ri_clamp = cfg["w_mse"].get("ri_clamp")
        self.ri_scale = cfg["ri_scale"]

    def get_error(self, batch, hx=None, update_stats=False):
        # p p p o o o
        #         a a
        obs = prepare_obs(batch["obs"], batch["done"], self.frame_stack)
        steps, batch_size, *obs_shape = obs.shape
        obs = obs.view(batch_size * steps, *obs_shape)
        with torch.no_grad():
            z = self.encoder(obs)
        z = z.view(steps, batch_size, self.emb_size)

        action = batch["action"][self.frame_stack :]
        done = batch["done"][self.frame_stack - 1 : -1]
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
            if self.ri_clamp is not None:
                ri.clamp_(-self.ri_clamp, self.ri_clamp)
            ri *= self.ri_scale
        else:
            ri = 0
        return err, ri, hx

    def train(self):
        # this function is used only for pretrain, main training loop is in dqn learner
        batch_size = 16
        sample_steps = self.frame_stack - 1 + 100
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
        er = self.get_error(batch, update_stats=True)[0]
        loss = er.sum(0).mean()
        self.optim.step(loss)
        return {"loss_predictor": loss.item()}

    def load(self):
        cp = torch.load("models/predictor.pt", map_location=self.device)
        self.ri_mean, self.ri_std, model = cp
        self.model.load_state_dict(model)

    def save(self):
        data = [self.ri_mean, self.ri_std, self.model.state_dict()]
        torch.save(data, "models/predictor.pt")
