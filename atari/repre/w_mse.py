import random
import torch
from common.optim import ParamOptim
from repre.whitening import Whitening2d
from dqn.model import mnih_cnn
from dqn.prepare_obs import prepare_obs


class WMSE:
    batch_size: int = 256
    lr: float = 5e-4

    def __init__(self, buffer, cfg, device="cuda"):
        self.device = device
        self.buffer = buffer
        self.emb_size = cfg["w_mse"]["emb_size"]
        self.temporal_shift = cfg["w_mse"]["temporal_shift"]
        self.spatial_shift = cfg["w_mse"]["spatial_shift"]
        self.frame_stack = cfg["w_mse"]["frame_stack"]

        self.encoder = mnih_cnn(self.frame_stack, self.emb_size)
        self.encoder = self.encoder.to(self.device).train()
        self.optim = ParamOptim(lr=self.lr, params=self.encoder.parameters())
        self.w = Whitening2d(self.emb_size, track_running_stats=False)

    def load(self):
        cp = torch.load("models/w_mse.pt", map_location=self.device)
        self.encoder.load_state_dict(cp)

    def save(self):
        torch.save(self.encoder.state_dict(), "models/w_mse.pt")

    def train(self):
        def spatial():
            return random.randrange(-self.spatial_shift, self.spatial_shift + 1)

        sample_steps = self.frame_stack + self.temporal_shift
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
        idx2 = random.choices(range(1, self.temporal_shift + 1), k=self.batch_size)
        x0 = obs[0]
        x1 = obs[idx2, range(self.batch_size)]

        if self.spatial_shift > 0:
            for n in range(self.batch_size):
                for x in [x0, x1]:
                    shifts = spatial(), spatial()
                    x[n] = torch.roll(x[n], shifts=shifts, dims=(-2, -1))

        x0, x1 = self.encoder(x0), self.encoder(x1)
        z = self.w(torch.cat([x0, x1], dim=0))
        loss = (z[:len(x0)] - z[len(x0):]).pow(2).mean()
        self.optim.step(loss)
        return {"loss_wmse": loss.item()}
