import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class DQN(nn.Module):
    def __init__(self, rnn_size, device="cuda"):
        super(DQN, self).__init__()
        self.device = device
        self.rnn_size = rnn_size

        self.encoder = nn.Sequential(nn.Linear(4 + 4 + 1, 32), nn.ReLU())
        # self.rnn = nn.GRUCell(32, self.rnn_size)
        self.rnn = nn.GRU(32, self.rnn_size)
        self.adv = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, 4, bias=False),
        )
        self.val = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, 1),
        )

    def forward(self, obs, action, reward, done, hx=None, only_hx=False):
        mask = (1 - done).float()
        a = one_hot(action[:, :, 0], 4).float() * mask
        r = reward * mask
        x = torch.cat([obs.float(), a, r], 2)

        steps, batch, *rest = x.shape
        x = x.view(steps * batch, *rest)
        x = self.encoder(x).view(steps, batch, 32)

        # y = torch.empty(steps, batch, self.rnn_size, device=self.device)
        # for i in range(steps):
        #     if hx is not None:
        #         hx *= mask[i]
        #     y[i] = hx = self.rnn(x[i], hx)
        # hx = hx.clone().detach()
        y, hx = self.rnn(x, hx)
        if only_hx:
            return hx

        y = y.view(steps * batch, self.rnn_size)
        adv, val = self.adv(y), self.val(y)
        q = val + adv - adv.mean(1, keepdim=True)
        q = q.view(steps, batch, 4)
        return q, hx
