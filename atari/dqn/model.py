import torch
import torch.nn as nn
from torch.nn.functional import one_hot, relu
from dqn.prepare_obs import prepare_obs


def mnih_cnn(size_in, size_out):
    return nn.Sequential(
        nn.Conv2d(size_in, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, size_out),
    )


class DQN(nn.Module):
    def __init__(self, size_out, size_stack, device="cuda"):
        super(DQN, self).__init__()
        self.size_out = size_out
        self.size_stack = size_stack
        self.conv = mnih_cnn(size_stack, 512)
        self.rnn = nn.GRUCell(512 + 1 + size_out, 512)
        self.adv = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, size_out, bias=False)
        )
        self.val = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))
        self.device = device

    def forward(self, obs, action, reward, done, hx=None, only_hx=False):
        obs = prepare_obs(obs, done, self.size_stack)
        steps, batch, *img_shape = obs.shape
        obs = obs.view(steps * batch, *img_shape)
        x = relu(self.conv(obs))
        x = x.view(steps, batch, 512)

        pf = self.size_stack - 1
        mask = (1 - done[pf:]).float()
        a = one_hot(action[pf:, :, 0], self.size_out).float() * mask
        r = reward[pf:] * mask
        x = torch.cat([x, a, r], 2)

        y = torch.empty(steps, batch, 512, device=self.device)
        for i in range(steps):
            if hx is not None:
                hx *= mask[i]
            y[i] = hx = self.rnn(x[i], hx)
        hx = hx.clone().detach()
        if only_hx:
            return hx

        y = y.view(steps * batch, 512)
        adv, val = self.adv(y), self.val(y)
        q = val + adv - adv.mean(1, keepdim=True)
        q = q.view(steps, batch, self.size_out)
        return q, hx
