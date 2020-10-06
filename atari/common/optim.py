from dataclasses import dataclass
from typing import List
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


@dataclass
class ParamOptim:
    params: List[torch.Tensor]
    lr: float = 1e-3
    eps: float = 1e-8
    clip_grad: float = None

    def __post_init__(self):
        self.optim = Adam(self.params, lr=self.lr, eps=self.eps)

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            clip_grad_norm_(self.params, self.clip_grad)
        self.optim.step()
        return loss
