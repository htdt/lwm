from dataclasses import dataclass
import torch


@dataclass
class Sampler:
    num_env: int
    maxlen: int
    prior_exp: float = 0.9
    importance_sampling_exp: float = 0.6

    def __post_init__(self):
        self.cursor = self._size = 0
        self._prior = torch.empty(self.maxlen, self.num_env, 1)

    def append(self, prior):
        assert prior.shape == self._prior.shape[1:]
        self._prior[self.cursor] = prior
        self.cursor = (self.cursor + 1) % self.maxlen
        self._size = min(self.maxlen, self._size + 1)

    def sample(self, batch_size):
        p = self._prior[: len(self)].view(-1) ** self.prior_exp
        idx_flat = p.multinomial(batch_size, replacement=True)
        weights = p[idx_flat] ** -self.importance_sampling_exp
        weights /= weights.max()
        idx0 = idx_flat // self.num_env
        idx1 = idx_flat % self.num_env
        return idx0, idx1, weights

    def update_prior(self, idx0, idx1, prior):
        self._prior[idx0, idx1] = prior

    def stats(self):
        x = self._prior[: len(self)]
        return {
            "prior/mean": x.mean(),
            "prior/std": x.std(),
            "prior/max": x.max(),
        }

    def __len__(self):
        return self._size
