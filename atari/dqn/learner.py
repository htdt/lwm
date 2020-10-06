from copy import deepcopy
from functools import partial
import random
import torch

from common.optim import ParamOptim
from dqn.algo import get_td_error
from dqn.sampler import Sampler


def tde_to_prior(x, eta=0.9):
    return (eta * x.max(0).values + (1 - eta) * x.mean(0)).detach().cpu()


class Learner:
    def __init__(self, model, buffer, predictor, cfg):
        num_env = cfg["agent"]["actors"]
        model_t = deepcopy(model)
        model_t = model_t.cuda().eval()
        self.model, self.model_t = model, model_t
        self.buffer = buffer
        self.predictor = predictor
        self.optim = ParamOptim(params=model.parameters(), **cfg["optim"])

        self.batch_size = cfg["agent"]["batch_size"]
        self.unroll = cfg["agent"]["unroll"]
        self.unroll_prefix = (
            cfg["agent"]["burnin"]
            + cfg["agent"]["n_step"]
            + cfg["agent"]["frame_stack"]
            - 1
        )
        self.sample_steps = self.unroll_prefix + self.unroll
        self.hx_shift = cfg["agent"]["frame_stack"] - 1
        num_unrolls = (self.buffer.maxlen - self.unroll_prefix) // self.unroll

        if cfg["buffer"]["prior_exp"] > 0:
            self.sampler = Sampler(
                num_env=num_env,
                maxlen=num_unrolls,
                prior_exp=cfg["buffer"]["prior_exp"],
                importance_sampling_exp=cfg["buffer"]["importance_sampling_exp"],
            )
            self.s2b = torch.empty(num_unrolls, dtype=torch.long)
            self.hxs = torch.empty(num_unrolls, num_env, 512, device="cuda")
            self.hx_cursor = 0
        else:
            self.sampler = None

        self.target_tau = cfg["agent"]["target_tau"]
        self.td_error = partial(get_td_error, model=model, model_t=model_t, cfg=cfg)

    def _update_target(self):
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * (1.0 - self.target_tau) + s.data * self.target_tau)

    def append(self, step, hx, n_iter):
        self.buffer.append(step)

        if self.sampler is not None:
            if (n_iter + 1) % self.unroll == self.hx_shift:
                self.hxs[self.hx_cursor] = hx
                self.hx_cursor = (self.hx_cursor + 1) % len(self.hxs)

            k = n_iter - self.unroll_prefix
            if k > 0 and (k + 1) % self.unroll == 0:
                self.s2b[self.sampler.cursor] = self.buffer.cursor - 1
                x = self.buffer.get_recent(self.sample_steps)
                hx = self.hxs[self.sampler.cursor]
                with torch.no_grad():
                    loss, _ = self.td_error(x, hx)
                self.sampler.append(tde_to_prior(loss))

                if len(self.sampler) == self.sampler.maxlen:
                    idx_new = self.s2b[self.sampler.cursor - 1]
                    idx_old = self.s2b[self.sampler.cursor]
                    d = (idx_old - idx_new) % self.buffer.maxlen
                    assert self.unroll_prefix + self.unroll <= d
                    assert d < self.unroll_prefix + self.unroll * 2

    def loss_sampler(self, need_stat):
        idx0, idx1, weights = self.sampler.sample(self.batch_size)
        weights = weights.cuda()
        batch = self.buffer.query(self.s2b[idx0], idx1, self.sample_steps)
        hx = self.hxs[idx0, idx1]
        loss_pred, ri, _ = self.predictor.get_error(batch, update_stats=True)
        batch["reward"][1:] += ri
        td_error, log = self.td_error(batch, hx, need_stat=need_stat)
        self.sampler.update_prior(idx0, idx1, tde_to_prior(td_error))
        loss = td_error.pow(2).sum(0) * weights[..., None]
        loss_pred = loss_pred.sum(0) * weights[..., None]
        return loss, loss_pred, ri, log

    def loss_uniform(self, need_stat):
        if len(self.buffer) < self.buffer.maxlen:
            no_prev = set(range(self.sample_steps))
        else:
            no_prev = set(
                (self.buffer.cursor + i) % self.buffer.maxlen
                for i in range(self.sample_steps)
            )
        all_idx = list(set(range(len(self.buffer))) - no_prev)
        idx0 = torch.tensor(random.choices(all_idx, k=self.batch_size))
        idx1 = torch.tensor(
            random.choices(range(self.buffer.num_env), k=self.batch_size)
        )
        batch = self.buffer.query(idx0, idx1, self.sample_steps)
        loss_pred, ri, _ = self.predictor.get_error(batch, update_stats=True)
        batch["reward"][1:] += ri
        td_error, log = self.td_error(batch, None, need_stat=need_stat)
        loss = td_error.pow(2).sum(0)
        loss_pred = loss_pred.sum(0)
        return loss, loss_pred, ri, log

    def train(self, need_stat=True):
        loss_f = self.loss_uniform if self.sampler is None else self.loss_sampler
        loss, loss_pred, ri, log = loss_f(need_stat)
        self.optim.step(loss.mean())
        self.predictor.optim.step(loss_pred.mean())
        self._update_target()

        if need_stat:
            log.update(
                {
                    "ri_std": ri.std(),
                    "ri_mean": ri.mean(),
                    "ri_run_mean": self.predictor.ri_mean,
                    "ri_run_std": self.predictor.ri_std,
                    "loss_predictor": loss_pred.mean(),
                }
            )
            if self.sampler is not None:
                log.update(self.sampler.stats())
        return log
