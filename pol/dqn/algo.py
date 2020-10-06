import torch


def vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def inv_vf_rescaling(x):
    eps = 1e-3
    return torch.sign(x) * (
        (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps))) - 1) / (2 * eps)) ** 2
        - 1
    )


def get_td_error(batch, hx_start, model, model_t, cfg):
    burnin = cfg["agent"]["burnin"]
    gamma = cfg["agent"]["gamma"]

    if burnin > 0:
        with torch.no_grad():
            hx = model(**batch[:burnin], hx=hx_start, only_hx=True)
            hx_target = model_t(**batch[: burnin + 1], hx=hx_start, only_hx=True)
    else:
        hx = hx_target = None

    qs, _ = model(**batch[burnin:], hx=hx)

    with torch.no_grad():
        qs_target, _ = model_t(**batch[burnin + 1 :], hx=hx_target)

    action = batch["action"][burnin + 1 :]
    reward = batch["reward"][burnin + 1 :]
    done = batch["done"][burnin + 1 :].float()

    q = qs[:-1].gather(2, action)
    ns_action = qs[1:].argmax(2)[..., None].detach()
    next_q = qs_target.gather(2, ns_action)
    next_q = inv_vf_rescaling(next_q)
    target_q = next_q * gamma * (1 - done) + reward
    target_q = vf_rescaling(target_q)
    td_error = (q - target_q).abs()

    log = {
        "loss": td_error.mean().item(),
        "q_mean": qs.mean().item(),
        "q_std": qs.std().item(),
    }
    return td_error, log
