import torch


def prepare_obs(obs, done, fstack):
    assert obs.dtype == torch.uint8
    assert obs.shape[2] == 1

    if fstack > 1:
        obs = stack_frames(obs, fstack)
        done_stacked = stack_frames(done, fstack)
        obs = obs * obs_mask(done_stacked)
    return obs.float() / 128 - 1


def stack_frames(x, stack=4):
    """
    Args:
        x: [steps + stack - 1, batch, 1, ...] - flat trajectory with prefix = stack - 1
    Returns:
        [steps, batch, stack, ...] - each step (dim 0) includes stack of frames (dim 2)
    """
    shape = (x.shape[0] - stack + 1, x.shape[1], stack, *x.shape[3:])
    y = torch.empty(shape, dtype=x.dtype, device=x.device)
    for i in range(stack):
        y[:, :, i] = x[i : shape[0] + i, :, 0]
    return y


def obs_mask(done):
    """
    mask to zero out observations in 4-frame stack when done = 1
    """
    mask = 1 - done[:, :, 1:]
    for i in reversed(range(mask.shape[2] - 1)):
        mask[:, :, i] *= mask[:, :, i + 1]
    mask = torch.cat([mask, torch.ones_like(mask[:, :, -1:])], 2)
    mask = mask[..., None, None]
    return mask
