import torch.nn.functional as F


def pad_dim(x, dim, size=1, value=0, left=False):
    p = [0] * len(x.shape) * 2
    p[-(dim + 1) * 2 + int(not left)] = size
    return F.pad(x, p, value=value)
