from time import time
from collections import defaultdict
import numpy as np


def timer_log(num_iter=1000):
    log = {}
    mean_t = defaultdict(list)
    t = mark = None
    while True:
        prev_t, prev_mark = t, mark
        mark = yield log
        t = time()
        if prev_mark is not None:
            mean_t[prev_mark].append(t - prev_t)

        if mark is None and len(list(mean_t.values())[0]) >= num_iter:
            log = {"time/" + k: np.mean(v) * 1000 for k, v in mean_t.items()}
            mean_t = defaultdict(list)
        else:
            log = {}
