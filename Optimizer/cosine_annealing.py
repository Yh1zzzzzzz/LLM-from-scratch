import torch
import math
def cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate  + 0.5 * (max_learning_rate - min_learning_rate) \
                *(1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        