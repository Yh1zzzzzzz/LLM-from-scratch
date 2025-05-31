import torch
def gradient_clipping(parameters, max_l2_norm: float) -> None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
