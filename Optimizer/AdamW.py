from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the Learning rate.
            wight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                Momentum1 = state.get("momentum1", torch.zeros_like(p.data))
                Momentum2 = state.get("momentum2", torch.zeros_like(p.data))
                t = state.get("t", 0) 

                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                state["t"] = t + 1 # Increment iteration number.
                state["momentum1"] = beta1 * Momentum1 + (1 - beta1) * grad
                state["momentum2"] = beta2 * Momentum2 + (1 - beta2) * grad ** 2
                Momentum1 = state.get("momentum1", torch.zeros_like(p.data))
                Momentum2 = state.get("momentum2", torch.zeros_like(p.data))
                alpha_t = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))

                p.data -= alpha_t * Momentum1 / torch.sqrt(Momentum2 + eps)  # Update weight tensor in-place.
                p.data -= lr * p.data *  wight_decay # Apply weight decay

        return loss
