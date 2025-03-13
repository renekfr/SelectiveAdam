import math
import torch
from torch.optim.optimizer import Optimizer

class SelectiveAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, visibility):
        N = visibility.numel()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            state["step"] += 1
            step = state["step"]

            mask = visibility.bool()
            grad = param.grad[mask]

            exp_avg[mask] = beta1 * exp_avg[mask] + (1 - beta1) * grad
            exp_avg_sq[mask] = beta2 * exp_avg_sq[mask] + (1 - beta2) * grad * grad

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

            denom = exp_avg_sq[mask].sqrt().add_(eps)
            update = exp_avg[mask] / denom

            param.data[mask] += -step_size * update

        return None
