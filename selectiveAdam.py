import math
import torch
from torch.optim.optimizer import Optimizer

class SelectiveAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, visibility=None):
        for group in self.param_groups:
            assert len(group["params"]) == 1, "more than one tensor in group"

            param = group["params"][0]
            if param.grad is None:
                continue

            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            state["step"] += 1
            step = state["step"]

            if visibility is None:
                mask = torch.ones_like(param, dtype=torch.bool)
            else:
                mask = visibility.bool()

            grad = param.grad[mask]
            if group["weight_decay"] != 0:
                grad = grad + group["weight_decay"] * param.data[mask]

            exp_avg[mask] = beta1 * exp_avg[mask] + (1 - beta1) * grad
            exp_avg_sq[mask] = beta2 * exp_avg_sq[mask] + (1 - beta2) * (grad ** 2)

            bias_1 = 1 - beta1 ** step
            bias_2 = 1 - beta2 ** step
            step_size = lr * (bias_2 ** 0.5) / bias_1

            update = exp_avg[mask] / exp_avg_sq[mask].sqrt().add_(eps)

            param.data[mask] += -step_size * update

        return None
