#Author: Laurent VIT
import torch
import triton
import triton.language as tl

@triton.jit
def selective_adam_visibility_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    weight_ptr,
    mask_ptr,
    N, D,
    step_size,
    beta1, beta2,
    _beta1, _beta2,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_offset = offsets // D
    mask_vals = tl.load(mask_ptr + mask_offset, mask=(mask_offset < N), other=0)
    weight_vals = tl.load(weight_ptr + mask_offset, mask=(mask_offset < N), other=1.0)

    param = tl.load(param_ptr + offsets, mask=(offsets < N*D))
    grad = tl.load(grad_ptr + offsets, mask=(offsets < N*D))
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=(offsets < N*D))
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=(offsets < N*D))

    new_exp_avg = tl.where(mask_vals, beta1 * exp_avg + _beta1 * grad, exp_avg)
    new_exp_avg_sq = tl.where(mask_vals, beta2 * exp_avg_sq + _beta2 * grad * grad, exp_avg_sq)
    update = step_size * weight_vals * new_exp_avg / tl.sqrt(new_exp_avg_sq + eps)
    new_param = tl.where(mask_vals, param - update, param)

    tl.store(param_ptr + offsets, new_param, mask=(offsets < N*D))
    tl.store(exp_avg_ptr + offsets, new_exp_avg, mask=(offsets < N*D))
    tl.store(exp_avg_sq_ptr + offsets, new_exp_avg_sq, mask=(offsets < N*D))


def selective_adam_step_triton(param, grad, exp_avg, exp_avg_sq, mask, weight, step, lr, beta1, beta2, eps):
    N = param.shape[0]
    D = param.shape[1] if param.dim() > 1 else 1

    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    step_size = lr * (bias2 ** 0.5) / bias1
    _beta1 = 1 - beta1
    _beta2 = 1 - beta2

    BLOCK_SIZE = 512
    grid = lambda meta: (triton.cdiv(N * D, meta['BLOCK_SIZE']),)

    selective_adam_visibility_kernel[grid](
        param, grad, exp_avg, exp_avg_sq,
        weight, mask,
        N, D, step_size,
        beta1, beta2, _beta1, _beta2, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )


class SelectiveAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, vis_beta=0.9, vis_smooth=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps, vis_beta=vis_beta, vis_smooth=vis_smooth)
        super().__init__(params, defaults)

    def step(self, visibility_mask=None):
        for group in self.param_groups:
            param = group["params"][0]
            if param.grad is None:
                continue

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["running_vis"] = torch.zeros(param.shape[0], device=param.device)

            state["step"] += 1

            weight = torch.ones(param.shape[0], device=param.device)
            if visibility_mask is not None:
                mask_idx = torch.nonzero(visibility_mask, as_tuple=True)[0]
                state["running_vis"][mask_idx] = (
                    group["vis_beta"] * state["running_vis"][mask_idx] + (1 - group["vis_beta"])
                )
                weight[mask_idx] = 1.0 / (state["running_vis"][mask_idx] + group["vis_smooth"])

            mask = visibility_mask if visibility_mask is not None else torch.ones(param.shape[0], device=param.device, dtype=torch.bool)

            selective_adam_step_triton(
                param.data, param.grad, 
                state["exp_avg"], state["exp_avg_sq"], 
                mask.to(torch.bool), weight,
                state["step"],
                group["lr"],
                group["betas"][0], group["betas"][1],
                group["eps"]
            )

        return [group['lr'] for group in self.param_groups]

    def zero_grad(self, set_to_none: bool = False, set_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
                else:
                    if set_grad:
                        p.grad = torch.zeros_like(p, dtype=p.dtype, device=p.device).contiguous()

