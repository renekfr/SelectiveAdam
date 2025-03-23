import torch
import triton
import triton.language as tl

@triton.jit
def selective_adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    mask_ptr,
    n_elements,
    step_size,
    beta1, beta2,
    one_minus_beta1, one_minus_beta2,
    eps,
    weight_decay,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_mask = offsets < n_elements

    mask = tl.load(mask_ptr + offsets, mask=offset_mask, other=0)
    param = tl.load(param_ptr + offsets, mask=offset_mask)
    grad = tl.load(grad_ptr + offsets, mask=offset_mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=offset_mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=offset_mask)

    if weight_decay > 0:
        grad = tl.where(mask, grad + weight_decay * param, grad)

    new_exp_avg = tl.where(mask, beta1 * exp_avg + one_minus_beta1 * grad, exp_avg)
    new_exp_avg_sq = tl.where(mask, beta2 * exp_avg_sq + one_minus_beta2 * grad * grad, exp_avg_sq)

    update = new_exp_avg / (tl.sqrt(new_exp_avg_sq) + eps)
    new_param = tl.where(mask, param - step_size * update, param)

    tl.store(exp_avg_ptr + offsets, new_exp_avg, mask=offset_mask)
    tl.store(exp_avg_sq_ptr + offsets, new_exp_avg_sq, mask=offset_mask)
    tl.store(param_ptr + offsets, new_param, mask=offset_mask)


def selective_adam_step_triton(param, grad, exp_avg, exp_avg_sq, mask, step, lr, beta1, beta2, eps, weight_decay):
    n_elements = param.numel()
    
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
    one_minus_beta1 = 1 - beta1
    one_minus_beta2 = 1 - beta2
    
    param_flat = param.view(-1)
    grad_flat = grad.view(-1)
    exp_avg_flat = exp_avg.view(-1)
    exp_avg_sq_flat = exp_avg_sq.view(-1)
    mask_flat = mask.unsqueeze(1).expand_as(param).contiguous().view(-1).to(torch.uint8)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    selective_adam_kernel[grid](
        param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat, mask_flat,
        n_elements, step_size, beta1, beta2, one_minus_beta1, one_minus_beta2, eps, weight_decay,
        BLOCK_SIZE=BLOCK_SIZE
    )

class SelectiveAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None, visibility_mask=None):
        mask = visibility_mask
        for group in self.param_groups:
            assert len(group["params"]) == 1, "Chaque groupe doit contenir un seul tenseur."
            param = group["params"][0]
            if param.grad is None:
                continue

            state = self.state[param]

            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]

            state = self.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            
            state["step"] += 1
            step = state["step"]
            
            if visibility_mask is None:
                mask = (torch.abs(param.grad) > eps).any(dim=1)
            
            selective_adam_step_triton(
                param, param.grad, 
                state["exp_avg"], state["exp_avg_sq"], 
                mask, 
                step, 
                lr, 
                beta1, 
                beta2, 
                eps, 
                group["weight_decay"]
            )

    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
