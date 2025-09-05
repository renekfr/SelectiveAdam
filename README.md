# Selective Adam (with Triton acceleration) v0.1

`SelectiveAdam` is a custom optimizer based on pytorch [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). It is designed to **update only visible parameters** using a visibility mask. It leverages [Triton](https://github.com/openai/triton) to accelerate selective updates directly on the GPU, making it especially useful for dynamic scenarios like **3D Gaussian Splatting**.

---

## Installation

```bash
git clone https://github.com/renekfr/SelectiveAdam.git
cd SelectiveAdam
pip install -r requirements.txt
```

---

## Usage

The `visibility_mask` must be of shape `[N]`.
If not provided, a default mask is computed based on non-zero gradients — which is kinda nice 🙂

```python
optimizer = SelectiveAdam(params, eps=1e-15, betas=(0.9, 0.999))
optimizer.zero_grad()
optimizer.step(visibility_mask=visibility_mask)
```

---

## ⚠️ Limits
  
This custom optimizer does not support closure.
The visibility mask should be of shape [N].
If no mask is provided, only gradients strictly different from zero will be updated — others will remain unchanged.
  
For a standard implementation, refer to [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

## Results


### 3DGS Ray Tracer (Ours) | 7K Iter | Truck Dataset | 2M Splats
#### Selective Adam (Ours)
💡SSIM    :  0.860
💡PSNR    : 25.390
⌛Time    :  05:15

### Adam
💡SSIM    :  0.857
💡PSNR    : 25.209
⌛Time    :  05:48

#### Gsplat Adam
💡SSIM    :  0.851
💡PSNR    : 25.060
⌛Time    :   6:47

## Discussion

This custom optimizer is still in development — expect changes.
Let me know if you would add changes in this implementation.

I really wish I had found an implementation like this on GitHub. so I hope it helps some of you out there!

