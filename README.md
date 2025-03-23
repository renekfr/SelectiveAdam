# Selective Adam (with Triton acceleration)

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

## 🚀 Features

- 🎯 **Selective Updates**  
  Only updates parameters with non-zero gradients or defined by a visibility mask, keeping optimizer state consistent.

- ⚡ **Triton-Accelerated**  
  Uses a single compiled Triton kernel for all input sizes — avoids recompilation and runs fast on GPU.

- 📄 **License**
  MIT — free to use, modify, and integrate into personal or commercial projects.

---

- ⚠️ **Limits**
  
This custom optimizer does not support closure.
The visibility mask should be of shape [N].
If no mask is provided, only gradients strictly different from zero will be updated — others will remain unchanged.
  
For a standard implementation, refer to [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

## Results
### 3DGS Ray Tracer | 7K Iter | Truck Dataset
#### Adam
💡SSIM    : 0.855
💡PSNR    : 25.20

#### Selective Adam
💡SSIM    : 0.857
💡PSNR    : 25.19

## Discussion

🏗️ This custom optimizer is still in development — expect changes.
Let me know if you want to add badges (PyTorch version, Triton version, etc.) or changes you would add in this implementation.

