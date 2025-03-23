# Selective Adam (with Triton acceleration)

`SelectiveAdam` is a custom optimizer based on pytorch [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). It is designed to **update only visible parameters** using a visibility mask. It leverages [Triton](https://github.com/openai/triton) to accelerate selective updates directly on the GPU, making it especially useful for dynamic scenarios like **3D Gaussian Splatting** or **sparse visibility in ray tracing**.

---

## Installation

```bash
git clone https://github.com/renekfr/SelectiveAdam.git
cd SelectiveAdam
pip install -r requirements.txt
```

---

## 🚀 Features

- 🎯 **Selective Updates**  
  Only updates parameters with non-zero gradients or defined by a visibility mask, keeping optimizer state consistent.

- ⚡ **Triton-Accelerated**  
  Uses a single compiled Triton kernel for all input sizes — avoids recompilation and runs fast on GPU.

- 📄 **License**
  MIT — free to use, modify, and integrate into personal or commercial projects.

---

- ⚠️ **Limits**
  This custom optimizer doesn't support "closure" or non zero gradient updates.
  Please refer to pytorch [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

## Results
### 3DGS Ray Tracer | 7K Iter | Truck Dataset | 1 M Splats
#### Adam
💡SSIM    : 0.855
💡PSNR    : 25.20

#### Selective Adam
💡SSIM    : 0.848
💡PSNR    : 24.99

## Discussion

🏗️ This custom optimizer is still in development — expect changes.
Let me know if you want to add badges (PyTorch version, Triton version, etc.) or changes you would add in this implementation.

