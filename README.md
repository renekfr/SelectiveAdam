# Selective Adam

Selective Adam is an optimizer based on Adam that enables selective parameter updates by applying a visibility mask to the gradients. Initially designed for 3DGS, it can be used in any project that requires targeted parameter updates.

## Features

- **Selective Update**: Only the portions of the parameters corresponding to "visibility" (as defined by a mask) are updated. Either all parameters are updated!
- **Adam-Inspired**: Leverages adaptive learning rates through exponential moving averages and bias corrections.
- **Easy Integration**: Built to work seamlessly with PyTorch, making it simple to integrate into your projects.
- **Open Source**: Distributed under the MIT license, you are free to use and modify it as needed.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/renekfr/SelectiveAdam.git
cd SelectiveAdam
```

## Results
### 3DGS Ray Tracer | 7K Iter | Truck Dataset | 1 M Splats
#### Adam
💡SSIM    : 0.855
💡PSNR    : 25.20

#### Selective Adam
💡SSIM    : 0.848
💡PSNR    : 24.99

## Discussion

I did some test and lost a bit of PSNR / SSIM using selective Adam.
I believe that it must come from my visibility mask, not the optimizer.

An other point could be the convergence which could be slower at first, so more test will be done soon! :)

