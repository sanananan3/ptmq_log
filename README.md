# ptmq-pytorch

PyTorch implementation of "PTMQ: Post-training Multi-Bit Quantization of Neural Networks (Xu et al., AAAI 2024)"

[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29553)

---

## Getting Started

Running PTMQ

```bash
python run_ptmq.py --config configs/[config_file].yaml
```

Create your own configuration file in the `configs` directory.

---

## Useful Commands

### Initial Setup for Cloud GPUs ([runpod.io](https://runpod.io?ref=9t3u4v13))

```bash
# create virtual environment and install dependencies
python -m venv iris
source iris/bin/activate
pip install --upgrade pip
pip install torch torchvision easydict PyYAML scipy gdown

# download resnet18 weights
cd ../../ptmq-pytorch
python
import torch
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
torch.save(resnet18.state_dict(), "resnet18_weights.pth")
```

### Downloading Datasets

Use `imagenet-mini/train` for calibration, and `imagenet/val` (the full ImageNet1K validation set) for evaluation.

#### Mini-ImageNet from Kaggle

```bash
pip install kaggle
cd ~/dev
mkdir -p ~/dev/kaggle # add kaggle.json with {"username":"xxx","key":"xxx"} here
chmod 600 ~/dev/kaggle/kaggle.json
kaggle datasets download -d ifigotin/imagenetmini-1000
mkdir ~/dev/imagenet/train
apt
unzip ~/dev/imagenetmini-1000.zip -d ~/dev
```

#### ImageNet Validation Dataset

```bash
# download imagenet validation dataset (from public google drive)
mkdir -p imagenet/val
cd imagenet/val
gdown https://drive.google.com/uc?id=11omFedOvjslBRMFc-lrM3n2t0xP99FXB -O ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### `wandb` setup

```bash
pip install wandb
wandb login # enter API key
```

---

## Overview

PTMQ (post-training multi-bit quantization) is a post-training quantization method that performs **block-level activation quantization** with multiple bit-widths.

In a sense, it can be viewed as a knowledge distillation method, where the higher-precision features are used to guide the quantization of lower-precision features.

For starters, we can think of the target inference model to be W3A3. PTMQ has separate strategies in order to ensure that the weights and activations are better quantized. These are all employed during the block reconstruction phase, and weights and activations are optimized with a reconstruction loss (GD loss) and round loss, respectively.

**Weights** are better quantized using rounding optimization via AdaRound (Nagel et al., ICML 2020). This is done by minimizing the quantization error of the weights.

**Activations** are better quantized by using a multi-bit feature mixer, which is 3 separate quantizers for low, medium, and high bit-widths. We learn activation step sizes (Esser et al., ICLR 2020) to minimize the activation quantization error, via a group-wise distillation loss.

The novelty of this model is that through block reconstruction, we can quickly and efficiently quantize a full-precision model to multiple bit-widths, which can be flexibly be deployed based on the given hardware constraints in real-time.

---

## Reproducing Results

### ResNet

- Paper results
  - ResNet18 W3 ImageNet1K - top1 = 64.92 (t=100mins on Nvidia 3090)

- Experiments
  - reconstruciton batch size = 32
  - GPU: NVIDIA A40
  - imagenet1k_val dataset is used for calibration, reconstruction, and evaluation
  - `lmh`: bit precision for low, medium, and high bit-widths for the multi-bit feature mixer (MFM)

**`experiment log`**

- ❓ptmq implementation not showing results - debug (loss, accuracy etc.)
  - increasing lmh (345 -> 468) 
  - lmh = 888 for sanity check
    - 🔥**need to scale `recon_iters` for further experiments, but implementation seems to have bug (problem with loss function?)**
    - 🔥if we use gd_loss as reconstruction loss for blocks, what do we use for layers (MSE for first and last layers is probably right)
    - recheck MFM to see if it is implemented correctly
      - 🔥total_loss = recon_loss + round_loss
        - 🔥gd_loss: 0.008, loss_fp: 0.000, loss_hm: 0.004, loss_hl: 0.003
          - ~~perhaps gamma vals are too low?~~
          - gd_loss returns vals (fp_loss=0 always btw), but gd_loss has no updates?
            - gd_loss IS updating - maybe oscillation due to learning rate?
            - ~~perhaps gd_loss/recon_loss is too small, RELATIVE TO round_loss? ~~
            - ✅`f_low`, `f_med`, `f_high` are calculated during forward pass, but gd_loss uses a new calculated version of these values
              - 🔥eval results are good, but round_loss seems to accumulate as layers progress
      - should we add extra quantizers to modules?
      - hyperparameters
        - $\lambda_1 + \lambda_2 + \lambda_3 = 1$ must be true
        - $\gamma_1 + \gamma_2 + \gamma_3 =1$ is NOT true!
- ✅refactoring of quantizers?
  - ✅essence of ptmq --> to apply ptq to a single model, and be able to call inference on a variety of bitwidths
  - ✅model should be set so that we can set different bitwidths at inference time
  - ✅how do we set a custom bitwidth to propagate through the entire model? --> DONE (set inference mode for modules)

| repo (commit) | model | quant | recon_iters | recon_time (mins) | top1 | top5 |
| ------------- | ----- | ----- | ----------- | ---- | ---- | ---- |
| **BASELINE** | ResNet-18 | W3A_, lmh=? | ? | 100 (3090) | 64.92 | - |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/c415bc86a38e123aeb29ef1a505f293a3ae76c1c) | ResNet-18 | W3A8, lmh=345 | 100 | 2 | 35.09 | 59.71 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/c415bc86a38e123aeb29ef1a505f293a3ae76c1c) | ResNet-18 | W3A8, lmh=345 | 1000 | 20 | 35.70 | 60.41 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/c415bc86a38e123aeb29ef1a505f293a3ae76c1c) | ResNet-18 | W3A8, lmh=345 | 5000 | 80 | 35.54 | 60.30 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/c415bc86a38e123aeb29ef1a505f293a3ae76c1c) | ResNet-18 | W3A8, lmh=468 | 100 | 2 | 35.57 | 60.08 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/c415bc86a38e123aeb29ef1a505f293a3ae76c1c) | ResNet-18 | W3A8, lmh=888 | 100 | 2 | 35.78 | 60.40 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/022fc34d5911393edf71c14ea63b83d4247e193c) | ResNet-18 | W3A8, lmh=345 | 100 | 2 | 66.10 | 87.04 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/022fc34d5911393edf71c14ea63b83d4247e193c) | ResNet-18 | W3A8, lmh=345 | 1000 | 2 | 66.93 | 87.50 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/022fc34d5911393edf71c14ea63b83d4247e193c) | ResNet-18 | W3A3, lmh=468 | 1000 | 2 | 62.59 | 84.72 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/022fc34d5911393edf71c14ea63b83d4247e193c) | ResNet-18 | W3A3, lmh=246 | 1000 | 2 | 61.85 | 84.16 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/022fc34d5911393edf71c14ea63b83d4247e193c) | ResNet-18 | W3A3, lmh=246 | 100 | 2 | 56.70 | 80.15 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/022fc34d5911393edf71c14ea63b83d4247e193c) | ResNet-18 | W4A4, lmh=246 | 100 | 2 | 67.22 | 87.69 |
| [ptmq-pytorch](https://github.com/d7chong/ptmq-pytorch/tree/022fc34d5911393edf71c14ea63b83d4247e193c) | ResNet-18 | W5A5, lmh=246 | 100 | 2 | 68.63 | 88.37 |



### Vision Transformer

- Paper results
  - ViT-S/224/16 W8A8 - top1 = 78.16%

---

## Implementation

- we build off of the QDrop source code
- key differences from original source code
  - `quant_module.py`: add multi-bit feature for forward (`QuantLinear`, `QuantConv2D`, `QuantBlock`)
  - `ptmq_recon.py` reconstruction for ptmq, with MFM and GD loss (MSE for layer reconstruction, perhaps remove?)

---

## TODO

- [x] PTMQ - Key Contributions
  - [x] Multi-bit Feature Mixer (MFM)
  - [x] Group-wise Distill Loss (GD-Loss)
- [x] Fundamental Tools
  - [x] Rounding-based quantization (AdaRound)
  - [x] BatchNorm folding
- [x] Quantization Modules
  - [x] Layer quantization
  - [x] Block quantization
  - [x] Model quantization
- [ ] Reconstruction
  - [x] Block Reconstruction
- [ ] 🔥PTMQ - Sanity Test
  - [x] CNN - ResNet-18
  - [ ] 🔥Transformer - ViT
- [ ] Preliminary Results
  - [ ] 🔥PTMQ verification
    - [ ] 🔥CNN - ResNet-18
    - [ ] 🔥Transformer - ViT
- [ ] PTMQ - Mixed-Precision Quantization
  - [ ] Pareto Frontier search (ImageNet and ResNet-18)
- [ ] Final Overview
  - [ ] verify on most/all experiments
  - [ ] (partially) reproduce results