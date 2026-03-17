<h1 align="center">StableWorld: Towards Stable and Consistent Long Interactive Video Generation</h1>
<h3 align="center">An interactive video generation framework built on Oasis-500M</h3>

## 📝 Overview

**StableWorld** is a framework for stable and consistent long interactive video generation. In this repository, **Oasis-500M** is used as one of the base world models rather than the project name itself.

Built on top of the original action-conditioned autoregressive diffusion framework, StableWorld introduces a **stability-oriented eviction strategy** for long-horizon interactive video synthesis. Instead of always applying naive FIFO sliding-window truncation, StableWorld performs **similarity-aware history updates** to preserve cleaner visual memories and improve temporal consistency over long rollouts.

This repository supports both:

- **Original Oasis inference behavior**
- **StableWorld-enhanced inference behavior**

---

## ✨ Features

- **Oasis-500M** as a base action-conditioned world model
- **StableWorld eviction strategy** for long-horizon generation
- Support for both:
  - **vanilla Oasis sliding-window inference**
  - **StableWorld similarity-aware eviction inference**
- Image-prompt and video-prompt based generation
- Action-conditional autoregressive sampling
- Easy switching between original and StableWorld modes through command-line arguments

---

## 🤗 Model

This repository uses the public **Oasis-500M** checkpoints as base model weights:

- `oasis500m.safetensors`: DiT checkpoint
- `vit-l-20.safetensors`: ViT-VAE checkpoint

Please download them from Hugging Face before running inference.

---

## 📦 Requirements

We tested this repository on the following setup:

- Nvidia GPU with at least **24 GB** memory
- Linux operating system
- Python **3.10**
- CUDA-compatible PyTorch environment

Main dependencies:

- `torch`
- `torchvision`
- `einops`
- `diffusers`
- `timm`
- `av`
- `opencv-python`
- `imageio`
- `safetensors`
- `tqdm`

---

## ⚙️ Installation

Install the dependencies inside your current repository:

```bash
# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install einops diffusers timm av opencv-python imageio safetensors tqdm
````

---

## ⬇️ Download the Model Weights

Inside the current repository directory, run:

```bash
huggingface-cli login
huggingface-cli download Etched/oasis-500m oasis500m.safetensors
huggingface-cli download Etched/oasis-500m vit-l-20.safetensors
```

Or download them into a local directory:

```bash
huggingface-cli download Etched/oasis-500m oasis500m.safetensors --local-dir ./checkpoints
huggingface-cli download Etched/oasis-500m vit-l-20.safetensors --local-dir ./checkpoints
```

---

## 📌 Example Commands

### Vanilla Oasis Inference

```bash
python generate.py \
  --oasis-ckpt ./oasis500m.safetensors \
  --vae-ckpt ./vit-l-20.safetensors \
  --prompt-path sample_data/sample_image_0.png \
  --actions-path sample_data/sample_test.actions.pt \
  --output-path outputs/oasis_vanilla.mp4 \
  --num-frames 1200 \
  --ddim-steps 10 \
  --evict-mode False
```

---

### StableWorld Inference

```bash
python generate.py \
  --oasis-ckpt ./oasis500m.safetensors \
  --vae-ckpt ./vit-l-20.safetensors \
  --prompt-path sample_data/sample_image_0.png \
  --actions-path sample_data/sample_test.actions.pt \
  --output-path outputs/stableworld.mp4 \
  --num-frames 1200 \
  --ddim-steps 10 \
  --evict-mode True \
  --sim-threshold 0.78
```

---

## 📁 Repository Structure

A typical project structure looks like:

```text
.
├── generate.py
├── dit.py
├── vae.py
├── utils.py
├── media/
│   ├── arch.png
│   └── thumb.png
├── sample_data/
│   ├── sample_image_0.png
│   └── sample_actions_0.one_hot_actions.pt
├── outputs/
└── README.md
```

---


## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you find this codebase useful for your research, please kindly cite our paper:

```bibtex
@article{stableworld2026,
  title={StableWorld: Towards Stable and Consistent Long Interactive Video Generation},
  author={Ying Yang and Zhengyao Lv and Tianlin Pan and Haofan Wang and Binxin Yang and Hubery Yin and Chen Li and Ziwei Liu and Chenyang Si},
  journal={arXiv preprint arXiv:2601.15281},
  year={2026}
}
```


