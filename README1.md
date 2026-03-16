
<p align="center">
<h1 align="center">StableWorld: Towards Stable and Consistent Long Interactive Video Generation</h1>
<h3 align="center">A Stable Interactive Video Generation Framework with Matrix-Game-2.0 as One of Its Base Models</h3>
</p>


## 📝 Overview
**StableWorld** is a framework for stable and consistent long interactive video generation. **Matrix-Game-2.0** is one of the base models used in this project rather than the project name itself. On top of the original auto-regressive diffusion-based image-to-world framework, StableWorld adds a stability-oriented eviction strategy for long-horizon interactive video synthesis.

## 🤗 StableWorld Model
We provide three pretrained model weights including universal scenes, GTA driving scene and TempleRun game scene. Please refer to our HuggingFace page to reach these resources.

## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

## Installation
Create a conda environment and install dependencies:
```
conda create -n matrix-game-2.0 python=3.10 -y
conda activate matrix-game-2.0
# install apex and FlashAttention
# Our project also depends on [FlashAttention](https://github.com/Dao-AILab/flash-attention)
git clone https://github.com/SkyworkAI/Matrix-Game.git
cd Matrix-Game-2
pip install -r requirements.txt
python setup.py develop
```


## Quick Start
### Download checkpoints
```
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0
```

### Inference
After downloading pretrained models, you can use the following command to generate an interactive video with random action trajectories:
```
python inference.py \
    --config_path configs/inference_yaml/{your-config}.yaml \
    --checkpoint_path {path-to-the-checkpoint} \
    --img_path {path-to-the-input-image} \
    --output_folder outputs \
    --num_output_frames 150 \
    --seed 42 \
    --pretrained_model_path {path-to-the-vae-folder}
```
Or, you can use the script `inference_streaming.py` for generating the interactive videos with your own input actions and images:
```
python inference_streaming.py \
    --config_path configs/inference_yaml/{your-config}.yaml \
    --checkpoint_path {path-to-the-checkpoint} \
    --output_folder outputs \
    --seed 42 \
    --pretrained_model_path {path-to-the-vae-folder}
```

### StableWorld parameters
The `inference.py` script includes two additional parameters for StableWorld mode:

- `--evict_mode`: whether to enable the StableWorld eviction strategy during long video generation. When enabled, the pipeline performs similarity-based window updates to improve temporal stability.
- `--Threshold`: similarity threshold used by the StableWorld eviction logic. Default is `0.78`. Larger values make eviction more conservative, while smaller values make it easier to trigger the update strategy.

If you want to keep the original inference behavior, simply run `inference.py` without `--evict_mode`.

Example with StableWorld enabled:
```
python inference.py \
    --config_path configs/inference_yaml/inference_universal.yaml \
    --checkpoint_path {path-to-the-checkpoint} \
    --img_path {path-to-the-input-image} \
    --output_folder outputs \
    --num_output_frames 150 \
    --seed 42 \
    --pretrained_model_path {path-to-the-vae-folder} \
    --evict_mode True \
    --Threshold 0.78
```

### Tips
- In the current version, upward movement for camera may cause brief rendering glitches (e.g., black screens). A fix is planned for future updates. Adjust movement slightly or change direction to resolve it.


## ⭐ Acknowledgements

We would like to express our gratitude to:

- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) for their strong base model
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module
- [MineRL](https://github.com/minerllabs/minerl) for their excellent gym framework
- [Video-Pre-Training](https://github.com/openai/Video-Pre-Training) for their accurate Inverse Dynamics Model

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{stableworld2026,
  title={StableWorld: Towards Stable and Consistent Long Interactive Video Generation},
  author={Ying Yang and Zhengyao Lv and Tianlin Pan and Haofan Wang and Binxin Yang and Hubery Yin and Chen Li and Ziwei Liu and Chenyang Si},
  journal={arXiv preprint arXiv:2601.15281},
  year={2026}
}
```
