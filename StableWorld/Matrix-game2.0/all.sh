#!/usr/bin/env bash
set -e


OUTPUT_DIR="/opt/liblibai-models/user-workspace/yangying/Matrix-Game/Matrix-Game-2/new_demo_define_48_mian乱序/new1"
CONFIG_PATH="configs/inference_yaml/inference_universal.yaml"
CHECKPOINT_PATH="/opt/liblibai-models/user-workspace/yangying/Matrix-Game/Matrix-Game-2/checkpoint/base_distilled_model/base_distill.safetensors"
PRETRAINED_PATH="/opt/liblibai-models/user-workspace/yangying/Matrix-Game/Matrix-Game-2/checkpoint"
NUM_FRAMES=120
SEED=48
GPU_ID=6
Threshold=0.78

CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_folder "$OUTPUT_DIR" \
    --num_output_frames $NUM_FRAMES \
    --seed $SEED \
    --pretrained_model_path "$PRETRAINED_PATH" \
    --Threshold $Threshold

