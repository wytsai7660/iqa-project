#!/bin/bash
# Sequential Training Script for Three Tasks
# This script trains Scene -> Distortion -> Quality in sequence

set -e  # Exit on error

echo "======================================"
echo "Sequential Multi-Task Training"
echo "======================================"

# Configuration
DATASET="datasets/koniq-10k/"
BASE_MODEL="src/owl3"  # Base model to start from
BASE_OUTPUT="outputs/202510261700"
# MAX_STEPS=100  # Reduce for quick testing
BATCH_SIZE=1
GRAD_ACCUM=8

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🎯 Sequential Training: Each task builds on the previous one"
echo "   Scene → Distortion → Quality"
echo ""

echo ""
echo "Step 1/3: Training Scene Classification"
echo "========================================"
echo "Starting from base model: $BASE_MODEL"
echo ""

uv run -m src.new_train.train_scene \
    --dataset_paths $DATASET \
    --output_dir $BASE_OUTPUT/01_scene \
    --model_name_or_path $BASE_MODEL \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    # --max_steps $MAX_STEPS

echo ""
echo "✅ Scene training completed!"
echo ""

echo ""
echo "Step 2/3: Training Distortion Classification"
echo "==========================================="
# Use the final model from scene training (includes config.json now)
SCENE_MODEL="$BASE_OUTPUT/01_scene/final"
echo "Loading from Scene checkpoint: $SCENE_MODEL"
echo ""

uv run -m src.new_train.train_distortion \
    --dataset_paths $DATASET \
    --output_dir $BASE_OUTPUT/02_distortion \
    --model_name_or_path $SCENE_MODEL \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    # --max_steps $MAX_STEPS

echo ""
echo "✅ Distortion training completed!"
echo ""

echo ""
echo "Step 3/3: Training Quality Assessment"
echo "======================================"
# Use the final model from distortion training
DISTORTION_MODEL="$BASE_OUTPUT/02_distortion/final"
echo "Loading from Distortion checkpoint: $DISTORTION_MODEL"
echo ""

uv run -m src.new_train.train_iqa_lora \
    --dataset_paths $DATASET \
    --output_dir $BASE_OUTPUT/03_quality \
    --model_name_or_path $DISTORTION_MODEL \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    # --max_steps $MAX_STEPS \
    --use_fidelity_loss

echo ""
echo "✅ Quality training completed!"
echo "Final model saved at: $BASE_OUTPUT/03_quality/final"

echo ""
echo "======================================"
echo "Sequential Training Completed!"
echo "======================================"
echo "Scene model: $BASE_OUTPUT/01_scene/final"
echo "Distortion model: $BASE_OUTPUT/02_distortion/final"
echo "Final model: $BASE_OUTPUT/03_quality/final"
echo ""
echo "Next step: Evaluate on datasets/live/"
