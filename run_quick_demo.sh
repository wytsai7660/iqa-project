#!/bin/bash
# Quick Sequential Training Demo (10 steps per task for demonstration)
# This script demonstrates the complete workflow quickly

set -e

echo "======================================"
echo "Quick Sequential Multi-Task Training Demo"
echo "======================================"

# Configuration for quick demo
DATASET="datasets/bid/"
BASE_OUTPUT="outputs/quick_demo"
MAX_STEPS=10  # Very small for quick demonstration
BATCH_SIZE=1
GRAD_ACCUM=2
LR=2e-4

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clean previous demo outputs
rm -rf $BASE_OUTPUT
mkdir -p $BASE_OUTPUT

echo ""
echo "========================================="
echo "Step 1/3: Training Scene Classification"
echo "========================================="
uv run -m src.new_train.train_scene \
    --dataset_paths $DATASET \
    --output_dir $BASE_OUTPUT/01_scene \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --save_steps 10 \
    --eval_steps 10 \
    --logging_steps 5

echo ""
echo "✅ Scene training completed!"

echo ""
echo "============================================"
echo "Step 2/3: Training Distortion Classification"
echo "============================================"
# Use the final scene model as starting point
uv run -m src.new_train.train_distortion \
    --dataset_paths $DATASET \
    --output_dir $BASE_OUTPUT/02_distortion \
    --model_name_or_path $BASE_OUTPUT/01_scene/final \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --save_steps 10 \
    --eval_steps 10 \
    --logging_steps 5

echo ""
echo "✅ Distortion training completed!"

echo ""
echo "========================================"
echo "Step 3/3: Training Quality Assessment"
echo "========================================"
# Use the final distortion model as starting point
uv run -m src.new_train.train_iqa_lora \
    --dataset_paths $DATASET \
    --output_dir $BASE_OUTPUT/03_quality \
    --model_name_or_path $BASE_OUTPUT/02_distortion/final \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --save_steps 10 \
    --eval_steps 10 \
    --logging_steps 5 \
    --use_fidelity_loss

echo ""
echo "✅ Quality training completed!"

echo ""
echo "======================================"
echo "Sequential Training Demo Completed!"
echo "======================================"
echo ""
echo "Models saved at:"
echo "  Scene:      $BASE_OUTPUT/01_scene/final"
echo "  Distortion: $BASE_OUTPUT/02_distortion/final"
echo "  Quality:    $BASE_OUTPUT/03_quality/final"
echo ""
echo "The quality model has learned all three tasks sequentially!"
echo ""
echo "To evaluate on datasets/live/, run:"
echo "  uv run python evaluate_model.py \\"
echo "    --model_path $BASE_OUTPUT/03_quality/final \\"
echo "    --dataset_paths datasets/live/"
