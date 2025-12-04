#!/bin/bash

# Launch Gradio demo for Image Quality Assessment
# Usage: ./run_demo.sh [model_path] [options]

# Default model path
MODEL_PATH="${1:-outputs/10302000_full/final_model}"

# Parse additional arguments
SHARE_FLAG=""
DEVICE="cuda"
PORT=7860

shift  # Remove first argument (model_path)

while [[ $# -gt 0 ]]; do
    case $1 in
        --share)
            SHARE_FLAG="--share"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Launching IQA Gradio Demo"
echo "=============================="
echo "Model: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Port: $PORT"
echo ""

# Convert model path to absolute path before changing directory
ABS_MODEL_PATH=$(realpath "$MODEL_PATH")

# Navigate to demo directory
cd "$(dirname "$0")"

# Run demo with absolute model path
uv run python gradio_demo.py \
    --model_path "$ABS_MODEL_PATH" \
    --device "$DEVICE" \
    --server_port "$PORT" \
    $SHARE_FLAG
