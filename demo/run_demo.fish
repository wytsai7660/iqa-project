#!/usr/bin/env fish

# Launch Gradio demo for Image Quality Assessment
# Usage: ./run_demo.fish [model_path] [options]

# Default model path
set -l model_path $argv[1]
if test -z "$model_path"
    set model_path "outputs/10302000_full/final_model"
end

# Parse additional arguments
set -l share_flag ""
set -l device "cuda"
set -l port 7860

set -e argv[1]  # Remove first argument

while test (count $argv) -gt 0
    switch $argv[1]
        case --share
            set share_flag "--share"
            set -e argv[1]
        case --device
            set device $argv[2]
            set -e argv[1 2]
        case --port
            set port $argv[2]
            set -e argv[1 2]
        case '*'
            echo "Unknown option: $argv[1]"
            exit 1
    end
end

echo "🚀 Launching IQA Gradio Demo"
echo "=============================="
echo "Model: $model_path"
echo "Device: $device"
echo "Port: $port"
echo ""

# Convert model path to absolute path before changing directory
set -l abs_model_path (realpath $model_path)

# Navigate to demo directory
cd (dirname (status --current-filename))

# Run demo with absolute model path
uv run python gradio_demo.py \
    --model_path $abs_model_path \
    --device $device \
    --server_port $port \
    $share_flag
