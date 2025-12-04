# Image Quality Assessment Demo

This directory contains a Gradio-based web demo for the Image Quality Assessment model.

## Features

The demo provides an interactive interface where users can:
- 📤 Upload an image
- 🏞️ View predicted scene type
- 🔍 View predicted distortion type
- ⭐ See the overall quality score (1-5 scale)
- 📊 Visualize the quality token distribution (bad, low, fair, good, awesome)

## How It Works

The demo uses the same sequential Q&A pipeline as `eval_sequential_model.py`:

1. **Scene Prediction**: Model first analyzes the scene type
2. **Distortion Prediction**: Using scene context, model identifies distortion type
3. **Quality Prediction**: Using both scene and distortion context, model predicts quality

Each step builds on the previous one, following a conversational approach to image quality assessment.

## Usage

### Quick Start

```bash
# From the demo directory
./run_demo.sh

# Or specify a different model
./run_demo.sh outputs/10302000_full/final_model

# Create a public share link
./run_demo.sh outputs/10302000_full/final_model --share

# Use CPU instead of GPU
./run_demo.sh outputs/10302000_full/final_model --device cpu

# Use a different port
./run_demo.sh outputs/10302000_full/final_model --port 8080
```

### Python Command

```bash
# From project root
uv run python demo/gradio_demo.py \
    --model_path outputs/10302000_full/final_model \
    --device cuda \
    --server_port 7860

# With public sharing
uv run python demo/gradio_demo.py \
    --model_path outputs/10302000_full/final_model \
    --share
```

## Arguments

- `--model_path`: Path to the trained model directory (default: `outputs/10302000_full/final_model`)
- `--device`: Device to use for inference (`cuda` or `cpu`, default: `cuda`)
- `--share`: Create a public Gradio share link
- `--server_name`: Server name for Gradio (default: `0.0.0.0`)
- `--server_port`: Server port for Gradio (default: `7860`)

## Requirements

The demo requires:
- gradio >= 5.15.1
- All other dependencies from the main project (see `pyproject.toml`)

Install with:
```bash
uv sync
```

## Model Support

The demo supports:
- LoRA adapter models (automatically loads base model + adapter)
- Full fine-tuned models

## Output Format

For each uploaded image, the demo displays:

1. **Scene Type**: e.g., "animal", "landscape", "indoor", etc.
2. **Distortion Type**: e.g., "noise", "blur", "compression", etc.
3. **Quality Score**: Numerical score from 1.0 (bad) to 5.0 (awesome)
4. **Quality Distribution**: Bar chart showing probability distribution over all quality levels:
   - bad (1.0)
   - low (2.0)
   - fair (3.0)
   - good (4.0)
   - awesome (5.0)

## Architecture

The demo is structured as:

```
demo/
├── gradio_demo.py      # Main demo application
├── run_demo.sh         # Convenience launcher script
└── README.md           # This file
```

Key components:
- `IQADemo` class: Handles model loading and inference pipeline
- `create_gradio_interface()`: Builds the Gradio UI
- Sequential prediction methods: `predict_scene()`, `predict_distortion()`, `predict_quality()`

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
- Use CPU: `--device cpu`
- Close other GPU processes
- Use a smaller model if available

### Model Not Found

Ensure the model path points to a valid model directory containing either:
- `adapter_config.json` (for LoRA models)
- `config.json` (for full models)

### Slow Inference

First inference may be slow due to model loading and compilation. Subsequent inferences should be faster.

## Example Usage

1. Launch the demo:
   ```bash
   ./run_demo.sh
   ```

2. Open your browser to `http://localhost:7860`

3. Upload an image or use the example images

4. Click "🔍 Analyze Image"

5. View the results:
   - Scene type prediction
   - Distortion type prediction
   - Quality score
   - Probability distribution chart

## Development

To modify the demo:
1. Edit `gradio_demo.py`
2. Restart the demo
3. Refresh your browser

The demo supports hot-reloading during development.
