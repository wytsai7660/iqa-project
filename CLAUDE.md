<!-- cspell:words deqa liqe mllm logits peft koniq dataclasses hyperparameters rmse srcc plcc csiq stddev kadid ipynb pyproject -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Introduction

This is an Image Quality Assessment (IQA) project that is based on the IQA
methods described in the papers deqa-score.pdf (DeQA-Score) and liqe.pdf (LIQE).
Read those two PDFs to understand their techniques. The goal of this project is
to combine the techniques of DeQA-Score and LIQE.

DeQA-Score uses the mPLUG-Owl2 multimodal large language model (MLLM) to predict
image mean opinion scores (MOS). Normally, an MLLM can't directly predict a
continuous MOS because it outputs discrete token probabilities. It solves this
using a special technique. It takes the ground truth MOS and standard deviation
of each image and constructs a soft label, which contains the probabilities that
the MOS of the image is 0.5-1.5, 1.5-2.5, ..., or 4.5-5.5. It then assigns each
of those 5 probabilities to the tokens "bad", "poor", "fair", "good", and
"excellent", respectively. Each token is assigned a score value from 1 to 5,
respectively. To get the probability of each token, the authors give the MLLM an
image and a prompt like "The quality of this image is" and have the MLLM predict
the next token. The probabilities are then calculated as the softmax of the 5
level tokens from the output logits. The final MOS prediction of the MLLM is
calculated as the weighted average of each token's score value and its
probability.

LIQE proposes that using multitask training to train the model to predict an
image's scene type, distortion type, and MOS at the same time helps improve its
performance on score prediction.

Both of these methods employ fidelity loss, where the model is trained not to
output absolute MOSs, but relative MOSs such that if the ground truth MOS of
image A is larger than that of image B, then the predicted MOS of image A should
also be larger than that of image B. This allows the model to use multiple
datasets for training, since different datasets have different MOS scales, so
their MOSs cannot be directly compared.

This project tries to do both: we will first ask the MLLM to answer the scene
type of an image, then following the same conversation, ask it to answer the
distortion type of an image. Finally, still in the same conversation, we will
ask it to predict the MOS of the image, using the same technique as DeQA-Score.

This project fine-tunes mPLUG-Owl3 (an MLLM) using LoRA for three sequential
tasks:

1. **Scene Classification** - Identify scene types (animal, cityscape, human,
   indoor, landscape, night, plant, still_life, others)
2. **Distortion Classification** - Identify distortion types (blur, color,
   contrast, jpeg compression, etc.)
3. **Quality Assessment** - Predict quality scores using soft labels
   (bad/low/fair/good/awesome corresponding to scores 1-5)

The model learns these tasks sequentially, where each task builds on the
previous one's context.

Your goal is to find any flaws and bugs in the training or evaluation process
and correct them to improve the model's PLCC and SRCC performance.

## Setup and Environment

**Package Manager**: This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run Python scripts/modules
uv run python <script.py>
uv run -m <module.path>
```

**Python Version**: 3.12 (see `.python-version`)

**Key Dependencies**:

- PyTorch 2.8.0 with flash-attention
- transformers 4.47.0
- peft (for LoRA)
- Custom mPLUG-Owl3 code in `src/owl3/`

## Training Commands

### Quick Demo (10 steps per task)

```bash
bash run_quick_demo.sh
```

### Full Sequential Training

```bash
bash run_sequential_training.sh
```

### Individual Task Training

```bash
# Stage 1: Scene Classification
uv run -m src.new_train.train_scene \
    --dataset_paths datasets/bid/ \
    --output_dir outputs/01_scene \
    --num_train_epochs 3

# Stage 2: Distortion Classification (loads Stage 1 model)
uv run -m src.new_train.train_distortion \
    --dataset_paths datasets/bid/ \
    --output_dir outputs/02_distortion \
    --model_name_or_path outputs/01_scene/final \
    --num_train_epochs 3

# Stage 3: Quality Assessment (loads Stage 2 model)
uv run -m src.new_train.train_iqa_lora \
    --dataset_paths datasets/bid/ \
    --output_dir outputs/03_quality \
    --model_name_or_path outputs/02_distortion/final \
    --num_train_epochs 3 \
    --use_fidelity_loss
```

### In-Memory Sequential Pipeline

```bash
uv run python train_sequential_pipeline.py \
    --dataset_paths datasets/bid/ \
    --output_dir outputs/sequential_pipeline \
    --num_train_epochs 3
```

### Evaluation

```bash
uv run python eval_sequential_model.py \
    --model_path outputs/03_quality/final \
    --dataset_paths datasets/koniq-10k/ \
    --split testing
```

### Simple Evaluation

```bash
uv run python evaluate_model.py \
    --model_path outputs/03_quality/final \
    --dataset_paths datasets/live/
```

## Code Architecture

### Core Modules

**`src/owl3/`** - mPLUG-Owl3 implementation

- Copied from upstream. Only slightly modified.
- Contains model architecture, processor, and configuration
- See `src/owl3/README.md` for upstream documentation

**`src/dataset.py`** - Main dataset class `PairDataset`

- Loads image pairs with quality labels from `labels.csv` files
- Supports three conversation modes based on flags:
  - `use_scene_labels=False, use_distortion_labels=False`: Image → Quality
    (basic)
  - `use_scene_labels=True, use_distortion_labels=False`: Image → Scene →
    Quality
  - `use_scene_labels=True, use_distortion_labels=True`: Image → Scene →
    Distortion → Quality (full)
- Returns `PairDatasetItem` with two images and their labels for ranking loss

**`src/config.py`** - Project-level configuration

- `SRC_DIR`: Source directory path
- `MODEL_DIR`: Points to `src/owl3/`
- `QUALITY_TOKENS`: Quality level tokens (currently empty, defaults in code)

### Training Framework (`src/new_train/`)

**`config.py`** - Training configuration dataclasses

- `ModelConfig`: LoRA settings, quality tokens, model paths
- `DataConfig`: Dataset paths and preprocessing settings
- `LossConfig`: Loss function weights and settings
- `TrainingConfig`: Training hyperparameters

**`model_wrapper.py`** - `IQAModelWrapper` class

- Wraps mPLUG-Owl3 with LoRA adapters
- Implements three custom loss functions:
  1. Cross-entropy loss for regular tokens
  2. KL divergence loss for quality level tokens (soft labels)
  3. Fidelity loss for pair-wise ranking
- Provides task-specific forward methods:
  - `forward_scene_task()`: Scene classification only
  - `forward_distortion_task()`: Scene + Distortion classification
  - `forward_quality_task()`: Full pipeline with quality prediction

**`dataset_adapter.py`** - `IQAPairDataset`

- Adapter that wraps `PairDataset` for training framework
- Converts dataset items to training batch format

**`iqa_trainer.py`** - Custom HuggingFace `Trainer` subclass

- `SimplifiedProgressCallback`: Clean progress logging
- Memory-efficient evaluation
- Custom metric computation during evaluation

**`metrics.py`** - Evaluation metrics

- `compute_quality_score_from_logits()`: Extracts quality scores from model
  logits using softmax over quality tokens
- `compute_iqa_metrics()`: MAE, RMSE, PLCC (Pearson), SRCC (Spearman)

**`plot_utils.py`** - Training visualization utilities

**`processor_no_cut.py`** - Custom processor

- `create_processor_no_cut()`: Creates processor without image cutting/splitting

### Training Scripts

- `train_scene.py`: Train scene classification (Stage 1)
- `train_distortion.py`: Train distortion classification (Stage 2)
- `train_iqa_lora.py`: Train quality assessment with LoRA (Stage 3)
- `train_sequential_pipeline.py`: All-in-one pipeline script

### Evaluation Scripts

- `eval_sequential_model.py`: True sequential evaluation (model generates
  context)
- `evaluate_model.py`: Simple quality evaluation
- Various `test_*.py` and `debug_*.py`: Unit tests and debugging utilities

### Dataset Preprocessing (`src/dataset_preprocessing_scripts/`)

Scripts to process raw IQA datasets into the format expected by `PairDataset`:

- `process_liqe_labels.py`: Process LIQE dataset labels
- `add_set_column.py`: Add train/val/test splits
- Various dataset-specific scripts (CSIQ, KADID-10k, KonIQ-10k, LIVE)

## Dataset Format

Datasets should be in `datasets/` with structure:

```plaintext
<dataset-name>/
├── labels.csv          # Required: MOS scores and metadata
├── images/             # Required: Distorted images
└── reference-images/   # Optional: For synthetic distortions
```

**`labels.csv` columns**:

- `filename`: Image filename (index column)
- `mos`: Mean opinion score (quality ground truth)
- `stddev`: Standard deviation (optional, for soft labels)
- `distortion`: Distortion type (11 categories)
- `scene1`, `scene2`, `scene3`: Scene types (up to 3)
- `reference`: Reference image filename (for synthetic distortions)
- `set`: Split (training/validation/testing)

Available datasets: bid, csiq, kadid-10k, koniq-10k, live, live-in-the-wild

## Important Notes

### Git Configuration

- **`.gitattributes`**: Strips output and metadata from Jupyter notebooks on
  commit
- Temporary matplotlib PNG files are ignored

### Memory Management

Set before training:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

### Quality Score Computation

Quality is represented as soft labels over 5 tokens: `["bad", "low", "fair",
"good", "awesome"]` corresponding to scores `[1.0, 2.0, 3.0, 4.0, 5.0]`.

Final score = Expected value using softmax over these token logits:

```plaintext
score = Σ p_i * score_i  where p_i = softmax(logits[quality_tokens])
```

### Loss Functions

1. **Cross-Entropy Loss**: Applied to all non-quality tokens (scene, distortion,
   filler text)
2. **KL Divergence Loss**: Applied to quality token position with Gaussian soft
   labels
3. **Fidelity Loss** (optional): Ranking loss for image pairs based on quality
   difference

### LoRA Configuration

Default target modules for LoRA fine-tuning:

```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

Default hyperparameters:

- `lora_r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`

### Model Checkpoints

Training saves:

- Periodic checkpoints at `save_steps` intervals
- `final/` directory with the final model
- TensorBoard logs in `runs/`

To load a checkpoint for continued training, use `--model_name_or_path
<checkpoint_dir>`.

### Jupyter Notebooks

Training notebooks are in the root directory:

- `train_quality_only.ipynb`: Quality-only training
- `train_sequential_configurable.ipynb`: Configurable sequential training
- `train_sequential_pipeline.ipynb`: Sequential pipeline

These are interactive alternatives to the Python scripts.

## Working with This Codebase

1. **Dataset location**: The `datasets/` directory contains the datasets to use.

2. **Model source**: The base mPLUG-Owl3 model is in `src/owl3/`. It is not
   loaded from HuggingFace Hub.

3. **Training workflow**: The three-stage sequential training is the recommended
   approach. Each stage loads the previous stage's model.

4. **Debugging**: Many `test_*.py` and `debug_*.py` scripts exist for verifying
   specific components. Run them with `uv run python <script.py>`.

5. **Type checking**: Configured with Pyright in `pyproject.toml` (currently
   using standard mode).
