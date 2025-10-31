"""
Main training script for IQA with LoRA fine-tuning.

Usage:
    # Single dataset training
    uv run -m src.new_train.train_iqa_lora \
        --dataset_paths data/koniq-10k \
        --output_dir outputs/iqa_lora \
        --num_train_epochs 3

    # Multi-dataset training with fidelity loss
    uv run -m src.new_train.train_iqa_lora \
        --dataset_paths data/koniq-10k data/spaq data/kadid-10k \
        --output_dir outputs/iqa_lora_multi \
        --num_train_epochs 3 \
        --use_fidelity_loss
"""
# Set environment variables BEFORE any imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism warning

import argparse
import sys
import shutil
from pathlib import Path

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.new_train.model_wrapper import IQAModelWrapper
from src.new_train.dataset_adapter import (
    IQAPairDataset,
    collate_fn_pair,
)
from src.new_train.processor_no_cut import create_processor_no_cut
from src.new_train.iqa_trainer import IQATrainer
from src.new_train.plot_utils import plot_training_curves, plot_metrics_summary, plot_correlation_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train IQA model with LoRA")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="src/owl3",
                        help="Path to pretrained model")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True,
                        help="Paths to dataset directories")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--image_size", type=int, default=378,
                        help="Image size")
    parser.add_argument("--use_scene_labels", action="store_true",
                        help="Enable scene type classification task")
    parser.add_argument("--use_distortion_labels", action="store_true",
                        help="Enable distortion type classification task")
    
    # Loss arguments
    parser.add_argument("--weight_ce", type=float, default=1.0,
                        help="Weight for cross entropy loss")
    parser.add_argument("--weight_kl", type=float, default=0.05,
                        help="Weight for KL divergence loss")
    parser.add_argument("--weight_fidelity", type=float, default=1.0,
                        help="Weight for fidelity loss")
    parser.add_argument("--use_fidelity_loss", action="store_true",
                        help="Use fidelity loss")
    parser.add_argument("--use_fix_std", action="store_true",
                        help="Use fixed std=1.0 in fidelity loss")
    parser.add_argument("--binary_fidelity_type", type=str, default="fidelity",
                        choices=["fidelity", "bce"],
                        help="Binary fidelity loss type")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/iqa_lora",
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps (overrides num_train_epochs if > 0)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Training batch size per device (default: 1 to avoid OOM)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Evaluation batch size per device (default: 1 to avoid OOM)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4 for effective batch size of 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="LR scheduler type")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dataloader_num_workers", type=int, default=12,
                        help="Number of dataloader workers (default: 0 to avoid fork issues)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing to save memory (default: True)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    # Use no-cut processor for training (as per Owl3 paper: no cropping for multi-image/video datasets)
    processor = create_processor_no_cut(tokenizer, image_size=args.image_size)
    
    # Create pair dataset (always use pair dataset)
    print("Creating pair dataset for training...")
    train_dataset = IQAPairDataset(
        dataset_paths=args.dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="training",
        max_length=args.max_length,
        use_scene_labels=args.use_scene_labels,
        use_distortion_labels=args.use_distortion_labels,
    )
    val_dataset = IQAPairDataset(
        dataset_paths=args.dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="validation",
        max_length=args.max_length,
        use_scene_labels=args.use_scene_labels,
        use_distortion_labels=args.use_distortion_labels,
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing model with LoRA...")
    model = IQAModelWrapper(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        weight_ce=args.weight_ce,
        weight_kl=args.weight_kl,
        weight_fidelity=args.weight_fidelity if args.use_fidelity_loss else 0.0,
        use_fix_std=args.use_fix_std,
        binary_fidelity_type=args.binary_fidelity_type,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # Evaluation settings
        eval_strategy="epoch",  # Evaluate at end of each epoch
        prediction_loss_only=False,  # Need to process predictions for PLCC/SRCC
        # Other settings
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=args.seed,
        disable_tqdm=False,  # Keep tqdm for training progress
        log_level="warning",  # Reduce logging verbosity
    )
    
    # Note: We don't use compute_metrics here to avoid OOM issues
    # Metrics are computed inside IQATrainer.prediction_step on-the-fly
    
    # Initialize trainer
    trainer = IQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_pair,
    )
    
    # Train
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    print("=" * 60)
    
    # Save final model (both full model and LoRA adapter)
    print("\nSaving final model...")
    final_model_path = f"{args.output_dir}/final"
    
    # Save LoRA adapter
    model.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Copy base model config for standalone loading
    base_model_path = Path(args.model_name_or_path)
    
    # Copy JSON/text config files
    for file in ["config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json"]:
        src = base_model_path / file
        if src.exists():
            dst = Path(final_model_path) / file
            shutil.copy2(src, dst)
            print(f"  Copied {file}")
    
    # Copy Python modeling files (required for trust_remote_code=True)
    for file in ["configuration_mplugowl3.py", "modeling_mplugowl3.py", 
                 "configuration_hyper_qwen2.py", "modeling_hyper_qwen2.py",
                 "image_processing_mplugowl3.py", "processing_mplugowl3.py"]:
        src = base_model_path / file
        if src.exists():
            dst = Path(final_model_path) / file
            shutil.copy2(src, dst)
            print(f"  Copied {file}")
    
    print(f"✅ Model saved to {final_model_path}")
    
    # Generate training curves
    print("\nGenerating training curves...")
    try:
        plot_training_curves(args.output_dir)
        plot_metrics_summary(args.output_dir)
        plot_correlation_metrics(args.output_dir)
        print("✅ All training curves and metrics saved to output directory")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
