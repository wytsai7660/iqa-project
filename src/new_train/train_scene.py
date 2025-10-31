"""
Training script for Scene Classification task.

This script trains only the scene classification component of the multi-task IQA model.
Memory-efficient approach: Only loads scene task data, avoiding OOM issues.

Usage:
    uv run -m src.new_train.train_scene \
        --dataset_paths datasets/bid/ \
        --output_dir outputs/scene_classifier \
        --num_train_epochs 3
"""
# Set environment variables BEFORE any imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import shutil
from pathlib import Path
import torch

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.new_train.model_wrapper import IQAModelWrapper
from src.new_train.dataset_adapter import IQAPairDataset
from src.new_train.processor_no_cut import create_processor_no_cut
from src.new_train.iqa_trainer import IQATrainer
from src.new_train.plot_utils import plot_training_curves


def collate_fn_scene(batch):
    """
    Collate function for scene classification task only.
    Only processes scene-related data to save memory.
    """
    import torch
    import numpy as np
    
    # Helper function to pad sequences
    def pad_and_stack(sequences, pad_value=0):
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padding_len = max_len - len(seq)
            if padding_len > 0:
                padded_seq = torch.cat([seq, torch.full((padding_len,), pad_value, dtype=seq.dtype)])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        return torch.stack(padded)
    
    # Extract scene data from batch
    scene_input_ids_A = [item["input_ids_scene_A"] for item in batch]
    scene_attention_mask_A = [item["attention_mask_scene_A"] for item in batch]
    scene_labels_A = [item["labels_scene_A"] for item in batch if "labels_scene_A" in item]
    scene_input_ids_B = [item["input_ids_scene_B"] for item in batch]
    scene_attention_mask_B = [item["attention_mask_scene_B"] for item in batch]
    scene_labels_B = [item["labels_scene_B"] for item in batch if "labels_scene_B" in item]
    
    # Pad and stack scene sequences
    input_ids_scene_A = pad_and_stack(scene_input_ids_A, pad_value=0)
    attention_mask_scene_A = pad_and_stack(scene_attention_mask_A, pad_value=0)
    input_ids_scene_B = pad_and_stack(scene_input_ids_B, pad_value=0)
    attention_mask_scene_B = pad_and_stack(scene_attention_mask_B, pad_value=0)
    
    # Create proper labels from raw labels (0=ignore, 1=train)
    if scene_labels_A:
        raw_labels_A = pad_and_stack(scene_labels_A, pad_value=0)
        # Convert: label=0 → -100 (ignore), label=1 → token_id (train)
        labels_scene_A = torch.where(raw_labels_A == 0, torch.tensor(-100, dtype=input_ids_scene_A.dtype), input_ids_scene_A)
        labels_scene_A[attention_mask_scene_A == 0] = -100  # Also mask padding
    else:
        # Fallback: use input_ids (old behavior)
        labels_scene_A = input_ids_scene_A.clone()
        labels_scene_A[attention_mask_scene_A == 0] = -100
    
    if scene_labels_B:
        raw_labels_B = pad_and_stack(scene_labels_B, pad_value=0)
        labels_scene_B = torch.where(raw_labels_B == 0, torch.tensor(-100, dtype=input_ids_scene_B.dtype), input_ids_scene_B)
        labels_scene_B[attention_mask_scene_B == 0] = -100
    else:
        labels_scene_B = input_ids_scene_B.clone()
        labels_scene_B[attention_mask_scene_B == 0] = -100
    
    # Stack pixel values
    pixel_values_A = torch.stack([item["pixel_values_A"] for item in batch])
    pixel_values_B = torch.stack([item["pixel_values_B"] for item in batch])
    
    # Collect media offsets
    media_offset_A = [item["media_offset_A"] for item in batch]
    media_offset_B = [item["media_offset_B"] for item in batch]
    
    return {
        # Scene task data
        "input_ids_scene_A": input_ids_scene_A,
        "attention_mask_scene_A": attention_mask_scene_A,
        "labels_scene_A": labels_scene_A,
        "input_ids_scene_B": input_ids_scene_B,
        "attention_mask_scene_B": attention_mask_scene_B,
        "labels_scene_B": labels_scene_B,
        
        # Shared data
        "pixel_values_A": pixel_values_A,
        "pixel_values_B": pixel_values_B,
        "media_offset_A": media_offset_A,
        "media_offset_B": media_offset_B,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Scene Classification task")
    
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
    
    # Loss arguments (scene task only uses CE loss)
    parser.add_argument("--weight_ce", type=float, default=1.0,
                        help="Weight for cross entropy loss")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/scene_classifier",
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing (default: False for scene task)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Convert dataset paths to Path objects
    dataset_paths = [Path(p) for p in args.dataset_paths]
    
    # Initialize tokenizer and processor
    print("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = create_processor_no_cut(tokenizer, image_size=args.image_size)
    
    # Create datasets with scene labels enabled
    print("Creating pair dataset for scene classification training...")
    train_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="training",
        use_scene_labels=True,  # Enable scene labels
        use_distortion_labels=False,  # Disable distortion labels
    )
    
    val_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="validation",
        use_scene_labels=True,
        use_distortion_labels=False,
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model with LoRA
    print("Initializing model with LoRA...")
    model = IQAModelWrapper(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        weight_ce=args.weight_ce,
        weight_kl=0.0,  # No KL loss for scene task
        weight_fidelity=0.0,  # No fidelity loss for scene task
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
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=args.seed,
    )
    
    # Create trainer with scene-specific compute_loss
    class SceneTrainer(IQATrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Compute loss for scene classification task only.
            """
            # Forward pass with scene inputs
            outputs = model(**inputs)
            
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
            
            if return_outputs:
                return loss, outputs
            return loss
        
        def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            """
            Override prediction_step for scene task - only compute loss, no metrics.
            """
            has_labels = "labels_scene_A" in inputs and "labels_scene_B" in inputs
            
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                if has_labels:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                else:
                    loss = None
            
            # Return (loss, None, None) - no logits or labels needed for scene task
            return (loss, None, None)
    
    trainer = SceneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_scene,
    )
    
    # Train
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save final model (both full model and LoRA adapter)
    print("\nSaving final model...")
    final_model_path = f"{args.output_dir}/final"
    
    # Save the full model using trainer (includes training_args.bin)
    trainer.save_model(final_model_path)
    
    # Additionally save LoRA adapter separately for easy loading
    lora_adapter_path = f"{args.output_dir}/lora_adapter"
    model.model.save_pretrained(lora_adapter_path)
    print(f"✅ Full model saved to {final_model_path}")
    print(f"✅ LoRA adapter saved to {lora_adapter_path}")
    
    # Copy base model config and tokenizer for standalone loading
    base_model_path = Path(args.model_name_or_path)
    
    # Copy JSON/text config files
    for file in ["config.json", "generation_config.json", "tokenizer_config.json", 
                 "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                 "preprocessor_config.json", "processor_config.json"]:
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
    
    # Generate training curves
    print("\nGenerating training curves...")
    plot_training_curves(
        output_dir=args.output_dir,
    )
    
    print("\n" + "=" * 60)
    print("Scene classification training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
