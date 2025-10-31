"""
Sequential Training Pipeline: Scene -> Distortion -> Quality

This script trains all three tasks in sequence without saving/loading checkpoints.
The model is kept in memory and passed between training stages.

Usage:
    uv run python train_sequential_pipeline.py \
        --dataset_paths datasets/bid/ \
        --output_dir outputs/sequential_pipeline \
        --num_train_epochs 3 \
        --max_steps 300
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
from pathlib import Path
import torch

from transformers import TrainingArguments, set_seed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.new_train.model_wrapper import IQAModelWrapper
from src.new_train.dataset_adapter import IQAPairDataset
from src.new_train.processor_no_cut import create_processor_no_cut
from src.new_train.iqa_trainer import IQATrainer
from src.new_train.plot_utils import plot_training_curves


def train_scene(model, processor, tokenizer, args):
    """
    Stage 1: Train Scene Classification
    """
    print("\n" + "=" * 80)
    print("STAGE 1/3: Scene Classification Training")
    print("=" * 80)
    
    from src.new_train.train_scene import collate_fn_scene
    
    # Create dataset
    print("\nCreating scene classification dataset...")
    dataset_paths = [Path(p) for p in args.dataset_paths]
    train_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="training",
        use_scene_labels=True,
        use_distortion_labels=False,
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
    
    # Training arguments
    output_dir = f"{args.output_dir}/01_scene"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1,  # Use 1 epoch when max_steps is set
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=args.report_to,
        load_best_model_at_end=False,
    )
    
    # Custom trainer for scene task
    class SceneTrainer(IQATrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model.forward_scene_task(
                pixel_values_A=inputs["pixel_values_A"],
                input_ids_scene_A=inputs["input_ids_scene_A"],
                attention_mask_scene_A=inputs["attention_mask_scene_A"],
                labels_scene_A=inputs["labels_scene_A"],
                media_offset_A=inputs["media_offset_A"],
                pixel_values_B=inputs["pixel_values_B"],
                input_ids_scene_B=inputs["input_ids_scene_B"],
                attention_mask_scene_B=inputs["attention_mask_scene_B"],
                labels_scene_B=inputs["labels_scene_B"],
                media_offset_B=inputs["media_offset_B"],
            )
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            has_labels = "labels_scene_A" in inputs and "labels_scene_B" in inputs
            with torch.no_grad():
                if has_labels:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                else:
                    loss = None
            return (loss, None, None)
    
    # Create trainer
    trainer = SceneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_scene,
    )
    
    # Train
    print("\nStarting scene training...")
    trainer.train()
    
    # Generate plots
    print("\nGenerating training curves...")
    plot_training_curves(output_dir=output_dir)
    
    print("\n✅ Scene training completed!")
    return model


def train_distortion(model, processor, tokenizer, args):
    """
    Stage 2: Train Distortion Classification (on top of Scene knowledge)
    """
    print("\n" + "=" * 80)
    print("STAGE 2/3: Distortion Classification Training")
    print("=" * 80)
    print("Building on Scene classification knowledge...")
    
    from src.new_train.train_distortion import collate_fn_distortion
    
    # Create dataset
    print("\nCreating distortion classification dataset...")
    dataset_paths = [Path(p) for p in args.dataset_paths]
    train_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="training",
        use_scene_labels=False,
        use_distortion_labels=True,
    )
    val_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="validation",
        use_scene_labels=False,
        use_distortion_labels=True,
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Training arguments
    output_dir = f"{args.output_dir}/02_distortion"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=args.report_to,
        load_best_model_at_end=False,
    )
    
    # Custom trainer for distortion task
    class DistortionTrainer(IQATrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model.forward_distortion_task(
                pixel_values_A=inputs["pixel_values_A"],
                input_ids_distortion_A=inputs["input_ids_distortion_A"],
                attention_mask_distortion_A=inputs["attention_mask_distortion_A"],
                labels_distortion_A=inputs["labels_distortion_A"],
                media_offset_A=inputs["media_offset_A"],
                pixel_values_B=inputs["pixel_values_B"],
                input_ids_distortion_B=inputs["input_ids_distortion_B"],
                attention_mask_distortion_B=inputs["attention_mask_distortion_B"],
                labels_distortion_B=inputs["labels_distortion_B"],
                media_offset_B=inputs["media_offset_B"],
            )
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            has_labels = "labels_distortion_A" in inputs and "labels_distortion_B" in inputs
            with torch.no_grad():
                if has_labels:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                else:
                    loss = None
            return (loss, None, None)
    
    # Create trainer
    trainer = DistortionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_distortion,
    )
    
    # Train
    print("\nStarting distortion training...")
    trainer.train()
    
    # Generate plots
    print("\nGenerating training curves...")
    plot_training_curves(output_dir=output_dir)
    
    print("\n✅ Distortion training completed!")
    return model


def train_quality(model, processor, tokenizer, args):
    """
    Stage 3: Train Quality Assessment (on top of Scene + Distortion knowledge)
    """
    print("\n" + "=" * 80)
    print("STAGE 3/3: Quality Assessment Training")
    print("=" * 80)
    print("Building on Scene + Distortion classification knowledge...")
    
    from src.new_train.dataset_adapter import collate_fn_pair
    
    # Create dataset
    print("\nCreating quality assessment dataset...")
    dataset_paths = [Path(p) for p in args.dataset_paths]
    train_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="training",
        use_scene_labels=False,
        use_distortion_labels=False,
    )
    val_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="validation",
        use_scene_labels=False,
        use_distortion_labels=False,
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Training arguments
    output_dir = f"{args.output_dir}/03_quality"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=args.report_to,
        load_best_model_at_end=False,
        metric_for_best_model="eval_plcc",
        greater_is_better=True,
    )
    
    # Create trainer
    trainer = IQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_pair,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting quality training...")
    trainer.train()
    
    # Generate plots
    print("\nGenerating training curves...")
    try:
        from src.new_train.plot_utils import plot_metrics_summary, plot_correlation_metrics
        plot_training_curves(output_dir)
        plot_metrics_summary(output_dir)
        plot_correlation_metrics(output_dir)
    except Exception as e:
        print(f"⚠️  Could not generate all plots: {e}")
    
    print("\n✅ Quality training completed!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Sequential Multi-Task IQA Training Pipeline")
    
    # Dataset
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True,
                        help="Paths to datasets")
    parser.add_argument("--output_dir", type=str, default="outputs/sequential_pipeline",
                        help="Output directory for all stages")
    
    # Model
    parser.add_argument("--model_name_or_path", type=str, default="src/owl3",
                        help="Base model path")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 for full epochs)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="none")
    
    # System
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Loss weights
    parser.add_argument("--use_fidelity_loss", action="store_true",
                        help="Use fidelity loss in quality training")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SEQUENTIAL MULTI-TASK IQA TRAINING PIPELINE")
    print("=" * 80)
    print(f"\n📁 Dataset: {args.dataset_paths}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔧 Base Model: {args.model_name_or_path}")
    print(f"🎯 Training: {args.num_train_epochs} epochs, max {args.max_steps} steps")
    print(f"⚙️  Batch Size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} (effective: {args.per_device_train_batch_size * args.gradient_accumulation_steps})")
    print(f"📊 Learning Rate: {args.learning_rate}")
    print()
    
    # Initialize model ONCE
    print("Initializing base model with LoRA...")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    processor = create_processor_no_cut(tokenizer)
    
    model = IQAModelWrapper(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        weight_fidelity=1.0 if args.use_fidelity_loss else 0.0,
    )
    
    print("✅ Model initialized!\n")
    
    # Stage 1: Scene Classification
    model = train_scene(model, processor, tokenizer, args)
    
    # Stage 2: Distortion Classification
    model = train_distortion(model, processor, tokenizer, args)
    
    # Stage 3: Quality Assessment
    model = train_quality(model, processor, tokenizer, args)
    
    # Save final model
    print("\n" + "=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)
    final_path = f"{args.output_dir}/final_model"
    model.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Final model saved to: {final_path}")
    
    print("\n" + "=" * 80)
    print("🎉 SEQUENTIAL TRAINING PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"  Stage 1 (Scene):      {args.output_dir}/01_scene/")
    print(f"  Stage 2 (Distortion): {args.output_dir}/02_distortion/")
    print(f"  Stage 3 (Quality):    {args.output_dir}/03_quality/")
    print(f"  Final Model:          {final_path}/")
    print()


if __name__ == "__main__":
    main()
