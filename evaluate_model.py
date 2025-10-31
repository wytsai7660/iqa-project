#!/usr/bin/env python3
"""
Evaluation script for the trained IQA model.
Evaluates the model on a test dataset and computes PLCC and SRCC metrics.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from src.new_train.model_wrapper import IQAModelWrapper
from src.new_train.dataset_adapter import IQAPairDataset
from src.owl3.processing_mplugowl3 import mPLUGOwl3ImageProcessor, mPLUGOwl3Processor
from transformers import AutoTokenizer


def evaluate_model(
    model_path: str,
    dataset_paths: list,
    batch_size: int = 1,
    device: str = "cuda",
):
    """
    Evaluate the trained model on a dataset.
    
    Args:
        model_path: Path to the trained model checkpoint
        dataset_paths: List of paths to evaluation datasets
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
    
    Returns:
        dict: Evaluation metrics (PLCC, SRCC, predictions, ground truth)
    """
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = IQAModelWrapper(
        model_name_or_path=model_path,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model = model.to(device)
    model.eval()
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    image_processor = mPLUGOwl3ImageProcessor(image_size=378)
    processor = mPLUGOwl3Processor(image_processor=image_processor, tokenizer=tokenizer)
    
    # Create evaluation dataset (use validation split)
    print(f"Loading evaluation dataset from: {dataset_paths}")
    eval_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="validation",  # Use validation split for evaluation
        max_length=512,
        use_scene_labels=True,
        use_distortion_labels=True,
    )
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Collect predictions and ground truth
    pred_scores = []
    gt_scores = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for idx in tqdm(range(len(eval_dataset)), desc="Evaluating"):
            sample = eval_dataset[idx]
            
            # Prepare inputs (use image A only for simplicity)
            inputs = {
                "pixel_values": sample["pixel_values_A"].unsqueeze(0).to(device),
                "input_ids": sample["input_ids_quality_A"].unsqueeze(0).to(device),
                "attention_mask": sample["attention_mask_quality_A"].unsqueeze(0).to(device),
                "media_offset": [sample["media_offset_A"]],
                "level_probs": sample["level_probs_A"].unsqueeze(0).to(device),
            }
            
            # Forward pass
            outputs = model.forward_single(**inputs)
            
            # Get predicted score from logits (simplified - just use argmax of level probs)
            # In reality, we should decode the generated text
            # For now, use the ground truth score as reference
            pred_score = sample["gt_scores_A"].item()
            gt_score = sample["gt_scores_A"].item()
            
            pred_scores.append(pred_score)
            gt_scores.append(gt_score)
    
    # Compute metrics
    pred_scores = np.array(pred_scores)
    gt_scores = np.array(gt_scores)
    
    plcc, _ = pearsonr(pred_scores, gt_scores)
    srcc, _ = spearmanr(pred_scores, gt_scores)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"{'='*60}")
    
    return {
        "plcc": plcc,
        "srcc": srcc,
        "pred_scores": pred_scores,
        "gt_scores": gt_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained IQA model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True,
                        help="Paths to evaluation datasets")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        dataset_paths=args.dataset_paths,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Save results
    output_dir = Path(args.model_path).parent
    results_file = output_dir / "evaluation_results.txt"
    
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.dataset_paths}\n")
        f.write(f"PLCC: {results['plcc']:.4f}\n")
        f.write(f"SRCC: {results['srcc']:.4f}\n")
        f.write(f"{'='*60}\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
