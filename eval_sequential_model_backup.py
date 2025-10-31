"""
Evaluate the sequential trained model on test set.

Usage:
    uv run python eval_sequential_model.py \
        --model_path outputs/test_sequential_koniq/final_model \
        --dataset_paths datasets/koniq-10k/ \
        --split testing
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.new_train.model_wrapper import IQAModelWrapper
from src.new_train.dataset_adapter import IQAPairDataset
from src.new_train.processor_no_cut import create_processor_no_cut


def evaluate_model(model, dataset, device="cuda", tokenizer=None, processor=None):
    """
    Evaluate model on dataset with sequential Q&A (scene -> distortion -> quality).
    
    Model does NOT see ground truth answers - uses its own predictions as context.
    
    Returns:
        dict: Dictionary containing predictions and ground truth
    """
    model.eval()
    model.to(device)
    
    all_predictions_A = []
    all_predictions_B = []
    all_gt_scores_A = []
    all_gt_scores_B = []
    all_gt_stds_A = []
    all_gt_stds_B = []
    
    print(f"\nEvaluating on {len(dataset)} samples...")
    print("Using sequential Q&A WITHOUT ground truth answers (model's own predictions as context)")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            try:
                item = dataset[idx]
                
                # Process both images with sequential Q&A
                pred_score_A = evaluate_one_image_sequential(
                    model, item, "A", device, tokenizer, processor
                )
                pred_score_B = evaluate_one_image_sequential(
                    model, item, "B", device, tokenizer, processor
                )
                
                if pred_score_A is None or pred_score_B is None:
                    print(f"\nWarning: Could not extract scores for sample {idx}")
                    continue
                
                # Store results
                all_predictions_A.append(pred_score_A)
                all_predictions_B.append(pred_score_B)
                all_gt_scores_A.append(item["gt_scores_A"])
                all_gt_scores_B.append(item["gt_scores_B"])
                all_gt_stds_A.append(item["gt_stds_A"])
                all_gt_stds_B.append(item["gt_stds_B"])
                
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return {
        "predictions_A": np.array(all_predictions_A),
        "predictions_B": np.array(all_predictions_B),
        "gt_scores_A": np.array(all_gt_scores_A),
        "gt_scores_B": np.array(all_gt_scores_B),
        "gt_stds_A": np.array(all_gt_stds_A),
        "gt_stds_B": np.array(all_gt_stds_B),
    }


def evaluate_one_image_sequential(
    model,
    processor,
    tokenizer,
    pixel_values: torch.Tensor,
    media_offset: torch.Tensor,
    gt_scene: str,
    gt_distortion: str,
    gt_quality: float,
    level_token_ids: List[int],
    level_scores: List[float],
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Sequential Q&A evaluation:
    1. Ask scene → get model's scene answer
    2. Ask distortion (with model's scene answer context) → get model's distortion answer
    3. Ask quality (with model's scene+distortion answers context) → get quality logits
    """
    model.eval()
    
        # === Step 3: Ask quality question (with scene+distortion answer context) ===
    quality_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "<|image|>\n"},
        {"role": "user", "content": "What is the scene type of this image?"},
        {"role": "assistant", "content": scene_response},
        {"role": "user", "content": "What is the distortion type of this image?"},
        {"role": "assistant", "content": distortion_response},
        {"role": "user", "content": "What do you think about the quality of this image?"},
        {"role": "assistant", "content": "The quality of this image is "},
    ]
    
    # Use processor for proper image token handling
    quality_inputs = processor(
        messages=quality_messages,
        images=None,
        return_tensors="pt"
    )
    
    quality_input_ids = quality_inputs['input_ids'].to(device)
    quality_attention_mask = quality_inputs['attention_mask'].to(device)
    
    # Use processor to properly handle image token
    scene_inputs = processor(
        messages=scene_messages,
        images=None,  # pixel_values already preprocessed
        return_tensors="pt"
    )
    
    scene_input_ids = scene_inputs['input_ids'].to(device)
    scene_attention_mask = scene_inputs['attention_mask'].to(device)
    
    # Generate scene answer
    with torch.no_grad():
        scene_out = model.model.generate(
            input_ids=scene_input_ids,
            pixel_values=pixel_values,
            media_offset=media_offset,
            attention_mask=scene_attention_mask,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
        )
    
    # Decode scene response (decode all generated tokens)
    scene_response = tokenizer.decode(
        scene_out[0],
        skip_special_tokens=True
    ).strip()


def compute_metrics(predictions, ground_truth):
    """
    Compute evaluation metrics.
    """
    from scipy.stats import pearsonr, spearmanr
    
    # MAE
    mae = np.mean(np.abs(predictions - ground_truth))
    
    # MSE
    mse = np.mean((predictions - ground_truth) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # PLCC (Pearson Linear Correlation Coefficient)
    plcc, _ = pearsonr(predictions, ground_truth)
    
    # SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = spearmanr(predictions, ground_truth)
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "plcc": plcc,
        "srcc": srcc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sequential IQA Model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--dataset_paths", type=str, nargs="+", required=True,
                        help="Paths to datasets")
    parser.add_argument("--split", type=str, default="testing",
                        help="Dataset split to evaluate on")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SEQUENTIAL IQA MODEL EVALUATION")
    print("=" * 80)
    print(f"\n📁 Model: {args.model_path}")
    print(f"📁 Dataset: {args.dataset_paths}")
    print(f"📊 Split: {args.split}")
    print(f"🖥️  Device: {args.device}")
    
    # Load tokenizer and processor
    print("\n🔧 Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    processor = create_processor_no_cut(tokenizer)
    
    # Load model
    print("🔧 Loading model...")
    model = IQAModelWrapper(
        model_name_or_path=args.model_path,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    
    print(f"✅ Model loaded from {args.model_path}")
    
    # Create dataset (with scene and distortion labels for sequential Q&A)
    print(f"\n📊 Creating {args.split} dataset...")
    dataset_paths = [Path(p) for p in args.dataset_paths]
    test_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split=args.split,
        use_scene_labels=True,  # Need scene labels for sequential Q&A
        use_distortion_labels=True,  # Need distortion labels for sequential Q&A
    )
    
    print(f"✅ Dataset size: {len(test_dataset)}")
    
    # Evaluate (with sequential Q&A - model's own answers as context)
    results = evaluate_model(
        model, 
        test_dataset, 
        device=args.device,
        tokenizer=tokenizer,
        processor=processor,
    )
    
    # Combine A and B predictions
    all_predictions = np.concatenate([results["predictions_A"], results["predictions_B"]])
    all_gt_scores = np.concatenate([results["gt_scores_A"], results["gt_scores_B"]])
    
    # Compute metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    metrics = compute_metrics(all_predictions, all_gt_scores)
    
    print(f"\n📊 Overall Metrics (n={len(all_predictions)}):")
    print(f"  MAE:        {metrics['mae']:.4f}")
    print(f"  RMSE:       {metrics['rmse']:.4f}")
    print(f"  MSE:        {metrics['mse']:.4f}")
    print("-" * 80)
    print(f"  PLCC:       {metrics['plcc']:.4f}  {'█' * int(metrics['plcc'] * 20)}")
    print(f"  SRCC:       {metrics['srcc']:.4f}  {'█' * int(metrics['srcc'] * 20)}")
    print("=" * 80)
    
    # Compute separate metrics for A and B
    metrics_A = compute_metrics(results["predictions_A"], results["gt_scores_A"])
    metrics_B = compute_metrics(results["predictions_B"], results["gt_scores_B"])
    
    print(f"\n📊 Image A Metrics (n={len(results['predictions_A'])}):")
    print(f"  MAE:  {metrics_A['mae']:.4f}  |  RMSE: {metrics_A['rmse']:.4f}")
    print(f"  PLCC: {metrics_A['plcc']:.4f}  |  SRCC: {metrics_A['srcc']:.4f}")
    
    print(f"\n📊 Image B Metrics (n={len(results['predictions_B'])}):")
    print(f"  MAE:  {metrics_B['mae']:.4f}  |  RMSE: {metrics_B['rmse']:.4f}")
    print(f"  PLCC: {metrics_B['plcc']:.4f}  |  SRCC: {metrics_B['srcc']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation completed!")
    print("=" * 80)
    

if __name__ == "__main__":
    main()
