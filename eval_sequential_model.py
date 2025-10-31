"""
Evaluate the sequential trained model on test set.

This script implements TRUE sequential Q&A evaluation where:
1. Model generates scene answer (no ground truth)
2. Model generates distortion answer (using its own scene answer as context)
3. Model predicts quality (using its own scene+distortion answers as context)

Quality score is computed as expected value from softmax of quality token logits.

Usage:
    uv run python eval_sequential_model.py \
        --model_path outputs/sequential_notebook/final_model \
        --dataset_paths datasets/koniq-10k/ \
        --split testing
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.new_train.model_wrapper import IQAModelWrapper
from src.new_train.dataset_adapter import IQAPairDataset
from src.new_train.processor_no_cut import create_processor_no_cut


def evaluate_one_image_sequential(
    model,
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
    Sequential Q&A evaluation for one image:
    1. Ask scene → get model's scene answer
    2. Ask distortion (with model's scene answer context) → get model's distortion answer
    3. Ask quality (with model's scene+distortion answers context) → get quality logits
    
    Args:
        model: IQAModelWrapper
        tokenizer: Tokenizer
        pixel_values: Preprocessed image tensor
        media_offset: Media offset for image
        gt_scene: Ground truth scene (not used during inference, only for logging)
        gt_distortion: Ground truth distortion (not used during inference, only for logging)
        gt_quality: Ground truth quality score
        level_token_ids: List of token IDs for ["bad", "low", "fair", "good", "awesome"]
        level_scores: List of scores [1.0, 2.0, 3.0, 4.0, 5.0]
        device: Device to run on
    
    Returns:
        Dictionary with:
            - predicted_score: Expected value from quality token probabilities
            - scene_response: Model's scene answer
            - distortion_response: Model's distortion answer
            - gt_quality: Ground truth quality
    """
    model.eval()
    
    # === Step 1: Ask scene question ===
    scene_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "<|image|>\n"},
        {"role": "user", "content": "What is the scene type of this image?"},
    ]
    
    # Use tokenizer.apply_chat_template to build text, then manually tokenize
    scene_text = tokenizer.apply_chat_template(
        scene_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    scene_encoding = tokenizer(
        scene_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    scene_input_ids = scene_encoding['input_ids'].to(device)
    scene_attention_mask = scene_encoding['attention_mask'].to(device)
    
    # Generate scene answer
    with torch.no_grad():
        scene_out = model.model.generate(
            input_ids=scene_input_ids,
            pixel_values=pixel_values,
            media_offset=media_offset,
            attention_mask=scene_attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
            output_scores=False,
            return_dict_in_generate=False,
        )
    
    # Decode scene response
    # Note: For this model, generate() returns only the NEW generated tokens
    scene_response = tokenizer.decode(
        scene_out[0],  # Decode all generated tokens
        skip_special_tokens=True
    ).strip()
    
    # === Step 2: Ask distortion question (with scene answer context) ===
    distortion_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "<|image|>\n"},
        {"role": "user", "content": "What is the scene type of this image?"},
        {"role": "assistant", "content": scene_response},
        {"role": "user", "content": "What is the distortion type of this image?"},
    ]
    
    distortion_text = tokenizer.apply_chat_template(
        distortion_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    distortion_encoding = tokenizer(
        distortion_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    distortion_input_ids = distortion_encoding['input_ids'].to(device)
    distortion_attention_mask = distortion_encoding['attention_mask'].to(device)
    
    # Generate distortion answer
    with torch.no_grad():
        distortion_out = model.model.generate(
            input_ids=distortion_input_ids,
            pixel_values=pixel_values,
            media_offset=media_offset,
            attention_mask=distortion_attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
            output_scores=False,
            return_dict_in_generate=False,
        )
    
    # Decode distortion response
    distortion_response = tokenizer.decode(
        distortion_out[0],  # Decode all generated tokens
        skip_special_tokens=True
    ).strip()
    
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
    
    quality_text = tokenizer.apply_chat_template(
        quality_messages,
        tokenize=False,
        add_generation_prompt=False  # Already has assistant prefix
    )
    
    quality_encoding = tokenizer(
        quality_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    quality_input_ids = quality_encoding['input_ids'].to(device)
    quality_attention_mask = quality_encoding['attention_mask'].to(device)
    
    # Forward pass to get logits at quality token position
    with torch.no_grad():
        quality_outputs = model.model(
            input_ids=quality_input_ids,
            pixel_values=pixel_values,
            media_offset=media_offset,
            attention_mask=quality_attention_mask,
        )
    
    # Extract logits at last position (after "The quality of this image is ")
    last_logits = quality_outputs.logits[0, -1, :]  # Shape: (vocab_size,)
    
    # Get logits for the 5 quality tokens
    level_logits = last_logits[level_token_ids]  # Shape: (5,)
    
    # Compute softmax probabilities
    level_probs = F.softmax(level_logits, dim=0)  # Shape: (5,)
    
    # Compute expected score: E[score] = Σ(prob_i × score_i)
    expected_score = sum(prob.item() * score for prob, score in zip(level_probs, level_scores))
    
    return {
        "predicted_score": expected_score,
        "scene_response": scene_response,
        "distortion_response": distortion_response,
        "gt_quality": gt_quality,
        "gt_scene": gt_scene,
        "gt_distortion": gt_distortion,
    }


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
    
    # Get quality token IDs
    level_names = ["bad", "low", "fair", "good", "awesome"]
    level_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    level_token_ids = [
        tokenizer.encode(f" {level_name}", add_special_tokens=False)[0]
        for level_name in level_names
    ]
    
    print(f"\nEvaluating on {len(dataset)} samples...")
    print("Using sequential Q&A WITHOUT ground truth answers (model's own predictions as context)")
    print(f"Quality token IDs: {level_token_ids}")
    
    with torch.no_grad():
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Get ground truth scene and distortion by parsing the messages from pair_dataset
                # We need to extract from the scene_qa and distortion_qa parts
                pair_item = dataset.pair_dataset[idx]
                
                # Helper function to extract scene/distortion from messages
                def extract_answer_from_messages(messages_dict, answer_key):
                    """Extract answer text from a processor output containing messages."""
                    # The messages should be in the original format before processing
                    # We can try to get it from the processor or reconstruct from tokens
                    # For now, we'll use a simpler approach: parse from the conversation in dataset
                    return "unknown"
                
                # Try to get scene and distortion from the raw dataset
                # Since pair_dataset uses random sampling for image_2, we can't easily trace back
                # Instead, let's extract from the quality_message which contains all Q&A
                quality_msg_A = pair_item['image_1']['quality_message']
                quality_msg_B = pair_item['image_2']['quality_message']
                
                # Extract scene and distortion by decoding the input_ids up to the quality question
                # The format is: <image> scene_q scene_a distortion_q distortion_a quality_q quality_a
                # We need to find and extract scene_a and distortion_a
                gt_scene_A = 'unknown'
                gt_distortion_A = 'unknown'
                gt_scene_B = 'unknown'
                gt_distortion_B = 'unknown'
                
                if 'input_ids' in quality_msg_A:
                    full_text_A = tokenizer.decode(quality_msg_A['input_ids'][0], skip_special_tokens=False)
                    # Parse scene answer
                    if "The scene type of this image is " in full_text_A:
                        scene_start = full_text_A.find("The scene type of this image is ") + len("The scene type of this image is ")
                        scene_end = full_text_A.find(".", scene_start)
                        if scene_end > scene_start:
                            gt_scene_A = full_text_A[scene_start:scene_end]
                    # Parse distortion answer
                    if "The distortion type of this image is " in full_text_A:
                        dist_start = full_text_A.find("The distortion type of this image is ") + len("The distortion type of this image is ")
                        dist_end = full_text_A.find(".", dist_start)
                        if dist_end > dist_start:
                            gt_distortion_A = full_text_A[dist_start:dist_end]
                
                if 'input_ids' in quality_msg_B:
                    full_text_B = tokenizer.decode(quality_msg_B['input_ids'][0], skip_special_tokens=False)
                    # Parse scene answer
                    if "The scene type of this image is " in full_text_B:
                        scene_start = full_text_B.find("The scene type of this image is ") + len("The scene type of this image is ")
                        scene_end = full_text_B.find(".", scene_start)
                        if scene_end > scene_start:
                            gt_scene_B = full_text_B[scene_start:scene_end]
                    # Parse distortion answer
                    if "The distortion type of this image is " in full_text_B:
                        dist_start = full_text_B.find("The distortion type of this image is ") + len("The distortion type of this image is ")
                        dist_end = full_text_B.find(".", dist_start)
                        if dist_end > dist_start:
                            gt_distortion_B = full_text_B[dist_start:dist_end]
                
                # Get preprocessed data
                pixel_values_A = item['pixel_values_A'].unsqueeze(0).to(device)
                media_offset_A = item['media_offset_A'].unsqueeze(0).to(device)
                gt_quality_A = item['gt_scores_A']  # Ground truth MOS
                
                pixel_values_B = item['pixel_values_B'].unsqueeze(0).to(device)
                media_offset_B = item['media_offset_B'].unsqueeze(0).to(device)
                gt_quality_B = item['gt_scores_B']
                
                # Evaluate image A
                result_A = evaluate_one_image_sequential(
                    model=model,
                    tokenizer=tokenizer,
                    pixel_values=pixel_values_A,
                    media_offset=media_offset_A,
                    gt_scene=gt_scene_A,
                    gt_distortion=gt_distortion_A,
                    gt_quality=gt_quality_A,
                    level_token_ids=level_token_ids,
                    level_scores=level_scores,
                    device=device
                )
                
                # Evaluate image B
                result_B = evaluate_one_image_sequential(
                    model=model,
                    tokenizer=tokenizer,
                    pixel_values=pixel_values_B,
                    media_offset=media_offset_B,
                    gt_scene=gt_scene_B,
                    gt_distortion=gt_distortion_B,
                    gt_quality=gt_quality_B,
                    level_token_ids=level_token_ids,
                    level_scores=level_scores,
                    device=device
                )
                
                all_predictions_A.append(result_A['predicted_score'])
                all_predictions_B.append(result_B['predicted_score'])
                all_gt_scores_A.append(result_A['gt_quality'])
                all_gt_scores_B.append(result_B['gt_quality'])
                
                # Print first few examples
                if idx < 3:
                    print(f"\n--- Sample {idx} ---")
                    print(f"Image A:")
                    print(f"  GT Scene: {gt_scene_A}, Pred Scene: {result_A['scene_response']}")
                    print(f"  GT Distortion: {gt_distortion_A}, Pred Distortion: {result_A['distortion_response']}")
                    print(f"  GT Quality: {gt_quality_A:.2f}, Pred Quality: {result_A['predicted_score']:.2f}")
                    print(f"Image B:")
                    print(f"  GT Scene: {gt_scene_B}, Pred Scene: {result_B['scene_response']}")
                    print(f"  GT Distortion: {gt_distortion_B}, Pred Distortion: {result_B['distortion_response']}")
                    print(f"  GT Quality: {gt_quality_B:.2f}, Pred Quality: {result_B['predicted_score']:.2f}")
                
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
    }


def compute_metrics(predictions, ground_truth):
    """
    Compute IQA metrics: PLCC, SRCC, MAE, RMSE
    """
    from scipy import stats
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(ground_truth))
    predictions = predictions[valid_mask]
    ground_truth = ground_truth[valid_mask]
    
    if len(predictions) < 2:
        return {
            "plcc": 0.0,
            "srcc": 0.0,
            "mae": float('inf'),
            "rmse": float('inf')
        }
    
    # PLCC (Pearson Linear Correlation Coefficient)
    plcc, _ = stats.pearsonr(predictions, ground_truth)
    
    # SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = stats.spearmanr(predictions, ground_truth)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(predictions - ground_truth))
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    
    return {
        "plcc": plcc,
        "srcc": srcc,
        "mae": mae,
        "rmse": rmse
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate sequential Q&A IQA model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/sequential_notebook/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        default=["datasets/koniq-10k/"],
        help="Paths to dataset directories"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="testing",
        choices=["training", "validation", "testing", "full"],
        help="Dataset split to evaluate (use 'full' for all data)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("SEQUENTIAL Q&A MODEL EVALUATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_paths}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\nLoading model...")
    model_path = args.model_path
    
    # Check if this is a LoRA adapter directory or full model
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora_adapter = adapter_config_path.exists()
    
    if is_lora_adapter:
        print("  Detected LoRA adapter, loading base model + adapter...")
        # Load base model config to get the base model path
        import json
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path", "src/owl3")
        print(f"  Base model: {base_model_path}")
        print(f"  Adapter: {model_path}")
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        processor = create_processor_no_cut(tokenizer)
        
        # Load base model first
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        
        print("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        print("  Loading LoRA adapter...")
        model_with_adapter = PeftModel.from_pretrained(base_model, model_path)
        
        # Create wrapper without re-initializing LoRA
        model = IQAModelWrapper.__new__(IQAModelWrapper)
        nn.Module.__init__(model)
        model.model = model_with_adapter
        model.tokenizer = tokenizer
        
        # Initialize quality level configuration
        level_tokens = ["bad", "low", "fair", "good", "awesome"]
        level_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        model.level_tokens = level_tokens
        model.level_scores = torch.tensor(level_scores, dtype=torch.float32)
        
        # Get token IDs for quality levels
        level_words = ["bad", "low", "fair", "good", "awesome"]
        model.level_token_sequences = []
        for word in level_words:
            token_ids = tokenizer.encode(f" {word}", add_special_tokens=False)
            model.level_token_sequences.append(token_ids)
        
        # Loss weights (not used for evaluation but needed for compatibility)
        model.weight_ce = 1.0
        model.weight_kl = 0.05
        model.weight_fidelity = 1.0
        model.use_fix_std = False
        model.detach_pred_std = False
        model.binary_fidelity_type = "fidelity"
    else:
        print("  Loading full model...")
        # Load processor and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = create_processor_no_cut(tokenizer)
        
        # Load model
        model = IQAModelWrapper(model_path)
    
    model.to(args.device)
    model.eval()
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = IQAPairDataset(
        dataset_paths=[Path(p) for p in args.dataset_paths],
        processor=processor,
        tokenizer=tokenizer,
        split=args.split,
        use_scene_labels=True,
        use_distortion_labels=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataset=dataset,
        device=args.device,
        tokenizer=tokenizer,
        processor=processor
    )
    
    # Compute metrics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Combine A and B predictions
    all_predictions = np.concatenate([results['predictions_A'], results['predictions_B']])
    all_gt = np.concatenate([results['gt_scores_A'], results['gt_scores_B']])
    
    metrics = compute_metrics(all_predictions, all_gt)
    
    print(f"\n📊 Overall Performance (Sequential Q&A):")
    print(f"  Samples:    {len(all_predictions)}")
    print(f"  PLCC:       {metrics['plcc']:.4f}  {'█' * int(metrics['plcc'] * 20)}")
    print(f"  SRCC:       {metrics['srcc']:.4f}  {'█' * int(metrics['srcc'] * 20)}")
    print(f"  MAE:        {metrics['mae']:.4f}")
    print(f"  RMSE:       {metrics['rmse']:.4f}")
    
    # Plot scatter plot: GT vs Predicted Quality
    print("\n📈 Generating scatter plot...")
    output_plot_path = Path(args.model_path) / "eval_scatter_plot.png"
    
    plt.figure(figsize=(10, 10))
    plt.scatter(all_gt, all_predictions, alpha=0.5, s=20, edgecolors='none')
    
    # Plot diagonal line (perfect prediction)
    min_val = min(all_gt.min(), all_predictions.min())
    max_val = max(all_gt.max(), all_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('Ground Truth Quality Score', fontsize=14)
    plt.ylabel('Predicted Quality Score', fontsize=14)
    plt.title(f'Quality Score Prediction\nPLCC: {metrics["plcc"]:.4f}, SRCC: {metrics["srcc"]:.4f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Scatter plot saved to: {output_plot_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
