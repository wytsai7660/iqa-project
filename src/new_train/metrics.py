"""
Evaluation metrics for IQA model
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, Tuple


def compute_quality_score_from_logits(
    logits: torch.Tensor,
    level_positions: torch.Tensor,
    level_token_sequences: list,
) -> torch.Tensor:
    """
    Compute quality score from logits at level token position.
    
    The score is computed as the expected value of the discrete distribution:
    E[score] = Σ p_i * score_i, where score_i ∈ [1, 2, 3, 4, 5]
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        level_positions: Position of level token in each sequence [batch_size]
        level_token_sequences: List of token sequences for each quality level
                               [[bad_tokens], [low_tokens], [fair_tokens], [good_tokens], [awesome_tokens]]
    
    Returns:
        scores: Predicted quality scores [batch_size]
    """
    import torch.nn.functional as F
    
    batch_size = logits.shape[0]
    
    # Extract logits at level token positions (position before the level token)
    level_logits = logits[torch.arange(batch_size), level_positions - 1]  # [batch_size, vocab_size]
    
    # Get probabilities for level tokens only (closed-set softmax)
    # All quality words are single tokens now
    level_logits_subset = torch.stack([
        level_logits[:, token_seq[0]] for token_seq in level_token_sequences
    ], dim=1)  # [batch_size, 5]
    
    probs = F.softmax(level_logits_subset, dim=1)  # [batch_size, 5]
    
    # Quality scores corresponding to [bad, low, fair, good, awesome]
    level_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=probs.device)
    
    # Compute expected score: E[score] = Σ p_i * score_i
    scores = torch.matmul(probs, level_scores)  # [batch_size]
    
    return scores


def compute_iqa_metrics(
    pred_scores: np.ndarray,
    gt_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute IQA evaluation metrics.
    
    Args:
        pred_scores: Predicted quality scores [N]
        gt_scores: Ground truth quality scores [N]
    
    Returns:
        Dictionary containing:
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - plcc: Pearson Linear Correlation Coefficient
        - srcc: Spearman Rank Correlation Coefficient
    """
    # Ensure numpy arrays
    pred_scores = np.array(pred_scores).flatten()
    gt_scores = np.array(gt_scores).flatten()
    
    # Check for valid data
    if len(pred_scores) == 0 or len(gt_scores) == 0:
        print("⚠️  Warning: Empty predictions or labels, returning zero metrics")
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'plcc': 0.0,
            'srcc': 0.0,
        }
    
    if len(pred_scores) != len(gt_scores):
        print(f"⚠️  Warning: Prediction and label lengths mismatch ({len(pred_scores)} vs {len(gt_scores)})")
        min_len = min(len(pred_scores), len(gt_scores))
        pred_scores = pred_scores[:min_len]
        gt_scores = gt_scores[:min_len]
    
    # Check for NaN or Inf values
    valid_mask = np.isfinite(pred_scores) & np.isfinite(gt_scores)
    if not valid_mask.all():
        print(f"⚠️  Warning: Found {(~valid_mask).sum()} invalid values, filtering them out")
        pred_scores = pred_scores[valid_mask]
        gt_scores = gt_scores[valid_mask]
    
    if len(pred_scores) < 2:
        print("⚠️  Warning: Less than 2 valid samples, cannot compute correlation")
        return {
            'mae': float(np.mean(np.abs(pred_scores - gt_scores))) if len(pred_scores) > 0 else 0.0,
            'mse': float(np.mean((pred_scores - gt_scores) ** 2)) if len(pred_scores) > 0 else 0.0,
            'rmse': float(np.sqrt(np.mean((pred_scores - gt_scores) ** 2))) if len(pred_scores) > 0 else 0.0,
            'plcc': 0.0,
            'srcc': 0.0,
        }
    
    # MAE
    mae = np.mean(np.abs(pred_scores - gt_scores))
    
    # MSE
    mse = np.mean((pred_scores - gt_scores) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # PLCC (Pearson Linear Correlation Coefficient)
    try:
        # Check if there's variance in the data (needed for correlation)
        if np.std(pred_scores) < 1e-10 or np.std(gt_scores) < 1e-10:
            print("⚠️  Warning: Zero variance in predictions or labels, PLCC set to 0")
            plcc = 0.0
        else:
            plcc, _ = stats.pearsonr(pred_scores, gt_scores)
            # Handle potential NaN from pearsonr
            if not np.isfinite(plcc):
                print("⚠️  Warning: PLCC computation resulted in NaN/Inf, setting to 0")
                plcc = 0.0
    except Exception as e:
        print(f"⚠️  Warning: PLCC computation failed: {e}, setting to 0")
        plcc = 0.0
    
    # SRCC (Spearman Rank Correlation Coefficient)
    try:
        srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        # Handle potential NaN from spearmanr
        if not np.isfinite(srcc):
            print("⚠️  Warning: SRCC computation resulted in NaN/Inf, setting to 0")
            srcc = 0.0
    except Exception as e:
        print(f"⚠️  Warning: SRCC computation failed: {e}, setting to 0")
        srcc = 0.0
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'plcc': float(plcc),
        'srcc': float(srcc),
    }


def extract_predictions_and_labels(
    predictions,
    label_ids,
    model,
    tokenizer,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract predicted scores and ground truth scores from model predictions.
    
    Note: For ground truth, we cannot reliably extract the MOS from label_ids alone.
    This function extracts quality levels from the predicted tokens.
    For accurate evaluation, we should pass gt_scores directly from the dataset.
    
    Args:
        predictions: Model predictions (logits)
        label_ids: Label IDs containing the ground truth tokens
        model: The IQA model with level_token_sequences
        tokenizer: Tokenizer for finding level positions
    
    Returns:
        pred_scores: Predicted quality scores [N]
        gt_level_scores: Ground truth level-based scores [N] (approximation)
    """
    # predictions is a tuple of (logits,) or just logits
    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions
    
    logits = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
    label_ids = torch.from_numpy(label_ids) if isinstance(label_ids, np.ndarray) else label_ids
    
    # Find level token positions in labels
    level_positions = model.find_level_token_position(label_ids)
    
    if level_positions is None:
        # If no level positions found, return empty arrays
        return np.array([]), np.array([])
    
    # Get valid positions
    valid_mask = level_positions >= 0
    if not valid_mask.any():
        return np.array([]), np.array([])
    
    # Compute predicted scores from logits
    pred_scores = compute_quality_score_from_logits(
        logits[valid_mask],
        level_positions[valid_mask],
        model.level_token_sequences,
    )
    
    # Extract ground truth scores from labels
    # The ground truth is encoded in the label at level_position
    # We map the token ID to the quality level [1-5]
    # Note: This gives us the discrete level, not the continuous MOS
    gt_scores = []
    valid_indices = torch.where(valid_mask)[0]
    
    for idx in valid_indices:
        i = idx.item()
        pos = level_positions[i].item()
        
        if pos >= 0 and pos < label_ids.shape[1]:
            token_id = label_ids[i, pos].item()
            # Find which quality level this token belongs to
            found = False
            for level_idx, token_seq in enumerate(model.level_token_sequences):
                if token_id in token_seq:
                    # Map level index to score: [0,1,2,3,4] -> [5,4,3,2,1]
                    score = 5.0 - level_idx
                    gt_scores.append(score)
                    found = True
                    break
            
            if not found:
                # Token not found, this shouldn't happen but handle it
                gt_scores.append(3.0)  # Use neutral score
    
    gt_scores = torch.tensor(gt_scores, device=pred_scores.device, dtype=pred_scores.dtype)
    
    # Ensure lengths match
    if len(gt_scores) != len(pred_scores):
        min_len = min(len(gt_scores), len(pred_scores))
        pred_scores = pred_scores[:min_len]
        gt_scores = gt_scores[:min_len]
    
    pred_scores = pred_scores.cpu().numpy()
    gt_scores = gt_scores.cpu().numpy()
    
    return pred_scores, gt_scores


def create_compute_metrics_fn(model, tokenizer):
    """
    Create a compute_metrics function for Trainer.
    
    Args:
        model: The IQA model
        tokenizer: The tokenizer
    
    Returns:
        compute_metrics function that takes EvalPrediction and returns metrics dict
    """
    def compute_metrics(eval_pred):
        """
        Compute metrics for evaluation.
        
        Args:
            eval_pred: EvalPrediction object with predictions and label_ids
        
        Returns:
            Dictionary of metrics
        """
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
        
        # Extract predicted scores and ground truth scores
        pred_scores, gt_scores = extract_predictions_and_labels(
            predictions, label_ids, model, tokenizer
        )
        
        if len(pred_scores) == 0:
            # No valid predictions
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'plcc': 0.0,
                'srcc': 0.0,
            }
        
        # Compute IQA metrics
        metrics = compute_iqa_metrics(pred_scores, gt_scores)
        
        return metrics
    
    return compute_metrics


def evaluate_model_metrics(
    model,
    dataset,
    processor,
    tokenizer,
    batch_size: int = 4,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and compute PLCC, SRCC, MAE, RMSE metrics.
    This function runs inference on the entire dataset after training completes.
    
    Args:
        model: The IQA model wrapper
        dataset: Validation dataset
        processor: Image processor
        tokenizer: Tokenizer
        batch_size: Batch size for inference
    
    Returns:
        Dictionary containing all evaluation metrics including PLCC and SRCC
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Set model to eval mode
    model.eval()
    device = next(model.model.parameters()).device
    
    # Create dataloader
    def collate_fn(batch):
        """Simple collate function"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
    )
    
    pred_scores_list = []
    gt_scores_list = []
    total_loss = 0.0
    num_batches = 0
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Accumulate loss
            if 'loss' in outputs:
                total_loss += outputs['loss'].item()
                num_batches += 1
            
            # Extract logits
            logits = outputs.get('logits')
            labels = batch.get('labels')
            
            if logits is not None and labels is not None:
                # Find level token positions
                level_positions = model.find_level_token_position(labels)
                
                if level_positions is not None and (level_positions >= 0).any():
                    valid_mask = level_positions >= 0
                    
                    if valid_mask.any():
                        # Compute predicted scores
                        pred_scores = compute_quality_score_from_logits(
                            logits[valid_mask],
                            level_positions[valid_mask],
                            model.level_token_sequences,
                        )
                        
                        # Extract ground truth scores
                        gt_scores = []
                        valid_indices = torch.where(valid_mask)[0]
                        for idx in valid_indices:
                            i = idx.item()
                            pos = level_positions[i].item()
                            if pos >= 0 and pos < labels.shape[1]:
                                token_id = labels[i, pos].item()
                                for level_idx, token_seq in enumerate(model.level_token_sequences):
                                    if token_id in token_seq:
                                        score = 5.0 - level_idx
                                        gt_scores.append(score)
                                        break
                                else:
                                    gt_scores.append(3.0)
                        
                        if len(gt_scores) > 0:
                            gt_scores = torch.tensor(gt_scores, device=pred_scores.device, dtype=pred_scores.dtype)
                            pred_scores_list.extend(pred_scores.cpu().numpy().tolist())
                            gt_scores_list.extend(gt_scores.cpu().numpy().tolist())
    
    # Compute metrics
    metrics = {}
    
    # Average loss
    if num_batches > 0:
        metrics['loss'] = total_loss / num_batches
    else:
        metrics['loss'] = 0.0
    
    # Compute IQA metrics (PLCC, SRCC, MAE, RMSE)
    if len(pred_scores_list) > 0:
        pred_scores = np.array(pred_scores_list)
        gt_scores = np.array(gt_scores_list)
        
        iqa_metrics = compute_iqa_metrics(pred_scores, gt_scores)
        metrics.update(iqa_metrics)
    else:
        print("⚠️  Warning: No valid predictions collected!")
        metrics.update({
            'mae': 0.0,
            'rmse': 0.0,
            'plcc': 0.0,
            'srcc': 0.0,
        })
    
    return metrics
