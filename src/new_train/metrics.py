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
