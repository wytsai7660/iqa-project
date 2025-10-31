#!/usr/bin/env python3
"""
Test script to diagnose KL loss NaN issue
"""
import torch
import torch.nn.functional as F

def test_kl_loss():
    print("Testing KL loss computation...")
    
    # Simulate the scenario
    batch_size = 2
    vocab_size = 151936
    
    # Create fake level_probs (should sum to 1)
    level_probs = torch.tensor([
        [0.1, 0.2, 0.3, 0.3, 0.1],  # Valid
        [0.0, 0.0, 1.0, 0.0, 0.0],  # Valid (one-hot)
    ])
    
    # Create fake logits
    level_logits = torch.randn(batch_size, vocab_size)
    
    # Level token IDs
    level_token_ids = [3873, 3347, 6624, 1661, 12456]  # bad, low, fair, good, awesome
    
    # Create target distribution
    target = torch.zeros(batch_size, vocab_size)
    for i, token_id in enumerate(level_token_ids):
        target[:, token_id] = level_probs[:, i]
    
    print(f"level_probs:\n{level_probs}")
    print(f"level_probs sum: {level_probs.sum(dim=1)}")
    print(f"target non-zero positions: {(target > 0).sum(dim=1)}")
    print(f"target sum: {target.sum(dim=1)}")
    
    # Compute KL divergence
    log_pred = F.log_softmax(level_logits, dim=-1)
    loss_kl = F.kl_div(log_pred, target, reduction="batchmean")
    
    print(f"\nKL loss: {loss_kl}")
    print(f"Is NaN: {torch.isnan(loss_kl)}")
    
    # Test with problematic case: target doesn't sum to 1
    print("\n" + "="*60)
    print("Testing with invalid target (doesn't sum to 1)...")
    target_bad = torch.zeros(batch_size, vocab_size)
    for i, token_id in enumerate(level_token_ids):
        target_bad[:, token_id] = level_probs[:, i] * 0.5  # Multiply by 0.5
    
    print(f"target_bad sum: {target_bad.sum(dim=1)}")
    loss_kl_bad = F.kl_div(log_pred, target_bad, reduction="batchmean")
    print(f"KL loss (bad target): {loss_kl_bad}")
    print(f"Is NaN: {torch.isnan(loss_kl_bad)}")
    
    # Test with all-zero target
    print("\n" + "="*60)
    print("Testing with all-zero target...")
    target_zero = torch.zeros(batch_size, vocab_size)
    loss_kl_zero = F.kl_div(log_pred, target_zero, reduction="batchmean")
    print(f"KL loss (zero target): {loss_kl_zero}")
    print(f"Is NaN: {torch.isnan(loss_kl_zero)}")

if __name__ == "__main__":
    test_kl_loss()
