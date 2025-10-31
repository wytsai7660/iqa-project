#!/usr/bin/env python3
"""
Test script to verify that split='full' works correctly
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from src.new_train.processor_no_cut import create_processor_no_cut
from src.new_train.dataset_adapter import IQAPairDataset

def test_full_split():
    """Test that split='full' returns all data"""
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained("src/owl3", trust_remote_code=True)
    processor = create_processor_no_cut(tokenizer)
    
    # Create datasets for different splits
    dataset_paths = ["datasets/koniq-10k/"]
    
    print("Creating datasets with different splits...")
    
    # Individual splits
    train_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="training",
    )
    
    val_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="validation",
    )
    
    test_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="testing",
    )
    
    # Full split
    full_dataset = IQAPairDataset(
        dataset_paths=dataset_paths,
        processor=processor,
        tokenizer=tokenizer,
        split="full",
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    print(f"Training set:   {len(train_dataset):5d} samples")
    print(f"Validation set: {len(val_dataset):5d} samples")
    print(f"Testing set:    {len(test_dataset):5d} samples")
    print(f"{'─'*80}")
    print(f"Sum of splits:  {len(train_dataset) + len(val_dataset) + len(test_dataset):5d} samples")
    print(f"Full dataset:   {len(full_dataset):5d} samples")
    print("="*80)
    
    # Verify that full dataset contains all data
    expected_total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    actual_total = len(full_dataset)
    
    if actual_total == expected_total:
        print("\n✅ SUCCESS: Full dataset contains all samples!")
        return True
    else:
        print(f"\n❌ ERROR: Expected {expected_total} samples but got {actual_total}")
        return False

if __name__ == "__main__":
    try:
        success = test_full_split()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
