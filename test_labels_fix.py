"""
Test script to verify the labels fix in dataset_adapter.py
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer
from src.owl3.processing_mplugowl3 import mPLUGOwl3Processor
from src.owl3.image_processing_mplugowl3 import mPLUGOwl3ImageProcessor
from src.dataset import PairDataset
from src.new_train.dataset_adapter import IQAPairDataset, collate_fn_pair

# Initialize
tokenizer = AutoTokenizer.from_pretrained("src/owl3", trust_remote_code=True)
image_processor = mPLUGOwl3ImageProcessor(image_size=378)
processor = mPLUGOwl3Processor(image_processor=image_processor, tokenizer=tokenizer)

# Create dataset
base_dataset = PairDataset(
    dataset_paths=[Path("datasets/koniq-10k")],
    processor=processor,
    tokenizer=tokenizer,
    split="validation"
)

dataset = IQAPairDataset(
    base_dataset,
    use_scene_labels=True,
    use_distortion_labels=False
)

# Get one sample
sample = dataset[0]

print("=" * 80)
print("Checking labels in dataset sample:")
print()

# Check scene task
if "input_ids_scene_A" in sample and "labels_scene_A" in sample:
    input_ids = sample["input_ids_scene_A"]
    labels = sample["labels_scene_A"]
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    print()
    
    print(f"First 50 input IDs: {input_ids[:50].tolist()}")
    print(f"First 50 labels: {labels[:50].tolist()}")
    print()
    
    # Check if they're different (they should be!)
    if torch.equal(input_ids, labels):
        print("❌ ERROR: Labels are still equal to input_ids!")
    else:
        print("✓ Labels are different from input_ids (as expected)")
    print()
    
    # Decode to see what's being trained
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("Full conversation:")
    print(decoded_input)
    print()
    
    # Find where labels change from 0 to 1
    label_changes = torch.where(labels[:-1] != labels[1:])[0]
    print(f"Label changes at positions: {label_changes.tolist()}")
    print()
    
    # Check label distribution
    num_ignore = (labels == 0).sum().item()
    num_train = (labels == 1).sum().item()
    total = labels.shape[0]
    print(f"Label=0 (ignore): {num_ignore}/{total} ({100*num_ignore/total:.1f}%)")
    print(f"Label=1 (train): {num_train}/{total} ({100*num_train/total:.1f}%)")
    print()

# Test collate function
print("=" * 80)
print("Testing collate_fn_pair:")
print()

batch = [sample]
collated = collate_fn_pair(batch)

if "labels_scene_A" in collated:
    labels_collated = collated["labels_scene_A"]
    input_ids_collated = collated["input_ids_scene_A"]
    
    print(f"Collated labels shape: {labels_collated.shape}")
    print(f"First 50 collated labels: {labels_collated[0, :50].tolist()}")
    print()
    
    # Count -100 (ignore tokens)
    num_ignore = (labels_collated[0] == -100).sum().item()
    # Count trained tokens (not -100 and not padding)
    num_trained = ((labels_collated[0] != -100) & (labels_collated[0] != 0)).sum().item()
    total = labels_collated[0].shape[0]
    
    print(f"After collate:")
    print(f"  -100 (ignore): {num_ignore}/{total} ({100*num_ignore/total:.1f}%)")
    print(f"  Trained tokens: {num_trained}/{total} ({100*num_trained/total:.1f}%)")
    print()
    
    # Verify labels are correct for answer tokens
    # Find answer positions (where original labels == 1)
    if "labels_scene_A" in sample:
        original_labels = sample["labels_scene_A"]
        answer_positions = torch.where(original_labels == 1)[0]
        
        if len(answer_positions) > 0:
            # Check a few answer positions
            pos_to_check = answer_positions[:5].tolist()
            print(f"Checking answer token positions: {pos_to_check}")
            for pos in pos_to_check:
                input_token = input_ids_collated[0, pos].item()
                label_token = labels_collated[0, pos].item()
                if input_token == label_token:
                    print(f"  ✓ Position {pos}: input={input_token}, label={label_token} (correct)")
                else:
                    print(f"  ❌ Position {pos}: input={input_token}, label={label_token} (WRONG!)")
            print()
        
        # Check question positions (where original labels == 0)
        question_positions = torch.where(original_labels == 0)[0]
        if len(question_positions) > 0:
            pos_to_check = question_positions[:5].tolist()
            print(f"Checking question token positions: {pos_to_check}")
            for pos in pos_to_check:
                label_token = labels_collated[0, pos].item()
                if label_token == -100:
                    print(f"  ✓ Position {pos}: label=-100 (correct, will be ignored)")
                else:
                    print(f"  ❌ Position {pos}: label={label_token} (WRONG! should be -100)")
            print()
    
    print("=" * 80)
    print("\nSUMMARY:")
    if num_trained > 0 and num_ignore > num_trained:
        print("✓✓✓ The fix is working correctly!")
        print("  - Labels are properly created from raw labels (0/1)")
        print("  - Question tokens have label=-100 (will be ignored in loss)")
        print("  - Answer tokens have label=token_id (will be trained)")
    else:
        print("❌ Something is still wrong with the labels")
