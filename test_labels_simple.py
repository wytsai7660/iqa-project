"""
Simple test to verify labels fix works
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer
from src.owl3.processing_mplugowl3 import mPLUGOwl3Processor
from src.owl3.image_processing_mplugowl3 import mPLUGOwl3ImageProcessor
from src.new_train.dataset_adapter import IQAPairDataset, collate_fn_pair

# Initialize
tokenizer = AutoTokenizer.from_pretrained("src/owl3", trust_remote_code=True)
image_processor = mPLUGOwl3ImageProcessor(image_size=378)
processor = mPLUGOwl3Processor(image_processor=image_processor, tokenizer=tokenizer)

# Create dataset
dataset = IQAPairDataset(
    dataset_paths=["datasets/koniq-10k"],
    processor=processor,
    tokenizer=tokenizer,
    split="validation",
    use_scene_labels=True,
    use_distortion_labels=False
)

# Get one sample
print("Getting sample...")
sample = dataset[0]

print("\n" + "=" * 80)
print("Checking if labels are properly extracted from processor output:")
print()

has_labels = "labels_scene_A" in sample
print(f"✓ labels_scene_A present: {has_labels}")

if has_labels:
    labels = sample["labels_scene_A"]
    input_ids = sample["input_ids_scene_A"]
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels type: {labels.dtype}")
    print(f"  First 20 labels: {labels[:20].tolist()}")
    print(f"  Labels are all 0/1: {((labels == 0) | (labels == 1)).all().item()}")
    print()

# Test collate function
print("Testing collate_fn_pair...")
batch = [sample]
collated = collate_fn_pair(batch)

print(f"✓ labels_scene_A in collated: {'labels_scene_A' in collated}")

if "labels_scene_A" in collated:
    labels_collated = collated["labels_scene_A"]
    input_ids_collated = collated["input_ids_scene_A"]
    
    print(f"  Collated labels shape: {labels_collated.shape}")
    print(f"  First 20 collated labels: {labels_collated[0, :20].tolist()}")
    print()
    
    # Count -100 and trained tokens
    num_ignore = (labels_collated[0] == -100).sum().item()
    num_trained = ((labels_collated[0] != -100) & (labels_collated[0] != 0)).sum().item()
    total = labels_collated.shape[1]
    
    print(f"Token statistics:")
    print(f"  -100 (ignore): {num_ignore}/{total} ({100*num_ignore/total:.1f}%)")
    print(f"  Trained: {num_trained}/{total} ({100*num_trained/total:.1f}%)")
    print()
    
    if num_trained > 0 and num_ignore > num_trained:
        print("✓✓✓ SUCCESS! Labels are working correctly:")
        print("    - Most tokens are -100 (will be ignored)")
        print("    - Some tokens are trained (answer tokens)")
        print("    - This is the correct behavior!")
    else:
        print("❌ WARNING: Label distribution seems wrong")
        print(f"    - Expected: ~70-80% tokens with -100")
        print(f"    - Got: {100*num_ignore/total:.1f}%")
