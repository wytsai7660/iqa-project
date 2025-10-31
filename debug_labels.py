"""
Debug script to check how labels are being created in the training data.
"""
import torch
from pathlib import Path
from src.new_train.dataset_adapter import IQAPairDataset, collate_fn_pair
from src.dataset import PairDataset
from src.owl3.processing_mplugowl3 import mPLUGOwl3Processor
from src.owl3.image_processing_mplugowl3 import mPLUGOwl3ImageProcessor
from transformers import AutoTokenizer

# Initialize tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained("src/owl3", trust_remote_code=True)
image_processor = mPLUGOwl3ImageProcessor(image_size=378)
processor = mPLUGOwl3Processor(image_processor=image_processor, tokenizer=tokenizer)

# Create dataset
from pathlib import Path
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
print("Sample structure:")
print(f"Keys: {sample.keys()}")
print()

# Check scene task
if "input_ids_scene_A" in sample:
    input_ids = sample["input_ids_scene_A"]
    print("Scene Task (Image A):")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Input IDs: {input_ids[:50]}")  # First 50 tokens
    print()
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"  Decoded text: {decoded[:500]}")  # First 500 chars
    print()

# Now check what the collate function creates as labels
batch = [sample]
collated = collate_fn_pair(batch)

print("=" * 80)
print("After collate_fn_pair:")
if "labels_scene_A" in collated:
    labels = collated["labels_scene_A"]
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels[0, :50]}")  # First 50 tokens
    print()
    
    # Check if labels match input_ids (this is the bug!)
    input_ids_collated = collated["input_ids_scene_A"]
    print(f"Do labels == input_ids? {torch.equal(labels, input_ids_collated)}")
    print()
    
    # Count non-ignore tokens
    non_ignore = (labels[0] != -100).sum().item()
    total = labels[0].shape[0]
    print(f"Non-ignore tokens: {non_ignore}/{total} ({100*non_ignore/total:.1f}%)")
    print()
    
    # Find where labels differ from input_ids (should be many places!)
    diff_mask = labels[0] != input_ids_collated[0]
    diff_indices = torch.where(diff_mask)[0]
    print(f"Positions where labels != input_ids: {diff_indices.tolist()[:20]}")  # First 20
    print()

print("=" * 80)
print("\nThis debug shows that labels are incorrectly set to input_ids.")
print("The processor creates proper labels (0 for questions, 1 for answers),")
print("but doesn't return them. The collate function then just uses input_ids.")
print("\nThis means the model is being trained to predict ALL tokens,")
print("including the question tokens, which is wrong!")
