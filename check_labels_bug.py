"""
Debug script to check the labels bug in dataset_adapter.py
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer
from src.owl3.processing_mplugowl3 import mPLUGOwl3Processor
from src.owl3.image_processing_mplugowl3 import mPLUGOwl3ImageProcessor
from src.dataset import PairDataset

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

# Get one item
print("Getting one item from PairDataset...")
pair_item = base_dataset[0]

# Check what keys are in the processed messages
scene_msg = pair_item["image_1"]["scene_type_message"]
print("\nKeys in scene_type_message:")
print(scene_msg.keys())

# Check input_ids
if "input_ids" in scene_msg:
    input_ids = scene_msg["input_ids"][0]
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"First 50 input IDs: {input_ids[:50].tolist()}")
    
    # Decode
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"\nDecoded text (first 500 chars):\n{decoded[:500]}")

# Check if labels are present
if "labels" in scene_msg:
    print("\n✓ Labels ARE present in the processor output!")
    labels = scene_msg["labels"][0]
    print(f"Labels shape: {labels.shape}")
    print(f"First 50 labels: {labels[:50].tolist()}")
else:
    print("\n✗ Labels are NOT present in the processor output!")
    print("This means the processor is not returning the label_chunk it creates.")

print("\n" + "=" * 80)
print("Checking the collate_fn_pair function in dataset_adapter.py...")
print("Line 171 does: padded_labels = torch.cat([item[input_ids_key], ...])")
print("This means labels are set to input_ids, NOT actual labels!")
print("\nThis is the bug: The model is trained to predict ALL tokens,")
print("including question tokens, not just answer tokens.")
