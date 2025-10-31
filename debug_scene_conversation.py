"""
Debug: Check what's in the scene conversation
"""
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
pair_item = base_dataset[0]
image_A = pair_item["image_1"]

# Check scene_type_message
scene_msg = image_A["scene_type_message"]
input_ids = scene_msg["input_ids"][0]
labels = scene_msg["labels"][0]

print("Scene conversation:")
print("=" * 80)
decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
print(decoded)
print()

print("=" * 80)
print("Labels (0=ignore, 1=train):")
print(f"Total tokens: {len(labels)}")
print(f"Ignore (0): {(labels == 0).sum().item()}")
print(f"Train (1): {(labels == 1).sum().item()}")
print()

# Find where labels == 1
train_positions = (labels == 1).nonzero(as_tuple=True)[0]
if len(train_positions) > 0:
    print(f"Training on {len(train_positions)} tokens at positions: {train_positions[:20].tolist()}")
    print()
    print("Tokens being trained:")
    for pos in train_positions[:20]:
        token_id = input_ids[pos].item()
        token_text = tokenizer.decode([token_id])
        print(f"  Position {pos}: token_id={token_id}, text='{token_text}'")
else:
    print("❌ NO TOKENS ARE BEING TRAINED!")
    print()
    print("This means the assistant's answer is missing or empty.")
    print("Let me check the raw scene_qa...")
    
# Check what PairDataset is creating
print("\n" + "=" * 80)
print("Checking PairDataset.get_one_image()...")
print()

# Access the internal method to see the messages before processing
import src.dataset as dataset_module
# Get the raw data
dataset_idx = 0
dataset_path_idx = 0
dataset_df = base_dataset.dataset_labels_data_frames[dataset_path_idx]
filename = dataset_df.index[dataset_idx]

print(f"Filename: {filename}")
print(f"Scene type from CSV: {dataset_df.loc[filename, 'scene']}")
print()

# Check if scene is empty
scene_type = dataset_df.loc[filename, "scene"]
if not scene_type or scene_type.strip() == "":
    print("❌ Scene type is EMPTY in the CSV!")
    print("This explains why there are no answer tokens to train on.")
