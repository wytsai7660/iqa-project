#!/usr/bin/env python3
"""
Test script to verify that dataset correctly includes/excludes context
based on use_scene_labels and use_distortion_labels flags.
"""

from pathlib import Path
from transformers import AutoTokenizer
from src.owl3.image_processing_mplugowl3 import mPLUGOwl3ImageProcessor
from src.owl3.processing_mplugowl3 import mPLUGOwl3Processor
from src.dataset import PairDataset

MODEL_DIR = "./src/owl3"

# Initialize tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
image_processor = mPLUGOwl3ImageProcessor(image_size=378)
processor = mPLUGOwl3Processor(image_processor=image_processor, tokenizer=tokenizer)

print("="*80)
print("Testing dataset context inclusion based on flags")
print("="*80)

# Test Case 1: WITH_SCENE=False, WITH_DISTORTION=True
print("\n1. Testing WITH_SCENE=False, WITH_DISTORTION=True")
print("-" * 60)

dataset = PairDataset(
    dataset_paths=[Path("datasets/live")],
    processor=processor,
    tokenizer=tokenizer,
    split="training",
    use_scene_labels=False,
    use_distortion_labels=True,
)

pair_item = dataset[0]
image_data = pair_item["image_1"]

# Decode messages to check content
distortion_input_ids = image_data["distortion_type_message"]["input_ids"][0]
distortion_text = tokenizer.decode(distortion_input_ids, skip_special_tokens=False)

quality_input_ids = image_data["quality_message"]["input_ids"][0]
quality_text = tokenizer.decode(quality_input_ids, skip_special_tokens=False)

print(f"Distortion message contains 'scene type': {'scene type' in distortion_text.lower()}")
print(f"Quality message contains 'scene type': {'scene type' in quality_text.lower()}")
print(f"Quality message contains 'distortion type': {'distortion type' in quality_text.lower()}")

assert 'scene type' not in distortion_text.lower(), "Distortion should NOT contain scene context when WITH_SCENE=False"
assert 'scene type' not in quality_text.lower(), "Quality should NOT contain scene context when WITH_SCENE=False"
assert 'distortion type' in quality_text.lower(), "Quality SHOULD contain distortion context when WITH_DISTORTION=True"

print("✅ Test Case 1 passed!")

# Test Case 2: WITH_SCENE=True, WITH_DISTORTION=False
print("\n2. Testing WITH_SCENE=True, WITH_DISTORTION=False")
print("-" * 60)

dataset = PairDataset(
    dataset_paths=[Path("datasets/live")],
    processor=processor,
    tokenizer=tokenizer,
    split="training",
    use_scene_labels=True,
    use_distortion_labels=False,
)

pair_item = dataset[0]
image_data = pair_item["image_1"]

distortion_input_ids = image_data["distortion_type_message"]["input_ids"][0]
distortion_text = tokenizer.decode(distortion_input_ids, skip_special_tokens=False)

quality_input_ids = image_data["quality_message"]["input_ids"][0]
quality_text = tokenizer.decode(quality_input_ids, skip_special_tokens=False)

print(f"Distortion message contains 'scene type': {'scene type' in distortion_text.lower()}")
print(f"Quality message contains 'scene type': {'scene type' in quality_text.lower()}")
print(f"Quality message contains 'distortion type': {'distortion type' in quality_text.lower()}")

assert 'scene type' in distortion_text.lower(), "Distortion SHOULD contain scene context when WITH_SCENE=True"
assert 'scene type' in quality_text.lower(), "Quality SHOULD contain scene context when WITH_SCENE=True"
assert 'distortion type' not in quality_text.lower(), "Quality should NOT contain distortion context when WITH_DISTORTION=False"

print("✅ Test Case 2 passed!")

# Test Case 3: WITH_SCENE=True, WITH_DISTORTION=True
print("\n3. Testing WITH_SCENE=True, WITH_DISTORTION=True")
print("-" * 60)

dataset = PairDataset(
    dataset_paths=[Path("datasets/live")],
    processor=processor,
    tokenizer=tokenizer,
    split="training",
    use_scene_labels=True,
    use_distortion_labels=True,
)

pair_item = dataset[0]
image_data = pair_item["image_1"]

distortion_input_ids = image_data["distortion_type_message"]["input_ids"][0]
distortion_text = tokenizer.decode(distortion_input_ids, skip_special_tokens=False)

quality_input_ids = image_data["quality_message"]["input_ids"][0]
quality_text = tokenizer.decode(quality_input_ids, skip_special_tokens=False)

print(f"Distortion message contains 'scene type': {'scene type' in distortion_text.lower()}")
print(f"Quality message contains 'scene type': {'scene type' in quality_text.lower()}")
print(f"Quality message contains 'distortion type': {'distortion type' in quality_text.lower()}")

assert 'scene type' in distortion_text.lower(), "Distortion SHOULD contain scene context when WITH_SCENE=True"
assert 'scene type' in quality_text.lower(), "Quality SHOULD contain scene context when WITH_SCENE=True"
assert 'distortion type' in quality_text.lower(), "Quality SHOULD contain distortion context when WITH_DISTORTION=True"

print("✅ Test Case 3 passed!")

# Test Case 4: WITH_SCENE=False, WITH_DISTORTION=False
print("\n4. Testing WITH_SCENE=False, WITH_DISTORTION=False (Quality only)")
print("-" * 60)

dataset = PairDataset(
    dataset_paths=[Path("datasets/live")],
    processor=processor,
    tokenizer=tokenizer,
    split="training",
    use_scene_labels=False,
    use_distortion_labels=False,
)

pair_item = dataset[0]
image_data = pair_item["image_1"]

quality_input_ids = image_data["quality_message"]["input_ids"][0]
quality_text = tokenizer.decode(quality_input_ids, skip_special_tokens=False)

print(f"Quality message contains 'scene type': {'scene type' in quality_text.lower()}")
print(f"Quality message contains 'distortion type': {'distortion type' in quality_text.lower()}")

assert 'scene type' not in quality_text.lower(), "Quality should NOT contain scene context when WITH_SCENE=False"
assert 'distortion type' not in quality_text.lower(), "Quality should NOT contain distortion context when WITH_DISTORTION=False"

print("✅ Test Case 4 passed!")

print("\n" + "="*80)
print("🎉 All tests passed! Dataset correctly handles context flags.")
print("="*80)
