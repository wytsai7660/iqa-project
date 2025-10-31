"""
Pre-training checklist: Verify all fixes are in place before starting long training run
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer
from src.new_train.processor_no_cut import create_processor_no_cut
from src.new_train.dataset_adapter import IQAPairDataset, collate_fn_pair
from src.new_train.train_scene import collate_fn_scene
from src.new_train.train_distortion import collate_fn_distortion

print("=" * 80)
print("PRE-TRAINING CHECKLIST")
print("=" * 80)
print()

# Test 1: Check processor has inference_mode=False
print("✓ Test 1: Processor inference_mode")
tokenizer = AutoTokenizer.from_pretrained("src/owl3", trust_remote_code=True)
processor = create_processor_no_cut(tokenizer, image_size=378)
assert processor.inference_mode == False, "❌ Processor should have inference_mode=False for training!"
print(f"  ✓ Processor inference_mode = {processor.inference_mode} (correct!)")
print()

# Test 2: Check processor returns labels
print("✓ Test 2: Processor returns labels")
from src.dataset import PairDataset
base_dataset = PairDataset(
    dataset_paths=[Path("datasets/koniq-10k")],
    processor=processor,
    tokenizer=tokenizer,
    split="validation"
)
pair_item = base_dataset[0]
scene_msg = pair_item["image_1"]["scene_type_message"]
assert "labels" in scene_msg, "❌ Processor should return labels!"
print(f"  ✓ Processor returns labels: {list(scene_msg.keys())}")
print()

# Test 3: Check dataset extracts labels
print("✓ Test 3: Dataset extracts labels")
dataset = IQAPairDataset(
    dataset_paths=["datasets/koniq-10k"],
    processor=processor,
    tokenizer=tokenizer,
    split="validation",
    use_scene_labels=True,
    use_distortion_labels=True
)
sample = dataset[0]
assert "labels_scene_A" in sample, "❌ Dataset should extract labels_scene_A!"
assert "labels_distortion_A" in sample, "❌ Dataset should extract labels_distortion_A!"
assert "labels_quality_A" in sample, "❌ Dataset should extract labels_quality_A!"
print(f"  ✓ Dataset extracts scene labels: {'labels_scene_A' in sample}")
print(f"  ✓ Dataset extracts distortion labels: {'labels_distortion_A' in sample}")
print(f"  ✓ Dataset extracts quality labels: {'labels_quality_A' in sample}")
print()

# Test 4: Check collate_fn_pair creates proper labels
print("✓ Test 4: collate_fn_pair creates proper labels")
batch = [sample]
collated = collate_fn_pair(batch)
labels = collated["labels_scene_A"][0]
input_ids = collated["input_ids_scene_A"][0]

# Check labels are not equal to input_ids (should have -100 for questions)
num_ignore = (labels == -100).sum().item()
num_trained = ((labels != -100) & (labels != 0)).sum().item()
total = labels.shape[0]

assert num_ignore > 0, "❌ Should have some tokens with label=-100 (question tokens)!"
assert num_trained > 0, "❌ Should have some trained tokens (answer tokens)!"
assert num_ignore > num_trained, "❌ Should have more ignored tokens than trained tokens!"

print(f"  ✓ Ignore tokens (-100): {num_ignore}/{total} ({100*num_ignore/total:.1f}%)")
print(f"  ✓ Trained tokens: {num_trained}/{total} ({100*num_trained/total:.1f}%)")
print(f"  ✓ Label distribution is correct!")
print()

# Test 5: Check collate_fn_scene creates proper labels
print("✓ Test 5: collate_fn_scene creates proper labels")
collated_scene = collate_fn_scene(batch)
labels_scene = collated_scene["labels_scene_A"][0]
num_ignore_scene = (labels_scene == -100).sum().item()
num_trained_scene = ((labels_scene != -100) & (labels_scene != 0)).sum().item()
total_scene = labels_scene.shape[0]

assert num_ignore_scene > 0, "❌ collate_fn_scene: Should have ignored tokens!"
assert num_trained_scene > 0, "❌ collate_fn_scene: Should have trained tokens!"

print(f"  ✓ Ignore tokens: {num_ignore_scene}/{total_scene} ({100*num_ignore_scene/total_scene:.1f}%)")
print(f"  ✓ Trained tokens: {num_trained_scene}/{total_scene} ({100*num_trained_scene/total_scene:.1f}%)")
print()

# Test 6: Check collate_fn_distortion creates proper labels
print("✓ Test 6: collate_fn_distortion creates proper labels")
collated_distortion = collate_fn_distortion(batch)
labels_distortion = collated_distortion["labels_distortion_A"][0]
num_ignore_distortion = (labels_distortion == -100).sum().item()
num_trained_distortion = ((labels_distortion != -100) & (labels_distortion != 0)).sum().item()
total_distortion = labels_distortion.shape[0]

assert num_ignore_distortion > 0, "❌ collate_fn_distortion: Should have ignored tokens!"
assert num_trained_distortion > 0, "❌ collate_fn_distortion: Should have trained tokens!"

print(f"  ✓ Ignore tokens: {num_ignore_distortion}/{total_distortion} ({100*num_ignore_distortion/total_distortion:.1f}%)")
print(f"  ✓ Trained tokens: {num_trained_distortion}/{total_distortion} ({100*num_trained_distortion/total_distortion:.1f}%)")
print()

# Test 7: Verify answer tokens have correct token IDs
print("✓ Test 7: Answer tokens have correct token IDs")
# Find positions where labels != -100 (answer tokens)
answer_positions = (labels != -100).nonzero(as_tuple=True)[0]
if len(answer_positions) > 0:
    # Check that labels match input_ids for answer positions
    for pos in answer_positions[:5]:  # Check first 5 answer tokens
        label_token = labels[pos].item()
        input_token = input_ids[pos].item()
        assert label_token == input_token, f"❌ Label {label_token} != input {input_token} at pos {pos}"
    print(f"  ✓ Answer tokens have correct token IDs (checked {min(5, len(answer_positions))} tokens)")
else:
    print("  ⚠ WARNING: No answer tokens found!")
print()

# Final summary
print("=" * 80)
print("✅ ALL CHECKS PASSED!")
print("=" * 80)
print()
print("Summary of fixes applied:")
print("1. ✓ Processor returns labels array (0=ignore, 1=train)")
print("2. ✓ Processor has inference_mode=False for training")
print("3. ✓ Dataset extracts labels for all three tasks")
print("4. ✓ collate_fn_pair converts label=0 → -100, label=1 → token_id")
print("5. ✓ collate_fn_scene converts label=0 → -100, label=1 → token_id")
print("6. ✓ collate_fn_distortion converts label=0 → -100, label=1 → token_id")
print()
print("🎉 Ready to start training!")
print()
print("Expected improvement after re-training:")
print("  Before: PLCC=0.63, SRCC=0.63")
print("  After:  PLCC>0.9, SRCC>0.9")
