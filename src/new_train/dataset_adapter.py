"""
Dataset adapter for IQA training.
Adapts the existing PairDataset to work with the new training framework.
"""
# Set environment variables BEFORE any imports that use tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
from scipy.stats import norm
import numpy as np

from ..dataset import PairDataset
from ..owl3.processing_mplugowl3 import mPLUGOwl3Processor


class IQAPairDataset(Dataset):
    """
    Pair dataset for IQA training with fidelity loss.
    Each sample contains two images from the same dataset for ranking.
    Supports multi-task training: scene type, distortion type, and quality assessment.
    """
    
    def __init__(
        self,
        dataset_paths: List[str],
        processor: mPLUGOwl3Processor,
        tokenizer,
        split: str = "training",
        max_length: int = 512,
        use_scene_labels: bool = False,
        use_distortion_labels: bool = False,
    ):
        self.pair_dataset = PairDataset(
            dataset_paths=[Path(p) for p in dataset_paths],
            processor=processor,
            tokenizer=tokenizer,
            split=split,
            use_scene_labels=use_scene_labels,
            use_distortion_labels=use_distortion_labels,
        )
        self.max_length = max_length
        self.use_scene_labels = use_scene_labels
        self.use_distortion_labels = use_distortion_labels
    
    def __len__(self):
        return len(self.pair_dataset)
    
    def __getitem__(self, idx):
        pair_item = self.pair_dataset[idx]
        
        image_A = pair_item["image_1"]
        image_B = pair_item["image_2"]
        
        # Compute scores and stds from level probabilities
        level_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        level_probs_A = image_A["level_probabilities"]
        gt_score_A = float(np.sum(level_probs_A * level_scores))
        # Add small epsilon to avoid sqrt of negative or zero values
        variance_A = np.sum(level_probs_A * (level_scores - gt_score_A) ** 2)
        gt_std_A = float(np.sqrt(max(variance_A, 0.0) + 1e-8))
        
        level_probs_B = image_B["level_probabilities"]
        gt_score_B = float(np.sum(level_probs_B * level_scores))
        # Add small epsilon to avoid sqrt of negative or zero values
        variance_B = np.sum(level_probs_B * (level_scores - gt_score_B) ** 2)
        gt_std_B = float(np.sqrt(max(variance_B, 0.0) + 1e-8))
        
        # Extract quality task data (main task with KL loss and fidelity loss)
        quality_messages_A = image_A["quality_message"]
        quality_messages_B = image_B["quality_message"]
        
        # Extract scene task data (always present now)
        scene_messages_A = image_A["scene_type_message"]
        scene_messages_B = image_B["scene_type_message"]
        
        # Extract distortion task data (always present now)
        distortion_messages_A = image_A["distortion_type_message"]
        distortion_messages_B = image_B["distortion_type_message"]
        
        # Extract input_ids, pixel_values, and media_offset for quality task
        input_ids_quality_A = quality_messages_A["input_ids"][0] if "input_ids" in quality_messages_A else None
        input_ids_quality_B = quality_messages_B["input_ids"][0] if "input_ids" in quality_messages_B else None
        labels_quality_A = quality_messages_A["labels"][0] if "labels" in quality_messages_A else None
        labels_quality_B = quality_messages_B["labels"][0] if "labels" in quality_messages_B else None
        pixel_values_A = quality_messages_A["pixel_values"][0] if "pixel_values" in quality_messages_A else None
        pixel_values_B = quality_messages_B["pixel_values"][0] if "pixel_values" in quality_messages_B else None
        media_offset_A = quality_messages_A["media_offset"][0] if "media_offset" in quality_messages_A else None
        media_offset_B = quality_messages_B["media_offset"][0] if "media_offset" in quality_messages_B else None
        
        # Create attention masks for quality task
        attention_mask_quality_A = torch.ones_like(input_ids_quality_A) if input_ids_quality_A is not None else None
        attention_mask_quality_B = torch.ones_like(input_ids_quality_B) if input_ids_quality_B is not None else None
        
        result = {
            # Image A - Quality task (main task)
            "pixel_values_A": pixel_values_A,
            "input_ids_quality_A": input_ids_quality_A,
            "labels_quality_A": labels_quality_A,
            "attention_mask_quality_A": attention_mask_quality_A,
            "media_offset_A": media_offset_A,
            "level_probs_A": torch.from_numpy(level_probs_A).float(),
            "gt_scores_A": torch.tensor(gt_score_A).float(),
            "gt_stds_A": torch.tensor(gt_std_A).float(),
            
            # Image B - Quality task (main task)
            "pixel_values_B": pixel_values_B,
            "input_ids_quality_B": input_ids_quality_B,
            "labels_quality_B": labels_quality_B,
            "attention_mask_quality_B": attention_mask_quality_B,
            "media_offset_B": media_offset_B,
            "level_probs_B": torch.from_numpy(level_probs_B).float(),
            "gt_scores_B": torch.tensor(gt_score_B).float(),
            "gt_stds_B": torch.tensor(gt_std_B).float(),
        }
        
        # Add scene task data (always present now)
        input_ids_scene_A = scene_messages_A["input_ids"][0] if "input_ids" in scene_messages_A else None
        labels_scene_A = scene_messages_A["labels"][0] if "labels" in scene_messages_A else None
        attention_mask_scene_A = torch.ones_like(input_ids_scene_A) if input_ids_scene_A is not None else None
        result["input_ids_scene_A"] = input_ids_scene_A
        result["labels_scene_A"] = labels_scene_A
        result["attention_mask_scene_A"] = attention_mask_scene_A
        
        input_ids_scene_B = scene_messages_B["input_ids"][0] if "input_ids" in scene_messages_B else None
        labels_scene_B = scene_messages_B["labels"][0] if "labels" in scene_messages_B else None
        attention_mask_scene_B = torch.ones_like(input_ids_scene_B) if input_ids_scene_B is not None else None
        result["input_ids_scene_B"] = input_ids_scene_B
        result["labels_scene_B"] = labels_scene_B
        result["attention_mask_scene_B"] = attention_mask_scene_B
        
        # Add distortion task data (always present now)
        input_ids_distortion_A = distortion_messages_A["input_ids"][0] if "input_ids" in distortion_messages_A else None
        labels_distortion_A = distortion_messages_A["labels"][0] if "labels" in distortion_messages_A else None
        attention_mask_distortion_A = torch.ones_like(input_ids_distortion_A) if input_ids_distortion_A is not None else None
        result["input_ids_distortion_A"] = input_ids_distortion_A
        result["labels_distortion_A"] = labels_distortion_A
        result["attention_mask_distortion_A"] = attention_mask_distortion_A
        
        input_ids_distortion_B = distortion_messages_B["input_ids"][0] if "input_ids" in distortion_messages_B else None
        labels_distortion_B = distortion_messages_B["labels"][0] if "labels" in distortion_messages_B else None
        attention_mask_distortion_B = torch.ones_like(input_ids_distortion_B) if input_ids_distortion_B is not None else None
        result["input_ids_distortion_B"] = input_ids_distortion_B
        result["labels_distortion_B"] = labels_distortion_B
        result["attention_mask_distortion_B"] = attention_mask_distortion_B
        
        return result


def collate_fn_pair(batch):
    """
    Collate function for pair dataset with multi-task support.
    Handles quality task (main task) and optionally scene/distortion tasks.
    """
    # Helper function to pad and stack sequences
    def pad_and_stack_sequence(batch, input_ids_key, attention_mask_key):
        """Pad sequences to same length and stack them."""
        # Infer labels key from input_ids_key
        labels_key = input_ids_key.replace("input_ids", "labels")
        
        # Find max length
        max_len = max(item[input_ids_key].shape[0] for item in batch if input_ids_key in item and item[input_ids_key] is not None)
        
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for item in batch:
            if input_ids_key in item and item[input_ids_key] is not None:
                seq_len = item[input_ids_key].shape[0]
                pad_len = max_len - seq_len
                
                # Pad input_ids with 0
                padded_input_ids = torch.cat([item[input_ids_key], torch.zeros(pad_len, dtype=item[input_ids_key].dtype)])
                input_ids_list.append(padded_input_ids)
                
                # Pad attention_mask with 0
                padded_attention_mask = torch.cat([item[attention_mask_key], torch.zeros(pad_len, dtype=item[attention_mask_key].dtype)])
                attention_mask_list.append(padded_attention_mask)
                
                # Create proper labels: label=0 → -100 (ignore), label=1 → token_id (train)
                if labels_key in item and item[labels_key] is not None:
                    raw_labels = item[labels_key]  # 0 or 1 for each token
                    input_ids_item = item[input_ids_key]
                    # Convert: label=0 → -100, label=1 → keep token_id
                    proper_labels = torch.where(raw_labels == 0, torch.tensor(-100, dtype=input_ids_item.dtype), input_ids_item)
                    # Pad with -100 (ignore index)
                    padded_labels = torch.cat([proper_labels, torch.full((pad_len,), -100, dtype=input_ids_item.dtype)])
                else:
                    # Fallback: if no labels, use input_ids (old behavior)
                    padded_labels = torch.cat([item[input_ids_key], torch.full((pad_len,), -100, dtype=item[input_ids_key].dtype)])
                labels_list.append(padded_labels)
        
        input_ids = torch.stack(input_ids_list) if len(input_ids_list) > 0 else None
        attention_mask = torch.stack(attention_mask_list) if len(attention_mask_list) > 0 else None
        labels = torch.stack(labels_list) if len(labels_list) > 0 else None
        
        return input_ids, attention_mask, labels
    
    # Process quality task (main task)
    input_ids_quality_A, attention_mask_quality_A, labels_quality_A = pad_and_stack_sequence(
        batch, "input_ids_quality_A", "attention_mask_quality_A"
    )
    input_ids_quality_B, attention_mask_quality_B, labels_quality_B = pad_and_stack_sequence(
        batch, "input_ids_quality_B", "attention_mask_quality_B"
    )
    
    # Extract pixel values and media offsets (shared across tasks)
    pixel_values_A_list = [item["pixel_values_A"] for item in batch if item["pixel_values_A"] is not None]
    pixel_values_B_list = [item["pixel_values_B"] for item in batch if item["pixel_values_B"] is not None]
    media_offset_A_list = [item["media_offset_A"] for item in batch if item["media_offset_A"] is not None]
    media_offset_B_list = [item["media_offset_B"] for item in batch if item["media_offset_B"] is not None]
    
    pixel_values_A = torch.stack(pixel_values_A_list) if len(pixel_values_A_list) > 0 else None
    pixel_values_B = torch.stack(pixel_values_B_list) if len(pixel_values_B_list) > 0 else None
    
    # Extract ground truth data
    level_probs_A = torch.stack([item["level_probs_A"] for item in batch])
    gt_scores_A = torch.stack([item["gt_scores_A"] for item in batch])
    gt_stds_A = torch.stack([item["gt_stds_A"] for item in batch])
    
    level_probs_B = torch.stack([item["level_probs_B"] for item in batch])
    gt_scores_B = torch.stack([item["gt_scores_B"] for item in batch])
    gt_stds_B = torch.stack([item["gt_stds_B"] for item in batch])
    
    result = {
        # Quality task (main task with KL loss and fidelity loss)
        "pixel_values_A": pixel_values_A,
        "input_ids_quality_A": input_ids_quality_A,
        "attention_mask_quality_A": attention_mask_quality_A,
        "labels_quality_A": labels_quality_A,
        "media_offset_A": media_offset_A_list if len(media_offset_A_list) > 0 else None,
        "level_probs_A": level_probs_A,
        "gt_scores_A": gt_scores_A,
        "gt_stds_A": gt_stds_A,
        
        "pixel_values_B": pixel_values_B,
        "input_ids_quality_B": input_ids_quality_B,
        "attention_mask_quality_B": attention_mask_quality_B,
        "labels_quality_B": labels_quality_B,
        "media_offset_B": media_offset_B_list if len(media_offset_B_list) > 0 else None,
        "level_probs_B": level_probs_B,
        "gt_scores_B": gt_scores_B,
        "gt_stds_B": gt_stds_B,
    }
    
    # Process scene task if present
    if "input_ids_scene_A" in batch[0]:
        input_ids_scene_A, attention_mask_scene_A, labels_scene_A = pad_and_stack_sequence(
            batch, "input_ids_scene_A", "attention_mask_scene_A"
        )
        input_ids_scene_B, attention_mask_scene_B, labels_scene_B = pad_and_stack_sequence(
            batch, "input_ids_scene_B", "attention_mask_scene_B"
        )
        result.update({
            "input_ids_scene_A": input_ids_scene_A,
            "attention_mask_scene_A": attention_mask_scene_A,
            "labels_scene_A": labels_scene_A,
            "input_ids_scene_B": input_ids_scene_B,
            "attention_mask_scene_B": attention_mask_scene_B,
            "labels_scene_B": labels_scene_B,
        })
    
    # Process distortion task if present
    if "input_ids_distortion_A" in batch[0]:
        input_ids_distortion_A, attention_mask_distortion_A, labels_distortion_A = pad_and_stack_sequence(
            batch, "input_ids_distortion_A", "attention_mask_distortion_A"
        )
        input_ids_distortion_B, attention_mask_distortion_B, labels_distortion_B = pad_and_stack_sequence(
            batch, "input_ids_distortion_B", "attention_mask_distortion_B"
        )
        result.update({
            "input_ids_distortion_A": input_ids_distortion_A,
            "attention_mask_distortion_A": attention_mask_distortion_A,
            "labels_distortion_A": labels_distortion_A,
            "input_ids_distortion_B": input_ids_distortion_B,
            "attention_mask_distortion_B": attention_mask_distortion_B,
            "labels_distortion_B": labels_distortion_B,
        })
    
    return result
