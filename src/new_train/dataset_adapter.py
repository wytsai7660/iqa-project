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
    """
    
    def __init__(
        self,
        dataset_paths: List[str],
        processor: mPLUGOwl3Processor,
        tokenizer,
        split: str = "training",
        max_length: int = 512,
    ):
        self.pair_dataset = PairDataset(
            dataset_paths=[Path(p) for p in dataset_paths],
            processor=processor,
            tokenizer=tokenizer,
            split=split,
        )
        self.max_length = max_length
    
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
        gt_std_A = float(np.sqrt(np.sum(level_probs_A * (level_scores - gt_score_A) ** 2)))
        
        level_probs_B = image_B["level_probabilities"]
        gt_score_B = float(np.sum(level_probs_B * level_scores))
        gt_std_B = float(np.sqrt(np.sum(level_probs_B * (level_scores - gt_score_B) ** 2)))
        
        quality_messages_A = image_A["quality_message"]
        quality_messages_B = image_B["quality_message"]
        
        # Extract input_ids, pixel_values, and media_offset
        input_ids_A = quality_messages_A["input_ids"][0] if "input_ids" in quality_messages_A else None
        input_ids_B = quality_messages_B["input_ids"][0] if "input_ids" in quality_messages_B else None
        pixel_values_A = quality_messages_A["pixel_values"][0] if "pixel_values" in quality_messages_A else None
        pixel_values_B = quality_messages_B["pixel_values"][0] if "pixel_values" in quality_messages_B else None
        media_offset_A = quality_messages_A["media_offset"][0] if "media_offset" in quality_messages_A else None
        media_offset_B = quality_messages_B["media_offset"][0] if "media_offset" in quality_messages_B else None
        
        # Create attention masks (all 1s for valid tokens)
        attention_mask_A = torch.ones_like(input_ids_A) if input_ids_A is not None else None
        attention_mask_B = torch.ones_like(input_ids_B) if input_ids_B is not None else None
        
        return {
            # Image A
            "pixel_values_A": pixel_values_A,
            "input_ids_A": input_ids_A,
            "attention_mask_A": attention_mask_A,
            "media_offset_A": media_offset_A,
            "level_probs_A": torch.from_numpy(level_probs_A).float(),
            "gt_scores_A": torch.tensor(gt_score_A).float(),
            "gt_stds_A": torch.tensor(gt_std_A).float(),
            
            # Image B
            "pixel_values_B": pixel_values_B,
            "input_ids_B": input_ids_B,
            "attention_mask_B": attention_mask_B,
            "media_offset_B": media_offset_B,
            "level_probs_B": torch.from_numpy(level_probs_B).float(),
            "gt_scores_B": torch.tensor(gt_score_B).float(),
            "gt_stds_B": torch.tensor(gt_std_B).float(),
        }


def collate_fn_pair(batch):
    """Collate function for pair dataset."""
    # Find max length for padding (considering both A and B)
    max_len_A = max(item["input_ids_A"].shape[0] for item in batch if item["input_ids_A"] is not None)
    max_len_B = max(item["input_ids_B"].shape[0] for item in batch if item["input_ids_B"] is not None)
    
    # Pad and stack tensors for image A
    pixel_values_A_list = []
    input_ids_A_list = []
    attention_mask_A_list = []
    labels_A_list = []
    media_offset_A_list = []
    
    for item in batch:
        if item["input_ids_A"] is not None:
            seq_len = item["input_ids_A"].shape[0]
            pad_len = max_len_A - seq_len
            
            # Pad input_ids with tokenizer.pad_token_id (should be 0)
            padded_input_ids = torch.cat([item["input_ids_A"], torch.zeros(pad_len, dtype=item["input_ids_A"].dtype)])
            input_ids_A_list.append(padded_input_ids)
            
            # Pad attention_mask with 0
            padded_attention_mask = torch.cat([item["attention_mask_A"], torch.zeros(pad_len, dtype=item["attention_mask_A"].dtype)])
            attention_mask_A_list.append(padded_attention_mask)
            
            # Pad labels with -100 (ignore index)
            padded_labels = torch.cat([item["input_ids_A"], torch.full((pad_len,), -100, dtype=item["input_ids_A"].dtype)])
            labels_A_list.append(padded_labels)
            
            if item["pixel_values_A"] is not None:
                pixel_values_A_list.append(item["pixel_values_A"])
            
            if item["media_offset_A"] is not None:
                media_offset_A_list.append(item["media_offset_A"])
    
    # Pad and stack tensors for image B
    pixel_values_B_list = []
    input_ids_B_list = []
    attention_mask_B_list = []
    labels_B_list = []
    media_offset_B_list = []
    
    for item in batch:
        if item["input_ids_B"] is not None:
            seq_len = item["input_ids_B"].shape[0]
            pad_len = max_len_B - seq_len
            
            # Pad input_ids with tokenizer.pad_token_id (should be 0)
            padded_input_ids = torch.cat([item["input_ids_B"], torch.zeros(pad_len, dtype=item["input_ids_B"].dtype)])
            input_ids_B_list.append(padded_input_ids)
            
            # Pad attention_mask with 0
            padded_attention_mask = torch.cat([item["attention_mask_B"], torch.zeros(pad_len, dtype=item["attention_mask_B"].dtype)])
            attention_mask_B_list.append(padded_attention_mask)
            
            # Pad labels with -100 (ignore index)
            padded_labels = torch.cat([item["input_ids_B"], torch.full((pad_len,), -100, dtype=item["input_ids_B"].dtype)])
            labels_B_list.append(padded_labels)
            
            if item["pixel_values_B"] is not None:
                pixel_values_B_list.append(item["pixel_values_B"])
            
            if item["media_offset_B"] is not None:
                media_offset_B_list.append(item["media_offset_B"])
    
    # Stack everything
    pixel_values_A = torch.stack(pixel_values_A_list) if len(pixel_values_A_list) > 0 else None
    input_ids_A = torch.stack(input_ids_A_list) if len(input_ids_A_list) > 0 else None
    attention_mask_A = torch.stack(attention_mask_A_list) if len(attention_mask_A_list) > 0 else None
    labels_A = torch.stack(labels_A_list) if len(labels_A_list) > 0 else None
    
    pixel_values_B = torch.stack(pixel_values_B_list) if len(pixel_values_B_list) > 0 else None
    input_ids_B = torch.stack(input_ids_B_list) if len(input_ids_B_list) > 0 else None
    attention_mask_B = torch.stack(attention_mask_B_list) if len(attention_mask_B_list) > 0 else None
    labels_B = torch.stack(labels_B_list) if len(labels_B_list) > 0 else None
    
    level_probs_A = torch.stack([item["level_probs_A"] for item in batch])
    gt_scores_A = torch.stack([item["gt_scores_A"] for item in batch])
    gt_stds_A = torch.stack([item["gt_stds_A"] for item in batch])
    
    level_probs_B = torch.stack([item["level_probs_B"] for item in batch])
    gt_scores_B = torch.stack([item["gt_scores_B"] for item in batch])
    gt_stds_B = torch.stack([item["gt_stds_B"] for item in batch])
    
    return {
        "pixel_values_A": pixel_values_A,
        "input_ids_A": input_ids_A,
        "attention_mask_A": attention_mask_A,
        "media_offset_A": media_offset_A_list if len(media_offset_A_list) > 0 else None,
        "labels_A": labels_A,
        "level_probs_A": level_probs_A,
        "gt_scores_A": gt_scores_A,
        "gt_stds_A": gt_stds_A,
        
        "pixel_values_B": pixel_values_B,
        "input_ids_B": input_ids_B,
        "attention_mask_B": attention_mask_B,
        "media_offset_B": media_offset_B_list if len(media_offset_B_list) > 0 else None,
        "labels_B": labels_B,
        "level_probs_B": level_probs_B,
        "gt_scores_B": gt_scores_B,
        "gt_stds_B": gt_stds_B,
    }
