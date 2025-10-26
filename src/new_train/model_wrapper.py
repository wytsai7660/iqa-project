"""
Model wrapper for IQA training with LoRA and custom loss functions.
"""
# Set environment variables BEFORE any imports that use tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


class IQAModelWrapper(nn.Module):
    """
    Wrapper for mPLUG-Owl3 with LoRA and custom IQA loss functions.
    
    Implements:
    1. Cross entropy loss for regular tokens
    2. KL divergence loss for quality level tokens (with soft labels)
    3. Fidelity loss for ranking (pair-wise comparison)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
        level_tokens: list = None,
        level_scores: list = None,
        weight_ce: float = 1.0,
        weight_kl: float = 0.05,
        weight_fidelity: float = 1.0,
        use_fix_std: bool = False,
        detach_pred_std: bool = False,
        binary_fidelity_type: str = "fidelity",
    ):
        super().__init__()
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        
        # Configure LoRA
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Quality level configuration
        if level_tokens is None:
            level_tokens = ["bad", "low", "fair", "good", "awesome"]
        if level_scores is None:
            level_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        self.level_tokens = level_tokens
        self.level_scores = torch.tensor(level_scores, dtype=torch.float32)
        
        # Get token IDs for quality levels
        # Note: Quality words appear after "is " in the sentence, so they have a space prefix
        # We need to encode them with the space to get the correct tokens
        # e.g., " low" -> [3347], " awesome" -> [12456]
        level_words = ["bad", "low", "fair", "good", "awesome"]
        self.level_token_sequences = []
        for word in level_words:
            # Encode WITH space prefix to match how it appears in "The quality of this image is {word}."
            token_ids = self.tokenizer.encode(f" {word}", add_special_tokens=False)
            self.level_token_sequences.append(token_ids)
        
        # Loss weights
        self.weight_ce = weight_ce
        self.weight_kl = weight_kl
        self.weight_fidelity = weight_fidelity
        
        # Fidelity loss settings
        self.use_fix_std = use_fix_std
        self.detach_pred_std = detach_pred_std
        self.binary_fidelity_type = binary_fidelity_type
        
    def find_level_token_position(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Find the position of the level token in the input sequence.
        Returns indices of shape [batch_size] or None if not found.
        
        Quality words are single tokens: bad=[3873], low=[3347], fair=[6624], good=[1661], awesome=[12456]
        """
        batch_size = input_ids.shape[0]
        positions = []
        
        for b in range(batch_size):
            # Look for any level token sequence
            found_pos = None
            for i in range(input_ids.shape[1]):
                # Check each level token sequence
                for token_seq in self.level_token_sequences:
                    seq_len = len(token_seq)
                    # Make sure we have enough space
                    if i + seq_len <= input_ids.shape[1]:
                        # Check if this position matches the sequence
                        match = True
                        for j, token_id in enumerate(token_seq):
                            if input_ids[b, i + j].item() != token_id:
                                match = False
                                break
                        if match:
                            found_pos = i  # Position of first token
                            break
                if found_pos is not None:
                    break
            positions.append(found_pos if found_pos is not None else -1)
        
        positions = torch.tensor(positions, device=input_ids.device)
        if (positions == -1).all():
            return None
        return positions
    
    def compute_kl_loss(
        self,
        logits: torch.Tensor,
        level_probs: torch.Tensor,
        level_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss for quality level tokens.
        
        All quality words are single tokens: bad=[3873], low=[3347], fair=[6624], good=[1661], awesome=[12456]
        We use the logits at level_positions-1 to predict the quality word token.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            level_probs: Target probability distribution [batch_size, 5]
            level_positions: Position of level token in each sequence [batch_size]
        
        Returns:
            KL divergence loss (scalar)
        """
        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]
        
        # Extract logits at level token positions (previous position for next token prediction)
        level_logits = logits[torch.arange(batch_size), level_positions - 1]  # [batch_size, vocab_size]
        
        # Create target distribution using only the FIRST token of each quality word
        target = torch.zeros(batch_size, vocab_size, device=logits.device, dtype=logits.dtype)
        for i, token_seq in enumerate(self.level_token_sequences):
            first_token_id = token_seq[0]  # Use first token for multi-token words
            target[:, first_token_id] = level_probs[:, i]
        
        # Compute KL divergence
        log_pred = F.log_softmax(level_logits, dim=-1)
        loss_kl = F.kl_div(log_pred, target, reduction="batchmean")
        
        return loss_kl
    
    def get_predicted_scores_and_stds(
        self,
        logits: torch.Tensor,
        level_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict quality scores and standard deviations from model logits.
        
        All quality words are single tokens: bad=[3873], low=[3347], fair=[6624], good=[1661], awesome=[12456]
        We use the logits at level_positions-1 to predict the quality word token.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            level_positions: Position of level token in each sequence [batch_size]
        
        Returns:
            scores: Predicted scores [batch_size]
            stds: Predicted standard deviations [batch_size]
        """
        batch_size = logits.shape[0]
        
        # Extract logits at level token positions
        level_logits = logits[torch.arange(batch_size), level_positions - 1]  # [batch_size, vocab_size]
        
        # Get probabilities for level tokens only (closed-set softmax)
        # Use first token of each sequence for multi-token words
        level_logits_subset = torch.stack([
            level_logits[:, token_seq[0]] for token_seq in self.level_token_sequences
        ], dim=1)  # [batch_size, 5]
        
        probs = F.softmax(level_logits_subset, dim=1)  # [batch_size, 5]
        
        # Compute expected score
        scores = torch.matmul(probs, self.level_scores.to(probs.device))  # [batch_size]
        
        # Compute standard deviation
        variances = (self.level_scores.to(probs.device).unsqueeze(0) - scores.unsqueeze(1)) ** 2
        stds = torch.sqrt(torch.sum(probs * variances, dim=1))  # [batch_size]
        
        return scores, stds
    
    def compute_fidelity_loss(
        self,
        pred_scores_A: torch.Tensor,
        pred_stds_A: torch.Tensor,
        gt_scores_A: torch.Tensor,
        gt_stds_A: torch.Tensor,
        pred_scores_B: torch.Tensor,
        pred_stds_B: torch.Tensor,
        gt_scores_B: torch.Tensor,
        gt_stds_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fidelity loss based on Thurstone's model.
        
        P(A > B) = Φ((μ_A - μ_B) / √(σ_A² + σ_B²))
        where Φ is the Gaussian CDF.
        
        Loss = 1 - √(pred_prob * gt_prob) - √((1-pred_prob) * (1-gt_prob))
        """
        eps = 1e-8
        
        # Compute predicted probability
        if self.use_fix_std:
            # Use fixed std = 1.0
            pred_prob = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / (2 ** 0.5)))
        else:
            # Use predicted std
            pred_var = pred_stds_A ** 2 + pred_stds_B ** 2 + eps
            if self.detach_pred_std:
                pred_var = pred_var.detach()
            pred_prob = 0.5 * (1 + torch.erf(
                (pred_scores_A - pred_scores_B) / torch.sqrt(2 * pred_var)
            ))
        
        # Compute ground truth probability
        gt_var = gt_stds_A ** 2 + gt_stds_B ** 2 + eps
        gt_prob = 0.5 * (1 + torch.erf(
            (gt_scores_A - gt_scores_B) / torch.sqrt(2 * gt_var)
        ))
        gt_prob = gt_prob.detach()
        
        # Fidelity loss
        loss = 1 - torch.sqrt(pred_prob * gt_prob + eps) - torch.sqrt((1 - pred_prob) * (1 - gt_prob) + eps)
        return loss.mean()
    
    def compute_binary_fidelity_loss(
        self,
        pred_scores_A: torch.Tensor,
        gt_scores_A: torch.Tensor,
        pred_scores_B: torch.Tensor,
        gt_scores_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary fidelity loss when standard deviations are not available.
        """
        # Predicted probability (assuming fixed std = 1.0)
        pred_prob = 0.5 * (1 + torch.erf((pred_scores_A - pred_scores_B) / (2 ** 0.5)))
        
        # Ground truth (binary: A > B or not)
        gt_binary = (gt_scores_A > gt_scores_B).float()
        
        if self.binary_fidelity_type == "bce":
            # Binary cross entropy
            loss = F.binary_cross_entropy(pred_prob, gt_binary)
        else:  # "fidelity"
            # Fidelity loss with binary ground truth
            eps = 1e-8
            loss_1 = 1 - torch.sqrt(pred_prob[gt_binary == 1] + eps)
            loss_2 = 1 - torch.sqrt(1 - pred_prob[gt_binary == 0] + eps)
            loss = (loss_1.sum() + loss_2.sum()) / pred_prob.shape[0]
        
        return loss
    
    def forward_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        level_probs: Optional[torch.Tensor] = None,
        media_offset: Optional[list] = None,
        gt_scores: Optional[torch.Tensor] = None,  # Not used in single mode, but accepted
        gt_stds: Optional[torch.Tensor] = None,  # Not used in single mode, but accepted
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a single image.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            pixel_values: Image pixel values (optional)
            labels: Target labels for language modeling
            level_probs: Target probability distribution over quality levels
            media_offset: List indicating where images are in the sequence (required when pixel_values is not None)
            gt_scores: Ground truth scores (not used in single mode)
            gt_stds: Ground truth standard deviations (not used in single mode)
        
        Returns:
            Dictionary with 'loss', 'logits', 'loss_ce', 'loss_kl'
        """
        # Prepare inputs
        if pixel_values is not None:
            # When we have images, media_offset must be provided
            if media_offset is None:
                # Default: assume each sample has one image at the beginning
                batch_size = pixel_values.shape[0]
                media_offset = [[0]] * batch_size
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                media_offset=media_offset,
                labels=labels,
                return_dict=True,
            )
        else:
            # Text-only mode, pass empty media_offset list for each sample
            batch_size = input_ids.shape[0]
            media_offset = [[]] * batch_size
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                media_offset=media_offset,
                labels=labels,
                return_dict=True,
            )
        
        loss_ce = outputs.loss if outputs.loss is not None else 0
        logits = outputs.logits
        
        # Compute KL loss for level token
        loss_kl = 0
        if level_probs is not None and labels is not None:
            level_positions = self.find_level_token_position(labels)
            if level_positions is not None:
                valid_mask = level_positions >= 0
                if valid_mask.any():
                    loss_kl = self.compute_kl_loss(
                        logits[valid_mask],
                        level_probs[valid_mask],
                        level_positions[valid_mask],
                    )
        
        # Total loss
        total_loss = self.weight_ce * loss_ce + self.weight_kl * loss_kl
        
        return {
            "loss": total_loss,
            "logits": logits,
            "loss_ce": loss_ce,
            "loss_kl": loss_kl,
        }
    
    def forward_pair(
        self,
        input_ids_A: torch.Tensor,
        attention_mask_A: torch.Tensor,
        pixel_values_A: torch.Tensor,
        labels_A: torch.Tensor,
        level_probs_A: torch.Tensor,
        gt_scores_A: torch.Tensor,
        gt_stds_A: torch.Tensor,
        input_ids_B: torch.Tensor,
        attention_mask_B: torch.Tensor,
        pixel_values_B: torch.Tensor,
        labels_B: torch.Tensor,
        level_probs_B: torch.Tensor,
        gt_scores_B: torch.Tensor,
        gt_stds_B: torch.Tensor,
        media_offset_A: Optional[list] = None,
        media_offset_B: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a pair of images (for fidelity loss).
        
        Returns:
            Dictionary with 'loss' and individual loss components
        """
        # Forward for image A
        outputs_A = self.forward_single(
            input_ids_A, attention_mask_A, pixel_values_A, labels_A, level_probs_A, media_offset_A
        )
        
        # Forward for image B
        outputs_B = self.forward_single(
            input_ids_B, attention_mask_B, pixel_values_B, labels_B, level_probs_B, media_offset_B
        )
        
        # Get predicted scores
        # Note: We search in input_ids, not labels, because labels have -100 for padding
        level_positions_A = self.find_level_token_position(input_ids_A)
        level_positions_B = self.find_level_token_position(input_ids_B)
        
        pred_scores_A, pred_stds_A = self.get_predicted_scores_and_stds(
            outputs_A["logits"], level_positions_A
        )
        pred_scores_B, pred_stds_B = self.get_predicted_scores_and_stds(
            outputs_B["logits"], level_positions_B
        )
        
        # Compute fidelity loss
        if gt_stds_A is not None and gt_stds_B is not None:
            loss_fidelity = self.compute_fidelity_loss(
                pred_scores_A, pred_stds_A, gt_scores_A, gt_stds_A,
                pred_scores_B, pred_stds_B, gt_scores_B, gt_stds_B,
            )
        else:
            loss_fidelity = self.compute_binary_fidelity_loss(
                pred_scores_A, gt_scores_A, pred_scores_B, gt_scores_B
            )
        
        # Total loss
        total_loss = (
            outputs_A["loss"] + outputs_B["loss"] +
            self.weight_fidelity * loss_fidelity
        )
        
        return {
            "loss": total_loss,
            "loss_ce": outputs_A["loss_ce"] + outputs_B["loss_ce"],
            "loss_kl": outputs_A["loss_kl"] + outputs_B["loss_kl"],
            "loss_fidelity": loss_fidelity,
            # Include logits for validation metric computation
            "logits": outputs_A["logits"],  # Use logits from image A for metrics
        }
    
    def forward(self, **kwargs):
        """
        Forward pass dispatcher.
        """
        # Remove trainer-specific arguments that shouldn't be passed to model
        kwargs.pop('num_items_in_batch', None)
        kwargs.pop('return_outputs', None)
        
        # Check if this is a pair dataset (has both A and B images)
        if "input_ids_A" in kwargs and "input_ids_B" in kwargs:
            return self.forward_pair(**kwargs)
        else:
            return self.forward_single(**kwargs)
