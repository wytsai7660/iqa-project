"""
Training configuration for IQA with LoRA fine-tuning.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_name_or_path: str = "src/owl3"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quality level tokens
    level_tokens: List[str] = field(default_factory=lambda: [
        "bad", "low", "fair", "good", "awesome"
    ])
    level_scores: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Training settings
    tune_visual_abstractor: bool = True
    freeze_vision_model: bool = False


@dataclass
class DataConfig:
    """Dataset-related configuration."""
    dataset_paths: List[str] = field(default_factory=list)
    split: str = "training"
    max_length: int = 512
    image_size: int = 378
    
    # For pair dataset
    use_pair_dataset: bool = False


@dataclass
class LossConfig:
    """Loss function configuration."""
    # Loss weights
    weight_ce: float = 1.0  # Cross entropy for normal tokens
    weight_kl: float = 0.05  # KL divergence for level token
    weight_fidelity: float = 1.0  # Fidelity loss for ranking
    
    # Fidelity loss settings
    use_fidelity_loss: bool = False
    use_fix_std: bool = False  # Use fixed std=1.0 or predicted std
    detach_pred_std: bool = False  # Detach gradient for predicted std
    
    # Binary fidelity loss (when no std available)
    binary_fidelity_type: str = "fidelity"  # "fidelity" or "bce"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = "outputs/iqa_lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimization
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Distributed training
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    
    # Other
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
