"""
Custom Trainer with explicit eval loss logging and memory-efficient evaluation
"""
# Set environment variables BEFORE any imports that use tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import Trainer
from transformers.trainer_callback import TrainerCallback, PrinterCallback
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats


class SimplifiedProgressCallback(TrainerCallback):
    """
    Simplified callback that only prints epoch and training loss.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Only print training loss and epoch, suppress other logs"""
        if state.is_local_process_zero and logs is not None:
            # Only show train loss and epoch during training
            if 'loss' in logs and 'epoch' in logs:
                print(f"Epoch {logs['epoch']:.2f} | Loss: {logs['loss']:.4f}")
            # Show evaluation metrics when available
            elif 'eval_loss' in logs:
                print(f"\n{'='*70}")
                eval_mode = "Generated Context" if hasattr(self, 'trainer') and getattr(self.trainer, 'use_generated_context', False) else "GT Context (Teacher Forcing)"
                print(f"📊 Validation Results at Epoch {logs.get('epoch', 0):.2f} [{eval_mode}]")
                print(f"{'='*70}")
                
                # Total Loss
                print(f"  Total Loss: {logs['eval_loss']:.6f}")
                
                # Loss components (if available)
                if 'eval_loss_ce' in logs:
                    print(f"  - CE Loss:      {logs['eval_loss_ce']:.6f}")
                if 'eval_loss_kl' in logs:
                    print(f"  - KL Loss:      {logs['eval_loss_kl']:.6f}")
                if 'eval_loss_fidelity' in logs:
                    print(f"  - Fidelity Loss: {logs['eval_loss_fidelity']:.6f}")
                
                # Regression metrics (MAE, RMSE)
                if 'eval_mae' in logs:
                    print(f"\n  MAE:        {logs['eval_mae']:.4f}")
                if 'eval_rmse' in logs:
                    print(f"  RMSE:       {logs['eval_rmse']:.4f}")
                
                # Correlation metrics (PLCC, SRCC) - highlighted
                has_correlation = False
                if 'eval_plcc' in logs:
                    has_correlation = True
                if 'eval_srcc' in logs:
                    has_correlation = True
                
                if has_correlation:
                    print(f"  {'-'*66}")
                    print(f"  Correlation Metrics:")
                    if 'eval_plcc' in logs:
                        plcc = logs['eval_plcc']
                        plcc_bar = '█' * int(plcc * 20) + '░' * (20 - int(plcc * 20))
                        print(f"  PLCC:       {plcc:.4f}  [{plcc_bar}]")
                    if 'eval_srcc' in logs:
                        srcc = logs['eval_srcc']
                        srcc_bar = '█' * int(srcc * 20) + '░' * (20 - int(srcc * 20))
                        print(f"  SRCC:       {srcc:.4f}  [{srcc_bar}]")
                
                print(f"{'='*70}\n")


class IQATrainer(Trainer):
    """
    Custom Trainer that ensures eval_loss is properly logged and handles
    evaluation in a memory-efficient way by computing metrics on-the-fly.

    Also ensures that only LoRA adapters are saved during checkpointing.
    Supports weighted sampling to address MOS distribution imbalance.
    Generates scatter plots during evaluation to visualize predictions vs ground truth.

    IMPORTANT: During evaluation, uses TRUE sequential generation (no teacher forcing)
    to match test-time behavior.
    """

    def __init__(self, *args, use_weighted_sampling: bool = False, use_generated_context: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # Store references for metric computation
        self.eval_predictions = []
        self.eval_labels = []

        # Store loss components for detailed logging
        self.eval_loss_ce_list = []
        self.eval_loss_kl_list = []
        self.eval_loss_fidelity_list = []

        # Weighted sampling configuration
        self.use_weighted_sampling = use_weighted_sampling

        # Use generated context during evaluation (no teacher forcing)
        self.use_generated_context = use_generated_context

        # Remove default PrinterCallback and add our simplified one
        self.remove_callback(PrinterCallback)
        self.add_callback(SimplifiedProgressCallback)

    def get_train_dataloader(self):
        """
        Override to support weighted sampling based on MOS distribution.

        When use_weighted_sampling is True, uses WeightedRandomSampler to oversample
        images from underrepresented MOS bins (e.g., low-quality images).
        """
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from transformers.trainer_utils import seed_worker

        if not self.use_weighted_sampling:
            return super().get_train_dataloader()

        # Check if dataset supports weighted sampling
        if not hasattr(self.train_dataset, 'get_sample_weights'):
            print("⚠️  Warning: Dataset does not support get_sample_weights(), using default sampling")
            return super().get_train_dataloader()

        # Get sample weights from dataset
        sample_weights = self.train_dataset.get_sample_weights()
        print(f"📊 Using weighted sampling with {len(sample_weights)} samples")
        print(f"   Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

        # Create weighted sampler
        weighted_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(self.train_dataset),
            replacement=True,
        )

        # Create DataLoader with explicit arguments from TrainingArguments
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=weighted_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            worker_init_fn=seed_worker,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )

    def _save(self, output_dir=None, state_dict=None):
        """
        Override _save to only save LoRA adapters, not the full model.

        This ensures that checkpoint saves during training (every save_steps)
        only save the small adapter weights (~100MB) instead of the full model (~15GB).
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save the PEFT model (LoRA adapters only)
        # The model is IQAModelWrapper.model, which is a PeftModel
        if hasattr(self.model, 'model'):
            # IQAModelWrapper wraps the PEFT model in self.model
            self.model.model.save_pretrained(output_dir)
        else:
            # Fallback to default behavior if model structure is different
            super()._save(output_dir, state_dict)

        # Save tokenizer for convenience
        if hasattr(self.model, 'tokenizer'):
            self.model.tokenizer.save_pretrained(output_dir)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute loss for quality assessment task.
        
        For multi-task training, use train_scene.py and train_distortion.py separately
        to avoid OOM issues. This trainer focuses on the quality task only.
        """
        # Only process quality task
        quality_inputs = {**inputs, "active_task": "quality"}
        quality_outputs = model(**quality_inputs)
        
        quality_loss = quality_outputs.get("loss") if isinstance(quality_outputs, dict) else quality_outputs[0]
        
        # Prepare outputs
        if isinstance(quality_outputs, dict):
            outputs = {
                "loss": quality_loss,
                "loss_ce": quality_outputs.get("loss_ce", 0.0),
                "loss_kl": quality_outputs.get("loss_kl", 0.0),
                "loss_fidelity": quality_outputs.get("loss_fidelity", 0.0),
                "logits": quality_outputs.get("logits"),
            }
        else:
            outputs = quality_outputs
        
        # Return loss and outputs if requested
        if return_outputs:
            return quality_loss, outputs
        return quality_loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Override prediction_step to use TRUE sequential generation during evaluation.

        If use_generated_context=True:
            - Generates scene answer
            - Generates distortion answer (using generated scene)
            - Predicts quality (using generated scene + distortion)

        If use_generated_context=False (old behavior):
            - Uses ground truth context (teacher forcing)
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                # Always compute loss for logging
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                # Store loss components for detailed logging
                if isinstance(outputs, dict):
                    loss_ce = outputs.get("loss_ce", torch.tensor(0.0))
                    loss_kl = outputs.get("loss_kl", torch.tensor(0.0))
                    loss_fidelity = outputs.get("loss_fidelity", torch.tensor(0.0))

                    # Convert to float and store
                    if isinstance(loss_ce, torch.Tensor):
                        self.eval_loss_ce_list.append(loss_ce.item())
                    if isinstance(loss_kl, torch.Tensor):
                        self.eval_loss_kl_list.append(loss_kl.item())
                    if isinstance(loss_fidelity, torch.Tensor):
                        self.eval_loss_fidelity_list.append(loss_fidelity.item())

                # Compute predictions using sequential generation (if enabled)
                if self.use_generated_context:
                    # Use TRUE sequential generation (like eval_sequential_model.py)
                    pred_scores, gt_scores = self._evaluate_with_generated_context(model, inputs)

                    if len(pred_scores) > 0:
                        self.eval_predictions.extend(pred_scores)
                        self.eval_labels.extend(gt_scores)
                else:
                    # Old behavior: use teacher forcing (GT context)
                    # Extract logits from the quality forward pass
                    if isinstance(outputs, dict):
                        logits = outputs.get("logits")
                    else:
                        logits = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None

                    if logits is not None:
                        # For pair dataset, extract quality scores from image A
                        input_ids_quality_A = inputs.get("input_ids_quality_A")
                        labels_quality_A = inputs.get("labels_quality_A")

                        if input_ids_quality_A is not None and labels_quality_A is not None:
                            # Find level token positions in input_ids_quality_A
                            level_positions = self.model.find_level_token_position(input_ids_quality_A)
                            if level_positions is not None and (level_positions >= 0).any():
                                # Compute predicted scores for this batch
                                from src.new_train.metrics import compute_quality_score_from_logits
                                valid_mask = level_positions >= 0

                                if valid_mask.any():
                                    pred_scores = compute_quality_score_from_logits(
                                        logits[valid_mask],
                                        level_positions[valid_mask],
                                        self.model.level_token_sequences,
                                    )

                                    # Extract ground truth scores from gt_scores_A (if available)
                                    gt_scores_A = inputs.get("gt_scores_A")
                                    if gt_scores_A is not None:
                                        # Use the provided ground truth scores
                                        gt_scores = gt_scores_A[valid_mask]
                                        # Store for later aggregation
                                        self.eval_predictions.extend(pred_scores.cpu().numpy().tolist())
                                        self.eval_labels.extend(gt_scores.cpu().numpy().tolist())
            else:
                loss = None

        # Always return (loss, None, None) to avoid storing logits
        return (loss, None, None)

    def _evaluate_with_generated_context(self, model, inputs):
        """
        Evaluate using TRUE sequential generation (no teacher forcing).

        Process:
        1. Generate scene answer
        2. Generate distortion answer (using generated scene as context)
        3. Predict quality (using generated scene + distortion as context)

        Returns:
            pred_scores: List of predicted quality scores
            gt_scores: List of ground truth quality scores
        """
        import torch.nn.functional as F

        pred_scores = []
        gt_scores = []

        # Get batch size
        batch_size = inputs["pixel_values_A"].shape[0]

        # Get tokenizer from model
        tokenizer = self.model.tokenizer

        # Get quality level tokens
        level_names = ["bad", "low", "fair", "good", "awesome"]
        level_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        level_token_ids = [
            tokenizer.encode(f" {level_name}", add_special_tokens=False)[0]
            for level_name in level_names
        ]

        # Process each sample in batch
        for i in range(batch_size):
            try:
                # Extract data for this sample
                pixel_values = inputs["pixel_values_A"][i:i+1]  # Keep batch dim
                media_offset = [inputs["media_offset_A"][i]]
                gt_score = inputs["gt_scores_A"][i].item()

                # === Step 1: Generate scene answer ===
                scene_messages = [
                    {"role": "user", "content": "<|image|>\n"},
                    {"role": "user", "content": "What is the scene type of this image?"},
                ]

                scene_text = tokenizer.apply_chat_template(
                    scene_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                scene_encoding = tokenizer(scene_text, return_tensors="pt", add_special_tokens=False)
                scene_input_ids = scene_encoding['input_ids'].to(pixel_values.device)
                scene_attention_mask = scene_encoding['attention_mask'].to(pixel_values.device)

                # Generate scene answer
                scene_out = model.model.generate(
                    input_ids=scene_input_ids,
                    pixel_values=pixel_values,
                    media_offset=media_offset,
                    attention_mask=scene_attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                )

                scene_response = tokenizer.decode(scene_out[0], skip_special_tokens=True).strip()

                # === Step 2: Generate distortion answer ===
                distortion_messages = [
                    {"role": "user", "content": "<|image|>\n"},
                    {"role": "user", "content": "What is the scene type of this image?"},
                    {"role": "assistant", "content": scene_response},
                    {"role": "user", "content": "What is the distortion type of this image?"},
                ]

                distortion_text = tokenizer.apply_chat_template(
                    distortion_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                distortion_encoding = tokenizer(distortion_text, return_tensors="pt", add_special_tokens=False)
                distortion_input_ids = distortion_encoding['input_ids'].to(pixel_values.device)
                distortion_attention_mask = distortion_encoding['attention_mask'].to(pixel_values.device)

                # Generate distortion answer
                distortion_out = model.model.generate(
                    input_ids=distortion_input_ids,
                    pixel_values=pixel_values,
                    media_offset=media_offset,
                    attention_mask=distortion_attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                )

                distortion_response = tokenizer.decode(distortion_out[0], skip_special_tokens=True).strip()

                # === Step 3: Predict quality ===
                quality_messages = [
                    {"role": "user", "content": "<|image|>\n"},
                    {"role": "user", "content": "What is the scene type of this image?"},
                    {"role": "assistant", "content": scene_response},
                    {"role": "user", "content": "What is the distortion type of this image?"},
                    {"role": "assistant", "content": distortion_response},
                    {"role": "user", "content": "What do you think about the quality of this image?"},
                    {"role": "assistant", "content": "The quality of this image is "},
                ]

                quality_text = tokenizer.apply_chat_template(
                    quality_messages,
                    tokenize=False,
                    continue_final_message=True,
                )

                quality_encoding = tokenizer(quality_text, return_tensors="pt", add_special_tokens=False)
                quality_input_ids = quality_encoding['input_ids'].to(pixel_values.device)
                quality_attention_mask = quality_encoding['attention_mask'].to(pixel_values.device)

                # Forward pass to get logits
                quality_outputs = model.model(
                    input_ids=quality_input_ids,
                    pixel_values=pixel_values,
                    media_offset=media_offset,
                    attention_mask=quality_attention_mask,
                )

                # Extract logits at last position
                last_logits = quality_outputs.logits[0, -1, :]

                # Get logits for quality tokens
                level_logits = last_logits[level_token_ids]

                # Compute softmax probabilities
                level_probs = F.softmax(level_logits, dim=0)

                # Compute expected score
                expected_score = sum(prob.item() * score for prob, score in zip(level_probs, level_scores))

                pred_scores.append(expected_score)
                gt_scores.append(gt_score)

            except Exception as e:
                # Skip this sample if error occurs
                print(f"[WARNING] Error evaluating sample {i} in batch: {e}")
                continue

        return pred_scores, gt_scores
    
    def evaluation_loop(self, *args, **kwargs):
        """
        Override evaluation loop to compute aggregated metrics at the end.
        """
        # Reset accumulators
        self.eval_predictions = []
        self.eval_labels = []
        self.eval_loss_ce_list = []
        self.eval_loss_kl_list = []
        self.eval_loss_fidelity_list = []
        
        # Run parent's evaluation loop
        output = super().evaluation_loop(*args, **kwargs)
        
        # Debug: print collected predictions
        print(f"\n[DEBUG] Collected {len(self.eval_predictions)} predictions")
        print(f"[DEBUG] Collected {len(self.eval_loss_ce_list)} loss_ce values")
        print(f"[DEBUG] Collected {len(self.eval_loss_kl_list)} loss_kl values")
        print(f"[DEBUG] Collected {len(self.eval_loss_fidelity_list)} loss_fidelity values")
        
        # Add detailed loss components to metrics
        if output.metrics is not None:
            if len(self.eval_loss_ce_list) > 0:
                avg_loss_ce = np.mean(self.eval_loss_ce_list)
                output.metrics['eval_loss_ce'] = avg_loss_ce
                print(f"[DEBUG] Average loss_ce: {avg_loss_ce:.6f}")
            
            if len(self.eval_loss_kl_list) > 0:
                avg_loss_kl = np.mean(self.eval_loss_kl_list)
                output.metrics['eval_loss_kl'] = avg_loss_kl
                print(f"[DEBUG] Average loss_kl: {avg_loss_kl:.6f}")
            
            if len(self.eval_loss_fidelity_list) > 0:
                avg_loss_fidelity = np.mean(self.eval_loss_fidelity_list)
                output.metrics['eval_loss_fidelity'] = avg_loss_fidelity
                print(f"[DEBUG] Average loss_fidelity: {avg_loss_fidelity:.6f}")
        
        # Compute aggregated metrics if we have predictions
        if len(self.eval_predictions) > 0:
            from src.new_train.metrics import compute_iqa_metrics

            pred_scores = np.array(self.eval_predictions)
            gt_scores = np.array(self.eval_labels)

            print(f"[DEBUG] Computing metrics: pred shape={pred_scores.shape}, gt shape={gt_scores.shape}")

            # Log prediction distribution statistics
            pred_min = np.min(pred_scores)
            pred_max = np.max(pred_scores)
            pred_mean = np.mean(pred_scores)
            pred_std = np.std(pred_scores)
            pred_range = pred_max - pred_min

            print(f"\n[PREDICTION STATS]")
            print(f"  Min:   {pred_min:.4f}")
            print(f"  Max:   {pred_max:.4f}")
            print(f"  Mean:  {pred_mean:.4f}")
            print(f"  Std:   {pred_std:.4f}")
            print(f"  Range: {pred_range:.4f}")

            # Also log ground truth distribution for comparison
            gt_min = np.min(gt_scores)
            gt_max = np.max(gt_scores)
            gt_mean = np.mean(gt_scores)
            gt_std = np.std(gt_scores)
            print(f"\n[GROUND TRUTH STATS]")
            print(f"  Min:   {gt_min:.4f}")
            print(f"  Max:   {gt_max:.4f}")
            print(f"  Mean:  {gt_mean:.4f}")
            print(f"  Std:   {gt_std:.4f}")

            # Warn if predictions are collapsing to a narrow range
            if pred_range < 0.5:
                print(f"\n⚠️  WARNING: Predictions collapsed to narrow range ({pred_range:.4f})!")
                print(f"   This indicates the model is predicting similar scores for all images.")
            elif pred_std < 0.2:
                print(f"\n⚠️  WARNING: Low prediction variance (std={pred_std:.4f})!")
                print(f"   The model may not be distinguishing quality differences well.")

            metrics = compute_iqa_metrics(pred_scores, gt_scores)
            
            print(f"[DEBUG] Computed metrics: {metrics}")
            
            # Store metrics for logging callback
            self.eval_iqa_metrics = {f'eval_{k}': v for k, v in metrics.items()}
            
            # Add metrics to output.metrics dict (which is mutable)
            if output.metrics is not None:
                for k, v in self.eval_iqa_metrics.items():
                    output.metrics[k] = v
                print(f"[DEBUG] Added metrics to output.metrics: {list(self.eval_iqa_metrics.keys())}")
        
        # Generate scatter plot before clearing predictions
        if len(self.eval_predictions) > 0:
            self._generate_eval_scatter_plot(
                pred_scores=np.array(self.eval_predictions),
                gt_scores=np.array(self.eval_labels),
                step=self.state.global_step,
                metrics=output.metrics,
            )

        # Clear accumulators
        self.eval_predictions = []
        self.eval_labels = []
        self.eval_loss_ce_list = []
        self.eval_loss_kl_list = []
        self.eval_loss_fidelity_list = []

        return output

    def _generate_eval_scatter_plot(
        self,
        pred_scores: np.ndarray,
        gt_scores: np.ndarray,
        step: int,
        metrics: dict | None = None,
    ):
        """
        Generate and save a scatter plot of predictions vs ground truth.

        Args:
            pred_scores: Predicted quality scores [N]
            gt_scores: Ground truth quality scores [N]
            step: Current training step
            metrics: Optional dict with PLCC/SRCC metrics
        """
        try:
            # Create eval_plots directory under output_dir
            plot_dir = Path(self.args.output_dir) / "eval_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            plcc = metrics.get("eval_plcc", 0.0) if metrics else 0.0
            srcc = metrics.get("eval_srcc", 0.0) if metrics else 0.0

            fig, ax = plt.subplots(figsize=(8, 8))

            # Scatter plot
            ax.scatter(gt_scores, pred_scores, alpha=0.5, s=20, edgecolors='none')

            # Perfect prediction line
            min_val = min(gt_scores.min(), pred_scores.min())
            max_val = max(gt_scores.max(), pred_scores.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            # Labels and title
            ax.set_xlabel('Ground Truth Quality Score', fontsize=12)
            ax.set_ylabel('Predicted Quality Score', fontsize=12)
            ax.set_title(f'Step {step}: PLCC={plcc:.4f}, SRCC={srcc:.4f}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set consistent axis limits
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)

            # Save figure
            plot_path = plot_dir / f"step_{step:06d}.png"
            fig.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"📈 Saved evaluation plot to: {plot_path}")

        except Exception as e:
            print(f"⚠️  Warning: Failed to generate evaluation plot: {e}")
