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
                print(f"📊 Validation Results at Epoch {logs.get('epoch', 0):.2f}")
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
                
                print(f"======================================================================\n")


class IQATrainer(Trainer):
    """
    Custom Trainer that ensures eval_loss is properly logged and handles
    evaluation in a memory-efficient way by computing metrics on-the-fly.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store references for metric computation
        self.eval_predictions = []
        self.eval_labels = []
        
        # Store loss components for detailed logging
        self.eval_loss_ce_list = []
        self.eval_loss_kl_list = []
        self.eval_loss_fidelity_list = []
        
        # Remove default PrinterCallback and add our simplified one
        self.remove_callback(PrinterCallback)
        self.add_callback(SimplifiedProgressCallback)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
        Override prediction_step to compute metrics on-the-fly without storing all logits.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if has_labels:
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
                
                # Extract logits for this batch only
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                else:
                    logits = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
                
                # Always compute metrics during evaluation (ignore prediction_loss_only flag)
                # We need to extract quality scores to compute PLCC/SRCC
                if logits is not None:
                    # For pair dataset, extract quality scores from image A
                    # Use input_ids_quality_A to find level positions (not labels, which has -100 padding)
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
                            # Debug: why no level positions found?
                            if level_positions is None:
                                print("[DEBUG] level_positions is None")
                            elif not (level_positions >= 0).any():
                                print(f"[DEBUG] No valid level positions found: {level_positions}")
                    else:
                        print("[DEBUG] No input_ids_quality_A or labels_quality_A in inputs")
                else:
                    print("[DEBUG] No logits in outputs")
            else:
                loss = None
        
        # Always return (loss, None, None) to avoid storing logits
        return (loss, None, None)
    
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
            
            metrics = compute_iqa_metrics(pred_scores, gt_scores)
            
            print(f"[DEBUG] Computed metrics: {metrics}")
            
            # Store metrics for logging callback
            self.eval_iqa_metrics = {f'eval_{k}': v for k, v in metrics.items()}
            
            # Add metrics to output.metrics dict (which is mutable)
            if output.metrics is not None:
                for k, v in self.eval_iqa_metrics.items():
                    output.metrics[k] = v
                print(f"[DEBUG] Added metrics to output.metrics: {list(self.eval_iqa_metrics.keys())}")
        
        # Clear accumulators
        self.eval_predictions = []
        self.eval_labels = []
        self.eval_loss_ce_list = []
        self.eval_loss_kl_list = []
        self.eval_loss_fidelity_list = []
        
        return output
