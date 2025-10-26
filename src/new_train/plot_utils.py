"""
Utility functions for plotting training metrics
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from typing import List, Tuple, Optional


def read_tensorboard_scalars(log_dir: Path, tag: str) -> List[Tuple[int, float]]:
    """
    Read scalar values from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        tag: Scalar tag to read (e.g., 'train/loss', 'eval/loss')
    
    Returns:
        List of (step, value) tuples
    """
    # Find the latest run directory
    runs_dir = log_dir / "runs"
    if not runs_dir.exists():
        return []
    
    run_dirs = sorted(runs_dir.glob("*/"))
    if not run_dirs:
        return []
    
    latest_run = run_dirs[-1]
    
    # Find event file
    event_files = list(latest_run.glob("events.out.tfevents.*"))
    if not event_files:
        return []
    
    event_file = event_files[0]
    
    # Load events
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    # Get scalars
    tags = ea.Tags()
    if tag not in tags.get('scalars', []):
        return []
    
    events = ea.Scalars(tag)
    return [(event.step, event.value) for event in events]


def plot_training_curves(
    output_dir: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
):
    """
    Plot training and validation loss curves from TensorBoard logs.
    
    Args:
        output_dir: Directory containing the TensorBoard logs
        save_path: Path to save the plot (default: output_dir/training_curves.png)
        figsize: Figure size (width, height)
    """
    output_path = Path(output_dir)
    
    # Read metrics
    train_loss = read_tensorboard_scalars(output_path, 'train/loss')
    eval_loss = read_tensorboard_scalars(output_path, 'eval/loss')
    learning_rate = read_tensorboard_scalars(output_path, 'train/learning_rate')
    
    # Create figure with subplots
    if eval_loss:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = list(axes) + [None]  # Add None for eval loss subplot
    
    # Plot 1: Training Loss
    if train_loss:
        steps, losses = zip(*train_loss)
        axes[0].plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Step', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Add statistics
        final_loss = losses[-1]
        min_loss = min(losses)
        axes[0].axhline(y=min_loss, color='r', linestyle='--', alpha=0.5, label=f'Min: {min_loss:.4f}')
        axes[0].text(0.02, 0.98, f'Final: {final_loss:.4f}\nMin: {min_loss:.4f}',
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Validation Loss (if available)
    if eval_loss and axes[1] is not None:
        steps, losses = zip(*eval_loss)
        axes[1].plot(steps, losses, 'g-', linewidth=2, marker='o', markersize=6, label='Validation Loss')
        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        # Add statistics
        final_loss = losses[-1]
        min_loss = min(losses)
        axes[1].axhline(y=min_loss, color='r', linestyle='--', alpha=0.5)
        axes[1].text(0.02, 0.98, f'Final: {final_loss:.4f}\nMin: {min_loss:.4f}',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    elif axes[1] is not None:
        axes[1].text(0.5, 0.5, 'No Validation Loss Data', 
                    transform=axes[1].transAxes, fontsize=14,
                    ha='center', va='center')
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    
    # Plot 3: Combined view (if eval loss exists) or Learning Rate
    if eval_loss and len(axes) > 2 and axes[2] is not None:
        # Combined plot
        if train_loss:
            train_steps, train_losses = zip(*train_loss)
            axes[2].plot(train_steps, train_losses, 'b-', linewidth=2, alpha=0.7, label='Train Loss')
        
        eval_steps, eval_losses = zip(*eval_loss)
        axes[2].plot(eval_steps, eval_losses, 'g-', linewidth=2, marker='o', markersize=6, label='Val Loss')
        axes[2].set_xlabel('Step', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=10)
    elif learning_rate and axes[1] is not None:
        # Learning rate plot
        steps, lrs = zip(*learning_rate)
        axes[1].plot(steps, lrs, 'r-', linewidth=2, label='Learning Rate')
        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = output_path / "training_curves.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: {save_path}")
    
    # Also save as PDF for better quality
    pdf_path = save_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"📊 Training curves (PDF) saved to: {pdf_path}")
    
    plt.close()
    
    return save_path


def plot_metrics_summary(
    output_dir: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot a comprehensive summary of all training metrics.
    
    Args:
        output_dir: Directory containing the TensorBoard logs
        save_path: Path to save the plot (default: output_dir/metrics_summary.png)
        figsize: Figure size (width, height)
    """
    output_path = Path(output_dir)
    
    # Read all metrics
    train_loss = read_tensorboard_scalars(output_path, 'train/loss')
    eval_loss = read_tensorboard_scalars(output_path, 'eval/loss')
    learning_rate = read_tensorboard_scalars(output_path, 'train/learning_rate')
    grad_norm = read_tensorboard_scalars(output_path, 'train/grad_norm')
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Training Loss
    if train_loss:
        steps, losses = zip(*train_loss)
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    if eval_loss:
        steps, losses = zip(*eval_loss)
        axes[0, 1].plot(steps, losses, 'g-', linewidth=2, marker='o', markersize=6)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Validation Data', 
                       transform=axes[0, 1].transAxes, ha='center', va='center')
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
    
    # Plot 3: Learning Rate
    if learning_rate:
        steps, lrs = zip(*learning_rate)
        axes[1, 0].plot(steps, lrs, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 4: Gradient Norm
    if grad_norm:
        steps, norms = zip(*grad_norm)
        axes[1, 1].plot(steps, norms, 'm-', linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Norm', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = output_path / "metrics_summary.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Metrics summary saved to: {save_path}")
    
    plt.close()
    
    return save_path


def plot_correlation_metrics(
    output_dir: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot PLCC and SRCC correlation metrics over training.
    
    Args:
        output_dir: Directory containing the TensorBoard logs
        save_path: Path to save the plot (default: output_dir/correlation_metrics.png)
        figsize: Figure size (width, height)
    """
    output_path = Path(output_dir)
    
    # Read correlation metrics (try both slash and underscore formats)
    plcc = read_tensorboard_scalars(output_path, 'eval/plcc')
    if not plcc:
        plcc = read_tensorboard_scalars(output_path, 'eval_plcc')
    
    srcc = read_tensorboard_scalars(output_path, 'eval/srcc')
    if not srcc:
        srcc = read_tensorboard_scalars(output_path, 'eval_srcc')
    
    mae = read_tensorboard_scalars(output_path, 'eval/mae')
    if not mae:
        mae = read_tensorboard_scalars(output_path, 'eval_mae')
    
    rmse = read_tensorboard_scalars(output_path, 'eval/rmse')
    if not rmse:
        rmse = read_tensorboard_scalars(output_path, 'eval_rmse')
    
    if not plcc and not srcc:
        print("⚠️  No correlation metrics (PLCC/SRCC) found in TensorBoard logs")
        return None
    
    # Create figure with 2 or 4 subplots depending on available metrics
    if mae or rmse:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = list(axes) + [None, None]  # Pad for consistent indexing
    
    # Plot 1: PLCC (Pearson Linear Correlation Coefficient)
    if plcc:
        steps, values = zip(*plcc)
        axes[0].plot(steps, values, 'b-', linewidth=2.5, marker='o', markersize=8, label='PLCC')
        axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Perfect Correlation')
        axes[0].set_xlabel('Step', fontsize=12)
        axes[0].set_ylabel('PLCC', fontsize=12)
        axes[0].set_title('Pearson Linear Correlation Coefficient', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        axes[0].legend(fontsize=10)
        
        # Add statistics
        final_plcc = values[-1]
        max_plcc = max(values)
        axes[0].text(0.02, 0.02, f'Final: {final_plcc:.4f}\nMax: {max_plcc:.4f}',
                    transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='bottom', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        axes[0].text(0.5, 0.5, 'No PLCC Data', 
                    transform=axes[0].transAxes, fontsize=14,
                    ha='center', va='center')
        axes[0].set_title('PLCC', fontsize=14, fontweight='bold')
    
    # Plot 2: SRCC (Spearman Rank Correlation Coefficient)
    if srcc:
        steps, values = zip(*srcc)
        axes[1].plot(steps, values, 'g-', linewidth=2.5, marker='s', markersize=8, label='SRCC')
        axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Perfect Correlation')
        axes[1].set_xlabel('Step', fontsize=12)
        axes[1].set_ylabel('SRCC', fontsize=12)
        axes[1].set_title('Spearman Rank Correlation Coefficient', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(fontsize=10)
        
        # Add statistics
        final_srcc = values[-1]
        max_srcc = max(values)
        axes[1].text(0.02, 0.02, f'Final: {final_srcc:.4f}\nMax: {max_srcc:.4f}',
                    transform=axes[1].transAxes, fontsize=11,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        axes[1].text(0.5, 0.5, 'No SRCC Data', 
                    transform=axes[1].transAxes, fontsize=14,
                    ha='center', va='center')
        axes[1].set_title('SRCC', fontsize=14, fontweight='bold')
    
    # Plot 3: MAE (Mean Absolute Error) if available
    if mae and axes[2] is not None:
        steps, values = zip(*mae)
        axes[2].plot(steps, values, 'm-', linewidth=2.5, marker='^', markersize=8, label='MAE')
        axes[2].set_xlabel('Step', fontsize=12)
        axes[2].set_ylabel('MAE', fontsize=12)
        axes[2].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=10)
        
        # Add statistics
        final_mae = values[-1]
        min_mae = min(values)
        axes[2].text(0.02, 0.98, f'Final: {final_mae:.4f}\nMin: {min_mae:.4f}',
                    transform=axes[2].transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # Plot 4: RMSE (Root Mean Squared Error) if available
    if rmse and axes[3] is not None:
        steps, values = zip(*rmse)
        axes[3].plot(steps, values, 'c-', linewidth=2.5, marker='d', markersize=8, label='RMSE')
        axes[3].set_xlabel('Step', fontsize=12)
        axes[3].set_ylabel('RMSE', fontsize=12)
        axes[3].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(fontsize=10)
        
        # Add statistics
        final_rmse = values[-1]
        min_rmse = min(values)
        axes[3].text(0.02, 0.98, f'Final: {final_rmse:.4f}\nMin: {min_rmse:.4f}',
                    transform=axes[3].transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = output_path / "correlation_metrics.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Correlation metrics saved to: {save_path}")
    
    # Also save as PDF
    pdf_path = save_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"📊 Correlation metrics (PDF) saved to: {pdf_path}")
    
    plt.close()
    
    return save_path

