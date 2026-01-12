"""Plot training curves from saved history JSON file."""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_history(weights_dir, model_name=None):
    """
    Generate comprehensive training plots from history JSON.
    
    Args:
        weights_dir: Directory containing training_history.json
        model_name: Optional model name for title (auto-detected if not provided)
    """
    history_path = os.path.join(weights_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Auto-detect model name from directory
    if model_name is None:
        dir_name = os.path.basename(weights_dir)
        if 'rgb' in dir_name.lower() and 'rgbd' not in dir_name.lower():
            model_name = 'RGB'
        elif 'rgbd' in dir_name.lower():
            model_name = 'RGBD'
        else:
            model_name = 'Model'
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'{model_name} Model Training History', fontsize=14, fontweight='bold')
    
    # 1. Training & Validation Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, history['val_loss'], 'orange', label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. ADD Error
    ax = axes[0, 1]
    val_add = history.get('val_add', [])
    if val_add:
        ax.plot(epochs, val_add, 'g-', linewidth=1.5)
        best_idx = np.argmin(val_add)
        best_add = val_add[best_idx]
        ax.axhline(y=best_add, color='g', linestyle='--', alpha=0.5, label=f'Best: {best_add:.1f}mm')
        ax.scatter([best_idx + 1], [best_add], color='g', s=50, zorder=5)
        ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ADD Error (mm)')
    ax.set_title('ADD Error on Validation Set')
    ax.grid(True, alpha=0.3)
    
    # 3. Validation Accuracy (ACC@2cm only)
    ax = axes[0, 2]
    val_acc_2cm = history.get('val_acc_2cm', history.get('val_acc', []))
    if val_acc_2cm:
        ax.plot(epochs, val_acc_2cm, 'purple', label='ACC@2cm', linewidth=1.5)
        best_idx = np.argmax(val_acc_2cm)
        best_acc = val_acc_2cm[best_idx]
        ax.axhline(y=best_acc, color='purple', linestyle='--', alpha=0.5, label=f'Best: {best_acc:.1f}%')
        ax.scatter([best_idx + 1], [best_acc], color='purple', s=50, zorder=5)
        ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 4. Learning Rate Schedule
    ax = axes[1, 0]
    lr = history.get('lr', [])
    if lr:
        ax.plot(epochs, lr, 'brown', linewidth=1.5)
        ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    # 5. Loss Convergence (Last 30 Epochs)
    ax = axes[1, 1]
    n_last = min(30, len(history['train_loss']))
    if n_last > 5:
        last_epochs = list(range(len(epochs) - n_last + 1, len(epochs) + 1))
        ax.plot(last_epochs, history['train_loss'][-n_last:], 'b-', label='Train Loss', linewidth=1.5)
        ax.plot(last_epochs, history['val_loss'][-n_last:], 'orange', label='Val Loss', linewidth=1.5)
        ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Convergence (Last {n_last} Epochs)')
    ax.grid(True, alpha=0.3)
    
    # 6. Performance Summary (Text)
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate summary stats
    val_add = history.get('val_add', [])
    val_acc_2cm = history.get('val_acc_2cm', history.get('val_acc', []))
    
    summary_text = "Performance Summary\n" + "-" * 30 + "\n\n"
    
    if val_add:
        best_add_idx = np.argmin(val_add)
        summary_text += f"Best ADD Error:    {val_add[best_add_idx]:.2f} mm (Epoch {best_add_idx + 1})\n"
    
    if val_acc_2cm:
        best_acc_idx = np.argmax(val_acc_2cm)
        summary_text += f"Best ACC@2cm:      {val_acc_2cm[best_acc_idx]:.2f}% (Epoch {best_acc_idx + 1})\n"
    
    if history['val_loss']:
        best_loss_idx = np.argmin(history['val_loss'])
        summary_text += f"Best Val Loss:     {history['val_loss'][best_loss_idx]:.4f} (Epoch {best_loss_idx + 1})\n"
    
    summary_text += f"\nFinal Results (Epoch {len(epochs)}):\n"
    if val_add:
        summary_text += f"  - ADD Error:     {val_add[-1]:.2f} mm\n"
    if val_acc_2cm:
        summary_text += f"  - ACC@2cm:       {val_acc_2cm[-1]:.2f}%\n"
    summary_text += f"  - Train Loss:    {history['train_loss'][-1]:.4f}\n"
    summary_text += f"  - Val Loss:      {history['val_loss'][-1]:.4f}\n"
    
    # Improvement stats
    if val_add and len(val_add) > 1:
        improvement = val_add[0] - min(val_add)
        pct_improvement = (improvement / val_add[0]) * 100
        summary_text += f"\nImprovement:\n"
        summary_text += f"  - ADD: {val_add[0]:.1f} -> {min(val_add):.1f} mm ({pct_improvement:.1f}% better)\n"
    
    if val_acc_2cm and len(val_acc_2cm) > 1:
        summary_text += f"  - ACC@2cm: {val_acc_2cm[0]:.1f}% -> {max(val_acc_2cm):.1f}%\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(weights_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {plot_path}")
    
    # Also save summary text
    summary_path = os.path.join(weights_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} Model Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(summary_text.replace("Performance Summary\n" + "-" * 30 + "\n\n", ""))
    print(f"Training summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from history JSON')
    parser.add_argument('weights_dir', nargs='?', default=None,
                        help='Directory containing training_history.json')
    parser.add_argument('--name', '-n', default=None, help='Model name for plot title')
    parser.add_argument('--all', '-a', action='store_true', 
                        help='Plot for all weights directories')
    args = parser.parse_args()
    
    if args.all:
        # Plot for all weights directories
        for dirname in ['weights_rgb', 'weights_rgbd']:
            weights_dir = os.path.join(PROJECT_ROOT, dirname)
            if os.path.exists(os.path.join(weights_dir, 'training_history.json')):
                print(f"\nProcessing {dirname}...")
                plot_training_history(weights_dir)
    elif args.weights_dir:
        # Make path absolute if relative
        if not os.path.isabs(args.weights_dir):
            weights_dir = os.path.join(PROJECT_ROOT, args.weights_dir)
        else:
            weights_dir = args.weights_dir
        plot_training_history(weights_dir, args.name)
    else:
        print("Usage: python plot_training.py <weights_dir> [--name MODEL_NAME]")
        print("       python plot_training.py --all")
        print("\nExamples:")
        print("  python plot_training.py weights_rgb")
        print("  python plot_training.py weights_rgbd --name 'RGBD Model'")
        print("  python plot_training.py --all")


if __name__ == "__main__":
    main()
