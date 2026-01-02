"""Training visualization utilities."""

import os
import matplotlib.pyplot as plt


def plot_training_curves(history, save_dir, model_name="Model"):
    """
    Plot training curves and save to file.
    
    Args:
        history: dict with keys 'train_loss', 'val_loss', 'val_add', 'val_acc', 'lr'
        save_dir: directory to save the plot
        model_name: name to include in plot title
    
    Returns:
        path to saved plot
    """
    if len(history.get('train_loss', [])) == 0:
        print("No training history to plot")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # ADD plot
    axes[0, 1].plot(history['val_add'], label='Val ADD (mm)', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('ADD (mm)')
    axes[0, 1].set_title('Validation ADD Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy plot
    axes[1, 0].plot(history['val_acc'], label='Val ACC (%)', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Validation ADD-2cm Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate plot
    axes[1, 1].plot(history['lr'], label='Learning Rate', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.suptitle(f'{model_name} Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")
    
    return plot_path
