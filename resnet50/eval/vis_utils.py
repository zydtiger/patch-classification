"""
Visualization utilities for ResNet50 evaluation.

This module provides visualization functions that are compatible with the existing
evaluation framework while being adapted for ResNet50-specific outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path

# Import existing visualization utilities from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from eval.vis_utils import (
        create_confusion_matrix_plot,
        create_class_distribution_plot,
        create_training_curves_plot,
        create_roc_curves,
        create_precision_recall_curves
    )
    print("✅ Imported existing visualization utilities")
except ImportError as e:
    print(f"⚠️  Could not import existing utilities: {e}")
    print("Using fallback visualization functions")


def plot_resnet_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[Path] = None,
    title: str = "ResNet50 Training History"
):
    """
    Plot training history for ResNet50 model.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    plt.show()


def plot_class_performance_comparison(
    embedding_metrics: Dict,
    resnet_metrics: Dict,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Model Comparison by Class"
):
    """
    Compare performance between embedding-based and ResNet50 models by class.

    Args:
        embedding_metrics: Classification report from embedding model
        resnet_metrics: Classification report from ResNet50 model
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    # Extract metrics for each class
    embedding_f1 = [embedding_metrics[class_name]['f1-score'] for class_name in class_names]
    resnet_f1 = [resnet_metrics[class_name]['f1-score'] for class_name in class_names]

    embedding_precision = [embedding_metrics[class_name]['precision'] for class_name in class_names]
    resnet_precision = [resnet_metrics[class_name]['precision'] for class_name in class_names]

    embedding_recall = [embedding_metrics[class_name]['recall'] for class_name in class_names]
    resnet_recall = [resnet_metrics[class_name]['recall'] for class_name in class_names]

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(class_names))
    width = 0.35

    # F1 Score comparison
    ax1.bar(x - width/2, embedding_f1, width, label='Embedding Model', alpha=0.8)
    ax1.bar(x + width/2, resnet_f1, width, label='ResNet50', alpha=0.8)
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision comparison
    ax2.bar(x - width/2, embedding_precision, width, label='Embedding Model', alpha=0.8)
    ax2.bar(x + width/2, resnet_precision, width, label='ResNet50', alpha=0.8)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Recall comparison
    ax3.bar(x - width/2, embedding_recall, width, label='Embedding Model', alpha=0.8)
    ax3.bar(x + width/2, resnet_recall, width, label='ResNet50', alpha=0.8)
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Recall')
    ax3.set_title('Recall Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class comparison plot saved to: {save_path}")

    plt.show()


def plot_feature_extractor_outputs(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    class_names: List[str],
    num_samples: int = 1000,
    save_path: Optional[Path] = None,
    title: str = "ResNet50 Feature Extractor Outputs"
):
    """
    Visualize the feature extractor outputs using t-SNE.

    Args:
        model: Trained ResNet50 model
        data_loader: DataLoader for visualization data
        device: Device to run inference on
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
        title: Plot title
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder

    print("Extracting features for t-SNE visualization...")

    model.eval()
    features = []
    labels = []

    # Get feature extractor (remove classifier)
    feature_extractor = model.get_feature_extractor()

    sample_count = 0
    with torch.no_grad():
        for images, _, targets in data_loader:
            if sample_count >= num_samples:
                break

            images = images.to(device)
            batch_features = feature_extractor(images)
            batch_features = batch_features.view(batch_features.size(0), -1)

            features.extend(batch_features.cpu().numpy())
            labels.extend(targets.cpu().numpy())

            sample_count += len(images)

    features = np.array(features[:num_samples])
    labels = np.array(labels[:num_samples])

    print(f"Extracted {len(features)} features")

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # Create plot
    plt.figure(figsize=(12, 10))

    # Convert numeric labels to class names for coloring
    le = LabelEncoder()
    label_indices = le.fit_transform(labels)

    # Create scatter plot
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=label_indices,
        cmap='tab10',
        alpha=0.7,
        s=50
    )

    # Add legend
    legend1 = plt.legend(
        handles=scatter.legend_elements()[0],
        labels=class_names,
        title="Classes",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.gca().add_artist(legend1)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to: {save_path}")

    plt.show()


def plot_grad_cam_heatmap(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    class_name: str,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Generate Grad-CAM heatmap for model interpretability.

    Args:
        model: Trained ResNet50 model
        image: Input image tensor
        target_class: Target class index
        class_name: Target class name
        save_path: Path to save the plot
        title: Plot title
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("⚠️  pytorch-grad-cam not installed. Install with: pip install pytorch-grad-cam")
        return

    model.eval()

    # Setup Grad-CAM
    target_layers = [model.backbone.layer4[-1]]  # Last layer of ResNet50
    cam = GradCAM(model=model, target_layers=target_layers)

    # Create targets
    targets = [ClassifierOutputTarget(target_class)]

    # Generate CAM
    grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Convert image for visualization
    image_np = image.permute(1, 2, 0).numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # Create visualization
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = heatmap * 0.4 + image_np * 0.6

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image_np)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(heatmap)
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')

    ax3.imshow(superimposed_img)
    ax3.set_title(f'Superimposed ({class_name})')
    ax3.axis('off')

    if title is None:
        title = f'Grad-CAM Visualization for {class_name}'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to: {save_path}")

    plt.show()


def create_comprehensive_evaluation_report(
    embedding_results: Dict,
    resnet_results: Dict,
    save_dir: Path,
    class_names: List[str]
):
    """
    Create a comprehensive evaluation report comparing both models.

    Args:
        embedding_results: Results from embedding-based model
        resnet_results: Results from ResNet50 model
        save_dir: Directory to save all plots
        class_names: List of class names
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall performance comparison
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    embedding_values = [
        embedding_results['accuracy'],
        np.mean([embedding_results['classification_report'][cls]['precision'] for cls in class_names]),
        np.mean([embedding_results['classification_report'][cls]['recall'] for cls in class_names]),
        np.mean([embedding_results['classification_report'][cls]['f1-score'] for cls in class_names])
    ]

    resnet_values = [
        resnet_results['accuracy'],
        np.mean([resnet_results['classification_report'][cls]['precision'] for cls in class_names]),
        np.mean([resnet_results['classification_report'][cls]['recall'] for cls in class_names]),
        np.mean([resnet_results['classification_report'][cls]['f1-score'] for cls in class_names])
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, embedding_values, width, label='Embedding Model', alpha=0.8)
    plt.bar(x + width/2, resnet_values, width, label='ResNet50', alpha=0.8)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Overall Model Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Class-by-class comparison
    plot_class_performance_comparison(
        embedding_results['classification_report'],
        resnet_results['classification_report'],
        class_names,
        save_dir / 'class_comparison.png',
        'Class-wise Performance Comparison'
    )
    plt.close()

    # 3. Confusion matrices side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Embedding model confusion matrix
    sns.heatmap(
        embedding_results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1
    )
    ax1.set_title('Embedding Model Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    # ResNet50 confusion matrix
    sns.heatmap(
        resnet_results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2
    )
    ax2.set_title('ResNet50 Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comprehensive evaluation report saved to: {save_dir}")


# Fallback functions if existing utilities are not available
def create_confusion_matrix_plot(conf_matrix, class_names, save_path=None):
    """Fallback confusion matrix plot function."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_class_distribution_plot(class_counts, save_path=None):
    """Fallback class distribution plot function."""
    plt.figure(figsize=(10, 6))
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.bar(class_names, counts)
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()