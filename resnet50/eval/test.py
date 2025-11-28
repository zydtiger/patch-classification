import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import typer
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from model import ResNet50Classifier, create_model
from utils import create_image_data_loaders, get_best_device
from transforms import get_val_transforms
from embed import TIFDataset

app = typer.Typer()


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    class_to_idx: Dict[str, int]
) -> Dict[str, Any]:
    """
    Evaluate model performance on a dataset.

    Args:
        model: Trained model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_to_idx: Mapping from class names to indices

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    with torch.no_grad():
        for batch_idx, (images, targets_ohe, targets) in enumerate(data_loader):
            images = images.to(device)
            targets_ohe = targets_ohe.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets_ohe)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_targets, all_predictions)

    # Convert indices to class names for report
    target_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    # Classification report
    class_report = classification_report(
        all_targets,
        all_predictions,
        target_names=target_names,
        output_dict=True
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'targets': all_targets,
        'class_names': target_names
    }


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str = "Confusion Matrix"
):
    """
    Plot and save confusion matrix.

    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


@app.command()
def main(
    checkpoint_path: Path = typer.Option(
        ...,
        help="Path to model checkpoint file (.pth)"
    ),
    data_root: Path = typer.Option(
        "data/NCT-CRC-HE-100K",
        help="Root directory containing class folders with TIF images"
    ),
    output_dir: Path = typer.Option(
        "eval_results",
        help="Directory to save evaluation results"
    ),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    img_size: int = typer.Option(224, help="Input image size"),
    device: str = typer.Option("auto", help="Device to use (auto/cuda/mps/cpu)"),
    test_size: float = typer.Option(0.2, help="Fraction of data to use for testing"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility"),
    transform_config: str = typer.Option("standard", help="Transform configuration"),
):
    """
    Evaluate a trained ResNet50 model on test dataset.
    """
    # Set device
    if device == "auto":
        device = get_best_device()

    print(f"Evaluating model on device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root: {data_root}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint to get model info
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Determine number of classes from checkpoint
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            # Look for fc layer weight to determine number of classes
            for key, value in model_state.items():
                if 'fc' in key and 'weight' in key:
                    num_classes = value.shape[0]
                    break
            else:
                raise ValueError("Could not determine number of classes from checkpoint")
        else:
            raise ValueError("Invalid checkpoint format")

        print(f"Detected {num_classes} classes in model")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Create model
    model = create_model(num_classes=num_classes, feature_extract=False)

    # Load checkpoint
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        return

    model = model.to(device)

    # Create data loaders
    try:
        print("Creating data loaders...")
        train_loader, test_loader, class_to_idx = create_image_data_loaders(
            batch_size=batch_size,
            data_root=str(data_root),
            test_size=test_size,
            num_workers=4,
            img_size=img_size,
            random_seed=random_seed,
            transform_config=transform_config
        )

        print(f"Class mapping: {class_to_idx}")
        print(f"Test batches: {len(test_loader)}")

    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return

    # Evaluate model
    print("Running evaluation...")
    try:
        results = evaluate_model(model, test_loader, device, class_to_idx)

        print("\n=== Evaluation Results ===")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test Loss: {results['loss']:.4f}")

        # Print classification report
        print("\n=== Classification Report ===")
        print(classification_report(
            results['targets'],
            results['predictions'],
            target_names=results['class_names']
        ))

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return

    # Save results
    try:
        # Save detailed report
        report_path = output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("ResNet50 Model Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {checkpoint_path}\n")
            f.write(f"Dataset: {data_root}\n")
            f.write(f"Test samples: {len(results['targets'])}\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Loss: {results['loss']:.4f}\n\n")

            f.write("Classification Report:\n")
            f.write(classification_report(
                results['targets'],
                results['predictions'],
                target_names=results['class_names']
            ))

        print(f"📄 Classification report saved to: {report_path}")

        # Save confusion matrix plot
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            results['confusion_matrix'],
            results['class_names'],
            cm_path,
            title=f"Confusion Matrix (Acc: {results['accuracy']:.3f})"
        )
        print(f"📊 Confusion matrix saved to: {cm_path}")

        # Save raw results as numpy
        np.savez(
            output_dir / "evaluation_results.npz",
            predictions=results['predictions'],
            targets=results['targets'],
            class_names=results['class_names'],
            confusion_matrix=results['confusion_matrix'],
            accuracy=results['accuracy'],
            loss=results['loss']
        )

        print(f"💾 Raw results saved to: {output_dir / 'evaluation_results.npz'}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")

    print(f"\n✅ Evaluation completed! Results saved to: {output_dir}")


@app.command()
def compare_checkpoints(
    checkpoint_paths: list[Path] = typer.Argument(..., help="List of checkpoint files to compare"),
    data_root: Path = typer.Option(
        "data/NCT-CRC-HE-100K",
        help="Root directory containing class folders with TIF images"
    ),
    output_dir: Path = typer.Option(
        "comparison_results",
        help="Directory to save comparison results"
    ),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
):
    """
    Compare multiple model checkpoints on the same test set.
    """
    # Set device
    device = get_best_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(checkpoint_paths)} checkpoints on device: {device}")

    # Create test data loaders once
    try:
        _, test_loader, class_to_idx = create_image_data_loaders(
            batch_size=batch_size,
            data_root=str(data_root),
            test_size=0.2,
            num_workers=4,
            img_size=224
        )

    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return

    comparison_results = {}

    # Evaluate each checkpoint
    for checkpoint_path in checkpoint_paths:
        print(f"\nEvaluating: {checkpoint_path.name}")

        try:
            # Load checkpoint and create model
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Determine number of classes (assuming all models have same architecture)
            num_classes = 9  # Default for tissue classification
            model = create_model(num_classes=num_classes, feature_extract=False)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(device)

            # Evaluate
            results = evaluate_model(model, test_loader, device, class_to_idx)
            comparison_results[checkpoint_path.stem] = results

            print(f"Accuracy: {results['accuracy']:.4f}, Loss: {results['loss']:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_path.name}: {e}")
            continue

    # Create comparison table
    print(f"\n=== Comparison Results ===")
    print(f"{'Checkpoint':<30} {'Accuracy':<12} {'Loss':<12}")
    print("-" * 54)

    for name, results in comparison_results.items():
        print(f"{name:<30} {results['accuracy']:<12.4f} {results['loss']:<12.4f}")

    # Save comparison results
    try:
        comparison_path = output_dir / "comparison_results.txt"
        with open(comparison_path, 'w') as f:
            f.write("Model Comparison Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"{'Checkpoint':<30} {'Accuracy':<12} {'Loss':<12}\n")
            f.write("-" * 54 + "\n")

            for name, results in comparison_results.items():
                f.write(f"{name:<30} {results['accuracy']:<12.4f} {results['loss']:<12.4f}\n")

        print(f"\n📄 Comparison results saved to: {comparison_path}")

    except Exception as e:
        print(f"❌ Error saving comparison: {e}")


if __name__ == "__main__":
    app()