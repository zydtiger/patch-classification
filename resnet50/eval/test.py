import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import typer
from pathlib import Path
from typing import Dict, Any
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from resnet50.model import ResNet50Classifier
from resnet50.utils import create_image_data_loaders, get_best_device
from eval.vis_utils import (
    plot_confusion_matrix,
    plot_multiclass_roc_curve,
)

app = typer.Typer()


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: str, class_to_idx: Dict[str, int]
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
    all_probabilities = []
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

            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions, probabilities, and targets using append
            all_predictions.append(predicted.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Stack lists into numpy arrays
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    all_targets = np.concatenate(all_targets)

    # Calculate metrics
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_targets, all_predictions)

    # Convert indices to class names for report
    target_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    # Classification report
    class_report = classification_report(
        all_targets, all_predictions, target_names=target_names, output_dict=True
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "targets": all_targets,
        "class_names": target_names,
    }


@app.command()
def main(
    model_path: Path = typer.Option(..., help="Path to model checkpoint file (.pth)"),
    data_root: Path = typer.Option(
        "data/NCT-CRC-HE-100K",
        help="Root directory containing class folders with TIF images",
    ),
    name: str = typer.Option(..., help="Experiment name for organizing results"),
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
    print(f"Checkpoint: {model_path}")
    print(f"Data root: {data_root}")

    # Create output directory
    output_dir = Path(f"runs/{name}/figs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint to get model info
    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # Determine number of classes from checkpoint
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            # Look for the final fc layer weight (fc.8 is the final layer in our sequential)
            final_layer_keys = [key for key in model_state.keys() if key.endswith("fc.8.weight")]
            if final_layer_keys:
                num_classes = model_state[final_layer_keys[0]].shape[0]
            else:
                # Fallback: look for any fc layer that ends with weight
                for key, value in model_state.items():
                    if "fc" in key and "weight" in key:
                        num_classes = value.shape[0]
                        break
                else:
                    raise ValueError(
                        "Could not determine number of classes from checkpoint"
                    )
        else:
            raise ValueError("Invalid checkpoint format")

        print(f"Detected {num_classes} classes in model")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Create model
    model = ResNet50Classifier(num_classes=num_classes, feature_extract=False)

    # Load checkpoint
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
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
            transform_config=transform_config,
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
        print(
            classification_report(
                results["targets"],
                results["predictions"],
                target_names=results["class_names"],
            )
        )

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return

    # Save confusion matrix plot
    cm_path = output_dir / "confusion_matrix.png"
    cm_fig = plot_confusion_matrix(
        results["targets"],
        results["predictions"],
        len(results["class_names"]),
        results["class_names"],
    )
    cm_fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"📊 Confusion matrix saved to: {cm_path}")

    roc_path = output_dir / "roc_curve.png"
    roc_fig = plot_multiclass_roc_curve(
        results["targets"],
        results["probabilities"],
        len(results["class_names"]),
        results["class_names"],
    )
    roc_fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    print(f"📈 ROC curve saved to: {roc_path}")

    print(f"\n✅ Evaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    app()
