import torch
import torch.nn as nn
import typer
import os
import shutil
from pathlib import Path

from .model import VGG16Classifier
from .utils import (
    create_image_data_loaders,
    get_best_device,
    get_latest_checkpoint,
    create_optimizer,
    get_dataset_statistics,
    get_class_weights,
)

# Import training manager from parent directory
import sys

sys.path.append(str(Path(__file__).parent.parent))
from training_manager import TrainingManager
from early_stopping import EarlyStopping

app = typer.Typer()

# Global configuration constants
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
DROP_LAST = True


@app.command()
def main(
    name: str = typer.Option(..., help="Name of training experiment."),
    data_root: Path = typer.Option(
        "data/NCT-CRC-HE-100K",
        help="Root directory containing class folders with TIF images.",
    ),
    num_classes: int = typer.Option(9, help="Number of output classes."),
    img_size: int = typer.Option(224, help="Input image size for VGG16."),
    device: str = typer.Option(
        "auto", "--device", help="Device to train the model on (auto/cuda/mps/cpu)."
    ),
    learning_rate: float = typer.Option(
        1e-4, "--lr", help="Learning rate when training the model (lower for VGG16)."
    ),
    backbone_lr: float = typer.Option(
        5e-5,
        "--backbone-lr",
        help="Learning rate for backbone (only used in fine-tuning mode, lower for VGG16).",
    ),
    weight_decay: float = typer.Option(
        1e-3, "--wd", help="Weight decay for regularization (higher for VGG16)."
    ),
    batch_size: int = typer.Option(
        16, help="Batch size of training (smaller for VGG16 memory requirements)."
    ),
    num_epochs: int = typer.Option(
        40,
        help="Maximum number of epochs to train the model on (shorter to avoid overfitting).",
    ),
    test_size: float = typer.Option(
        0.2,
        help="Fraction of data to use for validation.",
    ),
    resume: bool = typer.Option(False, help="Resume training from latest checkpoint."),
    feature_extract: bool = typer.Option(
        True,
        help="Use feature extraction mode (freeze backbone) or fine-tuning mode.",
    ),
    balanced_sampling: bool = typer.Option(
        False,
        help="Use balanced sampling to handle class imbalance.",
    ),
    transform_config: str = typer.Option(
        "standard",
        help="Transform configuration (standard/minimal).",
    ),
    use_class_weights: bool = typer.Option(
        False,
        help="Use class weights to handle imbalance in loss function.",
    ),
    random_seed: int = typer.Option(
        42,
        help="Random seed for reproducibility.",
    ),
):
    """
    Train VGG16 classifier on histopathology images.

    VGG16-specific optimizations:
    - Smaller batch size (16) for memory efficiency
    - Lower learning rates (1e-4 classifier, 5e-5 backbone) for stability
    - Higher weight decay (1e-3) for regularization
    - Stronger augmentation to prevent overfitting
    - Shorter training (40 epochs) to avoid overfitting
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Determine device
    if device == "auto":
        device = get_best_device()

    print(f"Training VGG16 on device: {device}")
    print(f"Feature extraction mode: {feature_extract}")
    print(f"Transform configuration: {transform_config}")
    print(
        f"VGG16 optimizations: batch_size={batch_size}, lr={learning_rate}, wd={weight_decay}"
    )

    # Define checkpoint directory path
    checkpoint_dir = Path(f"runs/{name}/checkpoints")

    # Create model
    model = VGG16Classifier(num_classes=num_classes, feature_extract=feature_extract)

    print(
        f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    print(f"Total VGG16 parameters: {sum(p.numel() for p in model.parameters())}")

    # Get dataset statistics
    print("Analyzing dataset...")
    dataset_stats = get_dataset_statistics(str(data_root), transform_config)
    print(
        f"Dataset: {dataset_stats['total_samples']} samples, {dataset_stats['num_classes']} classes"
    )
    print(f"Class imbalance ratio: {dataset_stats['imbalance_ratio']:.2f}")

    # Create data loaders
    train_loader, val_loader, class_to_idx = create_image_data_loaders(
        batch_size=batch_size,
        data_root=str(data_root),
        test_size=test_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=DROP_LAST,
        prefetch_factor=PREFETCH_FACTOR,
        random_seed=random_seed,
        img_size=img_size,
        transform_config=transform_config,
    )

    print(f"Class mapping: {class_to_idx}")

    # Setup loss function with optional class weights
    if use_class_weights:
        class_weights = get_class_weights(dataset_stats["class_counts"], num_classes)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Create optimizer with appropriate learning rates
    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        feature_extract=feature_extract,
        backbone_lr=backbone_lr if not feature_extract else None,
    )

    # Setup early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Handle checkpoint directory based on resume flag
    if not resume and os.path.exists(checkpoint_dir):
        print(f"Removing existing checkpoint directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    # Initialize training manager
    trainer = TrainingManager(
        model,
        optimizer,
        criterion,
        early_stopping,
        logdir=f"runs/{name}",
        monitor="val_loss",
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Load checkpoint if specified
    if resume:
        try:
            latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
            print(f"Resuming from latest checkpoint: {latest_checkpoint}")
            trainer.load_checkpoint(latest_checkpoint)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Starting training from scratch...")

    # Start training using TrainingManager
    trainer.train(
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
    )

    # Load best model for final evaluation
    try:
        best_model_state = early_stopping.best_model
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")
    except:
        print("Warning: Could not load best model weights")

    # Final evaluation
    print("\nRunning final evaluation...")
    final_val_loss, final_val_acc = trainer.validate(val_loader, num_epochs)
    print(f"Final validation accuracy: {final_val_acc:.2f}%")

    # Save latest checkpoint after training completes
    latest_checkpoint_path = checkpoint_dir / "latest.pth"
    try:
        # Use existing utility function to find the latest checkpoint
        latest_file = get_latest_checkpoint(checkpoint_dir)
        shutil.copy2(latest_file, latest_checkpoint_path)
        print(f"Saved latest checkpoint to: {latest_checkpoint_path}")
    except FileNotFoundError:
        print("No checkpoint files found to copy as latest.pth")

    # Create .gitignore file in checkpoint directory
    gitignore_path = checkpoint_dir / ".gitignore"
    gitignore_content = "*.pth\n!latest.pth\n"

    with open(gitignore_path, "w") as f:
        f.write(gitignore_content)
    print(f"Created .gitignore file: {gitignore_path}")

    print(f"\nVGG16 training completed! Results saved to: runs/{name}/")


if __name__ == "__main__":
    app()
