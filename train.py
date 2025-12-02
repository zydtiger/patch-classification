import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import typer
import os
import shutil
from pathlib import Path

from model import PatchMIL, MulticlassMLP
from utils import create_data_loaders, get_latest_checkpoint
from training_manager import TrainingManager, EarlyStopping

app = typer.Typer()

# Global configuration constants
NUM_WORKERS = 12
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
DROP_LAST = True


@app.command()
def main(
    name: str = typer.Option(..., help="Name of training experiment."),
    embed_dir: Path = typer.Option(
        ..., help="Input path of the embeddings generated from the patches."
    ),
    embed_dim: int = typer.Option(
        ..., help="Embedding dimension of the foundation model."
    ),
    num_classes: int = typer.Option(..., help="Number of output classes."),
    dataset_type: str = typer.Option(
        ...,
        help="Type of dataset: 'cls' for classification using CLS token, 'patch' for patch-level MIL.",
    ),
    tiles_per_patch: int = typer.Option(
        256,
        "--tpp",
        help="The number of ViT tiles per image patch (only used for patch mode).",
    ),
    device: str = typer.Option("cpu", "--device", help="Device to train the model on."),
    learning_rate: float = typer.Option(
        0.01, "--lr", help="Learning rate when training the model."
    ),
    batch_size: int = typer.Option(256, help="Batch size of training."),
    num_epochs: int = typer.Option(
        32,
        help="Maximum number of epochs to train the model on.",
    ),
    resume: bool = typer.Option(False, help="Resume training from latest checkpoint."),
):
    # Define checkpoint directory path
    checkpoint_dir = Path(f"runs/{name}/checkpoints")

    assert dataset_type == "cls" or dataset_type == "patch"
    if dataset_type == "patch":
        model = PatchMIL(tiles_per_patch, embed_dim, num_classes)
        summary(model, input_size=(2, tiles_per_patch, embed_dim))
    else:  # cls
        model = MulticlassMLP(embed_dim, num_classes)
        summary(model, input_size=(2, embed_dim))

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=2, min_delta=0)

    # Get data loaders
    train_loader, val_loader, _ = create_data_loaders(
        batch_size=batch_size,
        dataset_type=dataset_type,
        embed_dim=embed_dim,
        num_classes=num_classes,
        test_size=0.2,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        drop_last=DROP_LAST,
        embed_dir=embed_dir,
    )

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

    # Train model
    trainer.train(num_epochs, train_loader, val_loader, num_classes)

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


if __name__ == "__main__":
    app()
