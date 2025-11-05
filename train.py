import numpy as np
import torch.nn as nn
import torch.optim as optim
import typer
import os
import shutil

from model import PatchMIL
from dataloader import get_loaders, get_latest_checkpoint
from training_manager import TrainingManager, EarlyStopping

app = typer.Typer()


@app.command()
def main(
    embed_folder: str = typer.Argument(
        ..., help="Input path of the embeddings generated from the patches."
    ),
    embed_dim: int = typer.Option(
        ..., "--embed_dim", help="Embedding dimension of the foundation model."
    ),
    tiles_per_patch: int = typer.Option(
        256, "--tpp", help="The number of ViT tiles per image patch."
    ),
    device: str = typer.Option(
        "cpu", "--device", help="Device to train the model on."
    ),
    num_classes: int = typer.Option(
        11, "--num_classes", help="Number of output classes."
    ),
    learning_rate: float = typer.Option(
        0.01, "--lr", help="Learning rate when training the model."
    ),
    batch_size: int = typer.Option(
        256, "--batch_size", help="Batch size of training."
    ),
    num_epochs: int = typer.Option(
        32,
        "--num_epochs",
        help="Maximum number of epochs to train the model on.",
    ),
    checkpoint_dir: str = typer.Option(
        "./checkpoints",
        "--checkpoint_dir",
        help="Path for storing each checkpoint after every epoch.",
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume training from latest checkpoint."
    ),
):
    # Initialize model, criterion, and optimizer
    model = PatchMIL(tiles_per_patch, embed_dim, num_classes)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=2, min_delta=0)

    # Get data loaders
    train_loader, val_loader = get_loaders(
        batch_size,
        embed_folder,
        "patch",
        embed_dim,
        num_classes,
        dtype=np.float16,
        test_size=0.2,
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
        logdir="runs/patch_classification_experiment",
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


if __name__ == "__main__":
    app()
