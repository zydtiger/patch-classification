import torch
from pathlib import Path
from typing import Literal, Tuple
from torch.utils.data import DataLoader
from dataloader import EmbeddingDataset


def get_best_device():
    """
    Get the best available device for PyTorch computations.

    Returns:
        str: The best available device ('cuda', 'mps', or 'cpu')

    Priority order:
        1. 'cuda' - CUDA GPU if available
        2. 'mps' - Apple Metal Performance Shaders if available
        3. 'cpu' - CPU as final fallback
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_data_loaders(
    batch_size: int,
    dataset_type: Literal["cls", "patch"],
    embed_dim: int,
    num_classes: int,
    test_size: float,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
    drop_last: bool,
    embed_dir: Path,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation data loaders with optimal settings.

    Args:
        batch_size: Batch size for data loading
        dataset_type: Type of dataset ("cls" or "patch")
        embed_dim: Dimension of embeddings
        num_classes: Number of classes
        test_size: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches to prefetch
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        drop_last: Whether to drop the last incomplete batch
        embed_dir: Directory containing embedding files

    Returns:
        Tuple of (train_loader, val_loader)
    """
    print(f"Creating {dataset_type} data loaders")

    # Create full dataset
    full_dataset = EmbeddingDataset(
        dataset_type=dataset_type,
        embed_dim=embed_dim,
        num_classes=num_classes,
        embed_dir=embed_dir,
    )

    # Calculate train/validation split
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * test_size)
    train_size = dataset_size - val_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Dataset split - Train: {train_size}, Validation: {val_size}")

    # Create data loaders with optimal settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,  # Keep all validation samples
    )

    print(
        f"Created data loaders - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    return train_loader, val_loader, full_dataset.class_to_idx


def get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Find the latest checkpoint file based on epoch number.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to the latest checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint files are found
    """
    pattern = "checkpoint_epoch_*.pth"
    checkpoint_files = list(checkpoint_dir.glob(pattern))

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir} with pattern {pattern}"
        )

    # Extract epoch numbers and find the latest
    latest_checkpoint = max(
        checkpoint_files,
        key=lambda x: int(x.stem.split("_epoch_")[-1]),
    )

    return latest_checkpoint
