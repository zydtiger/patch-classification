import torch
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .transforms import get_transforms_by_config
from .dataset import TIFDataset


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


def create_image_data_loaders(
    batch_size: int,
    data_root: str,
    test_size: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    random_seed: int = 42,
    stratify: bool = True,
    img_size: int = 224,
    transform_config: str = "standard",
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation data loaders for raw TIF images.

    Args:
        batch_size: Batch size for data loading
        data_root: Root directory containing class folders with TIF images
        test_size: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Whether to keep workers alive between epochs
        random_seed: Random seed for reproducible splits
        stratify: Whether to use stratified sampling for train/val split
        img_size: Target image size for resizing
        transform_config: Transform configuration ('standard', 'strong', 'minimal')

    Returns:
        Tuple of (train_loader, val_loader, class_to_idx)
    """
    print(f"Creating image data loaders from: {data_root}")

    transforms_dict = get_transforms_by_config(transform_config)

    # Create full dataset to get samples and class mapping
    full_dataset = TIFDataset(data_root=data_root, transform=None)

    # Use the dataset's built-in class_to_idx mapping
    class_to_idx = full_dataset.class_to_idx
    class_names = list(class_to_idx.keys())

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Total samples: {len(full_dataset)}")

    # Extract class labels using the same method as the dataset
    class_labels = []
    for sample_path in full_dataset.samples:
        # Use the same filename parsing logic as TIFDataset
        class_name = sample_path.stem.split("-")[0]
        class_labels.append(class_name)

    # Generate indices for train/val split
    indices = list(range(len(full_dataset)))

    if stratify:
        # Use stratified sampling to maintain class distribution
        train_indices, val_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=class_labels,
        )
    else:
        # Simple random split
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, random_state=random_seed
        )

    print(
        f"Dataset split - Train: {len(train_indices)}, Validation: {len(val_indices)}"
    )

    # Create train dataset with train transforms
    train_samples = [full_dataset.samples[i] for i in train_indices]
    train_dataset = TIFDataset(
        data_root=data_root,
        num_classes=len(class_names),
        transform=transforms_dict["train"],
    )
    train_dataset.samples = train_samples  # Override with train samples only
    train_dataset.class_to_idx = class_to_idx  # Ensure consistent class mapping

    # Create validation dataset with val transforms
    val_samples = [full_dataset.samples[i] for i in val_indices]
    val_dataset = TIFDataset(
        data_root=data_root,
        num_classes=len(class_names),
        transform=transforms_dict["val"],
    )
    val_dataset.samples = val_samples  # Override with val samples only
    val_dataset.class_to_idx = class_to_idx  # Ensure consistent class mapping

    # Create data loaders with optimal settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=False,  # Keep all validation samples
    )

    print(
        f"Created data loaders - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    return train_loader, val_loader, class_to_idx


def get_class_weights(class_counts: dict, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.

    Args:
        class_counts: Dictionary mapping class names to sample counts
        num_classes: Total number of classes

    Returns:
        torch.Tensor: Class weights tensor
    """
    total_samples = sum(class_counts.values())

    # Calculate inverse frequency weights
    weights = []
    for i in range(num_classes):
        class_name = list(class_counts.keys())[i]
        count = class_counts[class_name]
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    weights = torch.FloatTensor(weights)

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return weights


def get_dataset_statistics(data_root: str, transform_config: str = "standard") -> dict:
    """
    Calculate dataset statistics for analysis.

    Args:
        data_root: Root directory containing class folders
        transform_config: Transform configuration name

    Returns:
        dict: Dataset statistics
    """

    transforms_dict = get_transforms_by_config(transform_config)

    dataset = TIFDataset(data_root=data_root, transform=transforms_dict["val"])

    # Count samples per class using the same method as TIFDataset
    class_counts = {}
    for sample_path in dataset.samples:
        # Use the same filename parsing logic as TIFDataset
        class_name = sample_path.stem.split("-")[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    total_samples = len(dataset)

    # Calculate percentages
    class_percentages = {
        class_name: (count / total_samples) * 100
        for class_name, count in class_counts.items()
    }

    statistics = {
        "total_samples": total_samples,
        "num_classes": len(class_counts),
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "min_class_size": min(class_counts.values()),
        "max_class_size": max(class_counts.values()),
        "imbalance_ratio": max(class_counts.values()) / min(class_counts.values()),
    }

    return statistics


def create_balanced_sampler(
    data_root: str, samples_per_class: Optional[int] = None
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a balanced sampler for handling class imbalance.

    Args:
        data_root: Root directory containing class folders
        samples_per_class: Number of samples per class. If None, uses minority class size

    Returns:
        WeightedRandomSampler: Sampler for balanced batches
    """
    dataset = TIFDataset(data_root=data_root, transform=None)

    # Count samples per class using the same method as TIFDataset
    class_counts = {}
    sample_to_class = []

    for sample_path in dataset.samples:
        # Use the same filename parsing logic as TIFDataset
        class_name = sample_path.stem.split("-")[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        sample_to_class.append(class_name)

    # Determine samples per class
    if samples_per_class is None:
        samples_per_class = min(class_counts.values())

    # Calculate weights for balanced sampling
    class_weights = {
        class_name: 1.0 / count for class_name, count in class_counts.items()
    }
    sample_weights = [class_weights[class_name] for class_name in sample_to_class]

    # Ensure samples_per_class is an integer for type checker
    samples_per_class = int(samples_per_class)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=samples_per_class * len(class_counts),
        replacement=True,
    )

    return sampler


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


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    feature_extract: bool = True,
    backbone_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """
    Create optimizer with differential learning rates.

    Args:
        model: Model to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay for regularization
        feature_extract: Whether model is in feature extraction mode
        backbone_lr: Learning rate for backbone (if different from base lr)

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if feature_extract:
        # Only optimize classifier parameters
        params = model.backbone.fc.parameters()
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    if backbone_lr is not None:
        # Differential learning rates
        layer_params = model.get_layer_params()
        params = [
            {"params": layer_params["classifier"], "lr": learning_rate},
            {"params": layer_params["backbone"], "lr": backbone_lr},
        ]
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # Single learning rate for all parameters
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
