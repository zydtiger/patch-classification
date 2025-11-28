import torch
from torchvision import transforms
from typing import Dict


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get training transforms with enhanced data augmentation for VGG16.
    Note: Assumes input is already a tensor in [C, H, W] format.
    VGG16-specific: stronger augmentation to combat overfitting tendencies.

    Args:
        img_size (int): Target image size for VGG16 (default: 224)

    Returns:
        transforms.Compose: Training transform pipeline
    """
    return transforms.Compose(
        [
            # Resize to target size
            transforms.Resize((img_size, img_size)),
            # Enhanced data augmentation suitable for histopathology and VGG16
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),  # Increased from 15°
            # Enhanced color augmentation for VGG16 robustness
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2  # Increased values
            ),
            # Random erasing to prevent overfitting (VGG16-specific)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.3)),
            # Normalize with ImageNet statistics (input is already tensor)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],  # ImageNet std
            ),
        ]
    )


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get validation transforms without data augmentation.
    Note: Assumes input is already a tensor in [C, H, W] format.

    Args:
        img_size (int): Target image size for VGG16 (default: 224)

    Returns:
        transforms.Compose: Validation transform pipeline
    """
    return transforms.Compose(
        [
            # Resize to target size
            transforms.Resize((img_size, img_size)),
            # Normalize with ImageNet statistics (input is already tensor)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],  # ImageNet std
            ),
        ]
    )


def get_test_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get test transforms (same as validation).

    Args:
        img_size (int): Target image size for VGG16 (default: 224)

    Returns:
        transforms.Compose: Test transform pipeline
    """
    return get_val_transforms(img_size)


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms for single image processing.

    Args:
        img_size (int): Target image size for VGG16 (default: 224)

    Returns:
        transforms.Compose: Inference transform pipeline
    """
    return get_val_transforms(img_size)


def get_minimal_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get minimal transforms for baseline training.
    Note: Assumes input is already a tensor in [C, H, W] format.

    Args:
        img_size (int): Target image size for VGG16 (default: 224)

    Returns:
        transforms.Compose: Minimal transform pipeline
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# ImageNet statistics for reference
IMAGENET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor using ImageNet statistics.

    Useful for visualization purposes.

    Args:
        tensor (torch.Tensor): Normalized tensor

    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor(IMAGENET_STATS["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STATS["std"]).view(1, 3, 1, 1)

    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()

    return tensor * std + mean


# Transform configurations for VGG16 training strategies
# Note: Only standard and minimal configurations per user preference
TRANSFORM_CONFIGS = {
    "standard": {
        "train": get_train_transforms(),
        "val": get_val_transforms(),
        "test": get_test_transforms(),
    },
    "minimal": {
        "train": get_minimal_transforms(),
        "val": get_val_transforms(),
        "test": get_test_transforms(),
    },
}


def get_transforms_by_config(
    config_name: str = "standard",
) -> Dict[str, transforms.Compose]:
    """
    Get transforms by configuration name.

    Args:
        config_name (str): Configuration name ('standard', 'minimal')

    Returns:
        Dict[str, transforms.Compose]: Dictionary of transforms

    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name not in TRANSFORM_CONFIGS:
        raise ValueError(
            f"Unknown transform config: {config_name}. "
            f"Available configs: {list(TRANSFORM_CONFIGS.keys())}"
        )

    return TRANSFORM_CONFIGS[config_name]