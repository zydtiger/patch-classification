import torch
from torchvision import transforms
from typing import Dict


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get training transforms with data augmentation for medical images.
    Note: Assumes input is already a tensor in [C, H, W] format.

    Args:
        img_size (int): Target image size for ResNet50 (default: 224)

    Returns:
        transforms.Compose: Training transform pipeline
    """
    return transforms.Compose(
        [
            # Resize to target size
            transforms.Resize((img_size, img_size)),
            # Data augmentation suitable for histopathology
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # Color augmentation (conservative for medical images)
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
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
        img_size (int): Target image size for ResNet50 (default: 224)

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
        img_size (int): Target image size for ResNet50 (default: 224)

    Returns:
        transforms.Compose: Test transform pipeline
    """
    return get_val_transforms(img_size)


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get inference transforms for single image processing.

    Args:
        img_size (int): Target image size for ResNet50 (default: 224)

    Returns:
        transforms.Compose: Inference transform pipeline
    """
    return get_val_transforms(img_size)


# Advanced augmentation options for research
def get_strong_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Get strong training transforms with additional augmentations.
    Note: Assumes input is already a tensor in [C, H, W] format.

    Use this for experiments requiring more aggressive augmentation.

    Args:
        img_size (int): Target image size for ResNet50 (default: 224)

    Returns:
        transforms.Compose: Strong training transform pipeline
    """
    return transforms.Compose(
        [
            # Random resized crop for more variation
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            # Color augmentations
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
            ),
            # Random erasing for robustness
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
            # Normalize with ImageNet statistics (input is already tensor)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_transform_dict(
    img_size: int = 224, strong_aug: bool = False
) -> Dict[str, transforms.Compose]:
    """
    Get dictionary of transforms for different phases.

    Args:
        img_size (int): Target image size for ResNet50 (default: 224)
        strong_aug (bool): Whether to use strong augmentation for training

    Returns:
        Dict[str, transforms.Compose]: Dictionary containing transforms
    """
    if strong_aug:
        train_transform = get_strong_train_transforms(img_size)
    else:
        train_transform = get_train_transforms(img_size)

    return {
        "train": train_transform,
        "val": get_val_transforms(img_size),
        "test": get_test_transforms(img_size),
        "inference": get_inference_transforms(img_size),
    }


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


# Transform configurations for different training strategies
TRANSFORM_CONFIGS = {
    "standard": {
        "train": get_train_transforms(),
        "val": get_val_transforms(),
        "test": get_test_transforms(),
    },
    "strong": {
        "train": get_strong_train_transforms(),
        "val": get_val_transforms(),
        "test": get_test_transforms(),
    },
    "minimal": {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
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
        config_name (str): Configuration name ('standard', 'strong', 'minimal')

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
