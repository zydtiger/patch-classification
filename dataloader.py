import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Literal


class NPYEmbeddingDataset(Dataset):
    def __init__(
        self,
        npy_base_pth,
        dataset: Literal["cls"] | Literal["patch"],
        embed_dim,
        num_classes,
        dtype,
    ):
        self.dataset = dataset
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dtype = dtype

        self.cls_path = f"{npy_base_pth}_cls.npy"
        self.patch_path = f"{npy_base_pth}_patch.npy"
        self.label_path = f"{npy_base_pth}_labels.npy"

        # Calculate length from file size
        self.length = os.path.getsize(self.label_path) // (
            np.dtype(np.int8).itemsize
        )

        self.cls_mmap = None
        self.patch_mmap = None
        self.label_mmap = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset == "cls" and self.cls_mmap is None:
            self.cls_mmap = np.memmap(
                self.cls_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.length, self.embed_dim),
            )
        if self.dataset == "patch" and self.patch_mmap is None:
            patch_dset_size = os.path.getsize(self.patch_path) // (
                np.dtype(self.dtype).itemsize
            )
            self.patch_mmap = np.memmap(
                self.patch_path,
                dtype=self.dtype,
                mode="r",
                shape=(
                    self.length,
                    patch_dset_size // self.embed_dim // self.length,
                    self.embed_dim,
                ),
            )
        if self.label_mmap is None:
            self.label_mmap = np.memmap(
                self.label_path, dtype="int8", mode="r", shape=(self.length, 1)
            )

        if self.dataset == "cls":
            assert self.cls_mmap is not None
            embeddings = torch.from_numpy(self.cls_mmap[idx].copy()).to(
                torch.float32
            )
        else:
            assert self.patch_mmap is not None
            embeddings = torch.from_numpy(self.patch_mmap[idx].copy()).to(
                torch.float32
            )

        # Convert from 1-11 to 0-10
        label = self.label_mmap[idx][0] - 1
        label_ohe = torch.zeros(self.num_classes)
        label_ohe[label] = 1
        return embeddings, label_ohe, label

    def __del__(self):
        # Close the memory-mapped files by removing the reference
        if self.patch_mmap is not None:
            self.patch_mmap = None
        if self.label_mmap is not None:
            self.label_mmap = None


def get_batch_weights(labels: torch.Tensor, num_classes):
    """Calculate weights for a specific batch of labels"""
    labels = labels.flatten().int()
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_counts = class_counts + 1e-6
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    return weights[labels]


def get_loaders(
    batch_size,
    embed_folder,
    dataset: Literal["cls"] | Literal["patch"],
    embed_dim,
    num_classes,
    dtype,
    test_size=0.2,
):
    label_files = glob.glob(
        os.path.join(embed_folder, "embeddings_*_labels.npy")
    )
    indices = sorted(
        [int(f.split("embeddings_")[1].split("_")[0]) for f in label_files]
    )

    if not indices:
        raise FileNotFoundError(f"No embedding files found in {embed_folder}")

    # Calculate split point based on test_size
    split_idx = int(len(indices) * test_size)
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]

    # Create train datasets
    train_datasets = [
        NPYEmbeddingDataset(
            f"{embed_folder}/embeddings_{i:04d}",
            dataset,
            embed_dim,
            num_classes,
            dtype,
        )
        for i in train_indices
    ]
    train_combined = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_combined,
        batch_size=batch_size,
        num_workers=12,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # Create validation datasets
    val_datasets = [
        NPYEmbeddingDataset(
            f"{embed_folder}/embeddings_{i:04d}",
            dataset,
            embed_dim,
            num_classes,
            dtype,
        )
        for i in val_indices
    ]
    val_combined = ConcatDataset(val_datasets)
    val_loader = DataLoader(
        val_combined,
        batch_size=batch_size,
        num_workers=12,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=2,
        drop_last=True,
    )

    return train_loader, val_loader


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint file based on epoch number."""
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Extract epoch numbers and find the latest one
    latest_checkpoint = max(
        checkpoint_files,
        key=lambda x: int(x.split("_epoch_")[-1].replace(".pth", "")),
    )
    return latest_checkpoint
