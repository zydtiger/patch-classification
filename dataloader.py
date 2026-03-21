from pathlib import Path
from typing import Literal, Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Dataset for loading individual embedding files with corresponding labels.

    Each embedding file contains a 261x1280 array where:
    - Row 0 contains CLS embedding
    - Rows 5-256 contain patch embeddings (251 patches)
    Labels are extracted from the filename format: {class_name}-{filename}.npy
    """

    def __init__(
        self,
        dataset_type: Literal["cls", "patch"],
        embed_dim: int,
        num_classes: int,
        embed_dir: Path,
        class_to_idx: Optional[dict[str, int]] = None,
    ):
        """
        Initialize the embedding dataset.

        Args:
            dataset_type: Type of dataset ("cls" or "patch")
            embed_dim: Dimension of embeddings
            num_classes: Number of classes for one-hot encoding
            embed_dir: Directory containing embedding files
            class_to_idx: Optional precomputed class mapping to reuse
        """
        self.dataset_type = dataset_type
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.embed_dir = embed_dir

        self.samples = self._find_embedding_files()
        if class_to_idx is None:
            self.class_to_idx = self._create_class_mapping()
        else:
            self.class_to_idx = class_to_idx
            self._validate_class_mapping()

        print(
            f"Found {len(self.samples)} embedding files for {len(self.class_to_idx)} classes in {embed_dir}"
        )

    def _find_embedding_files(self) -> List[Path]:
        """Find all .npy embedding files in embed_dir."""
        if not self.embed_dir.exists():
            raise FileNotFoundError(f"Embedding directory not found: {self.embed_dir}")

        embedding_files = list(self.embed_dir.glob("*.npy"))

        if not embedding_files:
            raise FileNotFoundError(f"No .npy files found in {self.embed_dir}")

        return sorted(embedding_files)

    def _create_class_mapping(self) -> dict:
        """Create mapping from class names to indices."""
        class_names = set()
        for embedding_file in self.samples:
            class_names.add(self._extract_class_name(embedding_file))

        # Create sorted class to index mapping
        sorted_classes = sorted(class_names)
        return {class_name: idx for idx, class_name in enumerate(sorted_classes)}

    def _validate_class_mapping(self) -> None:
        """Ensure all sample labels exist in the provided class mapping."""
        sample_classes = {self._extract_class_name(embedding_file) for embedding_file in self.samples}
        unknown_classes = sorted(sample_classes - set(self.class_to_idx))

        if unknown_classes:
            raise ValueError(
                "Found classes in the embedding directory that are missing from the "
                f"provided class mapping: {unknown_classes}"
            )

    @staticmethod
    def _extract_class_name(embedding_file: Path) -> str:
        """Extract the class name from an embedding filename."""
        return embedding_file.stem.split("-")[0]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (embeddings, one_hot_label, label_index)
        """
        embedding_file = self.samples[idx]

        # Load embedding data (261 x 1280)
        embedding_data = np.load(embedding_file)

        # Extract embeddings based on dataset type
        if self.dataset_type == "cls":
            # Use CLS token (first row)
            embedding = embedding_data[0, :]  # Shape: (1280,)
        else:  # patch
            # Use patch embeddings (rows 5-255, inclusive)
            embedding = embedding_data[5:, :]  # Shape: (256, 1280)

        # Convert to tensor
        embedding_tensor = torch.from_numpy(embedding).float()

        # Extract class label from filename
        class_name = self._extract_class_name(embedding_file)
        label_idx = self.class_to_idx[class_name]

        # Create one-hot encoding
        label_ohe = torch.zeros(self.num_classes, dtype=torch.float32)
        label_ohe[label_idx] = 1.0

        return embedding_tensor, label_ohe, label_idx
