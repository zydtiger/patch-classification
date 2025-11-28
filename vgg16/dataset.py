import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import convert_image_dtype
from pathlib import Path
import tifffile


class TIFDataset(Dataset):
    """Dataset for loading TIF images from NCT-CRC-HE-100K structure"""

    def __init__(self, data_root, num_classes=9, transform=None):
        self.data_root = Path(data_root)
        self.num_classes = num_classes
        self.transform = transform
        self.samples = []

        # Get all class folders
        class_folders = [f.name for f in self.data_root.iterdir() if f.is_dir()]

        # Collect all TIF files with their class paths
        for class_name in sorted(class_folders):
            class_path = self.data_root / class_name
            tif_files = list(class_path.glob("*.tif"))

            for tif_file in sorted(tif_files):
                self.samples.append(tif_file)

        self.class_to_idx = self._create_class_mapping()

    def _create_class_mapping(self) -> dict:
        """Create mapping from class names to indices."""
        class_names = set()
        for embedding_file in self.samples:
            # Extract class name from filename: {class_name}-{filename}.npy
            class_name = embedding_file.stem.split("-")[0]
            class_names.add(class_name)

        # Create sorted class to index mapping
        sorted_classes = sorted(class_names)
        return {class_name: idx for idx, class_name in enumerate(sorted_classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tif_path = self.samples[idx]

        # Load TIF image
        image = tifffile.imread(tif_path)
        image_tensor = convert_image_dtype(torch.from_numpy(image))
        image_tensor = image_tensor.permute(2, 0, 1)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Extract class label from filename
        class_name = tif_path.stem.split("-")[0]
        label_idx = self.class_to_idx[class_name]

        # Create one-hot encoding
        label_ohe = torch.zeros(self.num_classes, dtype=torch.float32)
        label_ohe[label_idx] = 1.0

        return image_tensor, label_ohe, label_idx