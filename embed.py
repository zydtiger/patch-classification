import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import convert_image_dtype
import tifffile
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from pathlib import Path
from utils import get_best_device

# Global configuration
BATCH_SIZE = 4
DEVICE = get_best_device()
DATA_ROOT = Path("data/NCT-CRC-HE-100K")
OUTPUT_DIR = Path("data/NCT-CRC-HE-100K-embeddings")


class TIFDataset(Dataset):
    """Dataset for loading TIF images from NCT-CRC-HE-100K structure"""

    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
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

        return image_tensor


def load_model():
    """Load Virchow2 model for embedding generation"""
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.mlp.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )

    # Create transforms
    transforms = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    model = model.to(DEVICE)
    model.eval()

    return model, transforms


def main():
    """Process all TIF images and save embeddings incrementally"""

    print(f"Processing dataset from: {DATA_ROOT}")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")

    # Load model and transforms
    model, transforms = load_model()

    # Create dataset
    dataset = TIFDataset(DATA_ROOT, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Found {len(dataset)} images to process")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize incremental file storage
    processed_count = 0

    # Check if we're resuming from a previous run by scanning existing files
    existing_files = set()
    existing_files = {f.stem for f in OUTPUT_DIR.glob("*.npy")}
    print(f"Found {len(existing_files)} existing embeddings")

    # Process images in batches
    with torch.inference_mode(), torch.autocast(
        device_type=DEVICE, dtype=torch.float16
    ):
        for batch_idx, images in enumerate(
            tqdm(dataloader, desc="Generating embeddings")
        ):
            # Move images to device
            images = images.to(DEVICE, non_blocking=True)

            # Generate embeddings
            outputs = model(images)

            # Convert to numpy and move to CPU
            embeddings = outputs.cpu().numpy()  # Shape: (batch_size, 261, 1280)

            # Store each embedding with its corresponding filename key
            for i in range(embeddings.shape[0]):
                sample_idx = batch_idx * BATCH_SIZE + i
                if sample_idx < len(dataset):
                    tif_path = dataset.samples[sample_idx]
                    filename = tif_path.stem  # Use filename without extension

                    # Skip if already processed
                    if filename in existing_files:
                        continue

                    # Save individual embedding file (no memory accumulation)
                    embedding_path = OUTPUT_DIR / f"{filename}.npy"
                    np.save(embedding_path, embeddings[i])
                    processed_count += 1

            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {processed_count} embeddings so far...")

    # Final summary
    total_embeddings = len(list(OUTPUT_DIR.glob("*.npy")))

    print(f"Processing complete! Embeddings saved to {OUTPUT_DIR}")
    print(f"Total embeddings in directory: {total_embeddings}")
    print(f"New embeddings processed: {processed_count}")
    if total_embeddings > 0:
        sample_embedding = np.load(next(OUTPUT_DIR.glob("*.npy")))
        print(f"Embedding shape per sample: {sample_embedding.shape}")


if __name__ == "__main__":
    main()
