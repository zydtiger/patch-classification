import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import typer
from PIL import Image, ImageDraw
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

# Add the parent directory to the system path to allow for local module imports.
sys.path.append("..")
from utils import get_latest_checkpoint
from model import PatchMIL

# --- Constants and Configuration ---

# Disable the decompression bomb check to allow loading large images.
Image.MAX_IMAGE_PIXELS = None

# Define a colormap for visualizing the output classes.
# Each color corresponds to a class, with transparency (alpha=77).
CUSTOM_COLORMAP = [
    (70, 220, 219),      # Class 0 (islet)
    (80, 70, 219),       # Class 1 (duct)
    (70, 180, 81),       # Class 2 (smooth muscle)
    (219, 220, 80),      # Class 3 (fat)
    (131, 50, 191),      # Class 4 (acini)
    (240, 180, 231),     # Class 5 (stroma)
    (255, 255, 255),     # Class 6 (background)
    (209, 80, 75),       # Class 7 (PanIN)
    (160, 130, 40),      # Class 8 (nerve)
    (15, 80, 110),       # Class 9 (inflammation)
    (160, 30, 120),      # Class 10 (endothelium)
]

# --- Typer Application Setup ---
app = typer.Typer(
    help="WSI classification script using a Patch-based MIL model."
)


# --- Helper Functions ---


def read_region(
    wsi_data: np.ndarray, location: tuple[int, int], size: tuple[int, int]
) -> Image.Image:
    """
    Reads a specified region from a WSI numpy array.

    Args:
        wsi_data: The numpy array representing the WSI.
        location: A tuple (x, y) for the top-left corner of the region.
        size: A tuple (width, height) for the dimensions of the region.

    Returns:
        A PIL Image of the specified region, padded with white if the
        requested region goes out of bounds.
    """
    x, y = location
    width, height = size
    h, w = wsi_data.shape[:2]

    x_end = min(x + width, w)
    y_end = min(y + height, h)

    region_data = wsi_data[y:y_end, x:x_end]
    region_image = Image.fromarray(region_data)

    # Pad the image if the extracted region is smaller than the requested size.
    if region_image.size != size:
        padded_image = Image.new(region_image.mode, size, (255, 255, 255))
        padded_image.paste(region_image, (0, 0))
        return padded_image

    return region_image


# --- Main Command ---


@app.command()
def main(
    wsi_path: Path = typer.Argument(
        ...,
        help="Path to the input Whole Slide Image (WSI) file.",
        exists=True,
    ),
    mask_path: Path = typer.Option(
        ...,
        "--mask",
        help="Path to the corresponding tissue mask file.",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        "./output",
        "-o",
        "--output",
        help="Directory to save the final classified image.",
    ),
    patch_margin: int = typer.Option(
        112,
        "--patch_margin",
        help="The stride or step size between consecutive patches.",
    ),
    embed_dim: int = typer.Option(
        1280,
        "--embed_dim",
        help="Embedding dimension of the foundation model.",
    ),
    model_pth: str = typer.Option(
        str(get_latest_checkpoint(Path("../checkpoints"))),
        "--model_pth",
        help="Path to the trained classification model checkpoint.",
    ),
    tiles_per_patch: int = typer.Option(
        256,
        "--tiles_per_patch",
        "-tpp",
        help="Number of ViT tiles per image patch.",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="Device to run the model on ('cuda' or 'cpu').",
    ),
    num_classes: int = typer.Option(
        11, "--num_classes", help="Number of output classes for the model."
    ),
    batch_size: int = typer.Option(
        256,
        "--batch_size",
        help="Number of patches to process in a single batch.",
    ),
):
    """
    Processes a Whole Slide Image (WSI), classifies tissue patches using a
    pre-trained model, and generates a visualization of the results.
    """
    # --- Initialization ---
    print(f"Using device: {device}")
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Load Models ---
    print("Loading embedding model (Paige-AI/Virchow2)...", end="", flush=True)
    start_time = time.time()
    embed_model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.mlp.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    ).to(device)
    embed_model.eval()
    transforms = create_transform(
        **resolve_data_config(embed_model.pretrained_cfg, model=embed_model)
    )
    duration = time.time() - start_time
    print(f"\rEmbedding model (Paige-AI/Virchow2) loaded in {duration:.2f}s.\x1b[K")

    print(f"Loading classification model from: {model_pth}...", end="", flush=True)
    start_time = time.time()
    class_model = PatchMIL(tiles_per_patch, embed_dim, num_classes).to(device)
    checkpoint = torch.load(model_pth, map_location=device)
    class_model.load_state_dict(checkpoint["model_state_dict"])
    class_model.eval()
    duration = time.time() - start_time
    print(f"\rClassification model from {model_pth} loaded in {duration:.2f}s.\x1b[K")

    # --- Data Loading ---
    print(f"Processing WSI: {wsi_path}...", end="", flush=True)
    start_time = time.time()
    wsi_data = np.array(Image.open(wsi_path))
    mask_data = np.array(Image.open(mask_path))
    duration = time.time() - start_time
    print(f"\rWSI and mask processed in {duration:.2f}s.\x1b[K")

    wsi_h, wsi_w = wsi_data.shape[:2]
    mask_h, mask_w = mask_data.shape

    # --- Coordinate Generation ---
    print("Generating patch coordinates based on the tissue mask...", end="", flush=True)
    start_time = time.time()
    patch_size = (224, 224)
    downsample_factor = wsi_w // mask_w
    coords = []

    for y in range(0, wsi_h, patch_margin):
        for x in range(0, wsi_w, patch_margin):
            # Map the patch's center to the mask's coordinate system.
            mask_x = (x + patch_margin // 2) // downsample_factor
            mask_y = (y + patch_margin // 2) // downsample_factor

            # Check if the coordinate is within the mask bounds.
            if 0 <= mask_y < mask_h and 0 <= mask_x < mask_w:
                # If the mask is non-zero (i.e., contains tissue), add coordinates.
                if mask_data[mask_y, mask_x]:
                    coords.append((x, y))
    duration = time.time() - start_time
    print(f"\rPatch coordinates generated in {duration:.2f}s.\x1b[K")
    print(f"Found {len(coords)} candidate patches to classify.")

    # --- Patch Classification ---
    overlay = Image.new("RGBA", (wsi_w, wsi_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    batch_coords = []
    batch_patches = []

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.float16
    ):
        for i in tqdm(range(len(coords)), desc="Classifying patches"):
            x, y = coords[i]
            patch = read_region(wsi_data, (x, y), patch_size)
            batch_coords.append((x, y))
            batch_patches.append(patch)

            # Process a batch when it's full or it's the last patch.
            if len(batch_patches) == batch_size or i == len(coords) - 1:
                tensor_patches = torch.stack([transforms(p) for p in batch_patches]).to(device)  # type: ignore

                embeddings = embed_model(tensor_patches)
                patch_tokens = embeddings[:, 5:].clone()

                predictions = class_model(patch_tokens)
                predicted_classes = torch.argmax(predictions, dim=1)

                # Draw classified patches onto the overlay.
                for j, (px, py) in enumerate(batch_coords):
                    pred_class = int(predicted_classes[j].item())
                    color = CUSTOM_COLORMAP[pred_class]

                    # Offset to center the colored region over the patch margin area.
                    offset = (patch_size[0] - patch_margin) // 2
                    draw.rectangle(
                        [
                            px + offset,
                            py + offset,
                            px + offset + patch_margin,
                            py + offset + patch_margin,
                        ],
                        fill=color,
                    )

                batch_coords.clear()
                batch_patches.clear()

    # --- Final Image Creation ---
    print("Classification complete. Merging overlay with base image...", end="", flush=True)
    start_time = time.time()
    # Ensure WSI is in RGBA for compositing.
    wsi_thumb = Image.fromarray(wsi_data).convert("RGBA")
    final_image = Image.alpha_composite(wsi_thumb, overlay)
    duration = time.time() - start_time
    print(f"\rOverlay merged with base image in {duration:.2f}s.\x1b[K")

    output_path = output_dir / f"{wsi_path.stem}_classified.tif"
    print(f"Saving final image to: {output_path}...", end="", flush=True)
    start_time = time.time()
    final_image.save(output_path)
    duration = time.time() - start_time
    print(f"\rFinal image saved to {output_path} in {duration:.2f}s.\x1b[K")

    print("Done.")


if __name__ == "__main__":
    app()
