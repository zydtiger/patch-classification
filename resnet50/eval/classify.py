import torch
import torch.nn as nn
import numpy as np
import tifffile
import typer
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
from PIL import Image
import sys

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from model import ResNet50Classifier, create_model
from transforms import get_inference_transforms
from utils import get_best_device

app = typer.Typer()


class WholeSlideClassifier:
    """
    Classifier for whole slide images using sliding window approach.
    """

    def __init__(
        self,
        model_path: Path,
        patch_size: int = 224,
        stride: int = 112,
        batch_size: int = 32,
        device: str = "auto",
        num_classes: int = 9
    ):
        """
        Initialize whole slide classifier.

        Args:
            model_path: Path to trained model checkpoint
            patch_size: Size of patches to extract (should match model input)
            stride: Stride for sliding window (overlap = patch_size - stride)
            batch_size: Batch size for inference
            device: Device to run inference on
            num_classes: Number of output classes
        """
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Set device
        if device == "auto":
            self.device = get_best_device()
        else:
            self.device = device

        # Load model
        self.model = create_model(num_classes=num_classes, feature_extract=False)
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Setup transforms
        self.transforms = get_inference_transforms(patch_size)

        print(f"Whole slide classifier initialized")
        print(f"Device: {self.device}")
        print(f"Patch size: {patch_size}, Stride: {stride}")

    def load_model(self, checkpoint_path: Path):
        """Load model weights from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            print(f"✅ Model loaded from {checkpoint_path}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def extract_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches from image using sliding window.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Tuple of (patches, coordinates)
        """
        height, width = image.shape[:2]
        patches = []
        coordinates = []

        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                coordinates.append((x, y))

        return patches, coordinates

    def classify_patches(self, patches: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify a batch of patches.

        Args:
            patches: List of patches as numpy arrays

        Returns:
            Tuple of (predictions, probabilities)
        """
        predictions = []
        all_probabilities = []

        # Process patches in batches
        for i in range(0, len(patches), self.batch_size):
            batch_patches = patches[i:i + self.batch_size]
            batch_tensors = []

            # Convert patches to tensors and apply transforms
            for patch in batch_patches:
                # Convert to PIL Image for transforms
                if patch.dtype != np.uint8:
                    patch = (patch * 255).astype(np.uint8)

                pil_image = Image.fromarray(patch)
                tensor = self.transforms(pil_image)
                batch_tensors.append(tensor)

            # Stack into batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                batch_predictions = torch.argmax(outputs, dim=1)

                predictions.extend(batch_predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return np.array(predictions), np.array(all_probabilities)

    def classify_image(self, image_path: Path) -> dict:
        """
        Classify a whole slide image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing classification results
        """
        print(f"Processing image: {image_path}")

        # Load image
        try:
            if image_path.suffix.lower() in ['.tif', '.tiff']:
                image = tifffile.imread(image_path)
            else:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] > 3:  # More than 3 channels
                image = image[:, :, :3]

        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return {}

        print(f"Image shape: {image.shape}")

        # Extract patches
        patches, coordinates = self.extract_patches(image)
        print(f"Extracted {len(patches)} patches")

        # Classify patches
        predictions, probabilities = self.classify_patches(patches)

        # Create result map
        height, width = image.shape[:2]
        result_map = np.zeros((height, width, self.num_classes), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.int32)

        # Fill result map
        for (x, y), probs in zip(coordinates, probabilities):
            x_end = min(x + self.patch_size, width)
            y_end = min(y + self.patch_size, height)

            result_map[y:y_end, x:x_end] += probs
            count_map[y:y_end, x:x_end] += 1

        # Average the overlapping regions
        mask = count_map > 0
        for i in range(self.num_classes):
            result_map[:, :, i][mask] /= count_map[mask]

        # Get overall statistics
        class_counts = np.bincount(predictions, minlength=self.num_classes)
        class_percentages = class_counts / len(predictions) * 100

        # Get most confident patch
        max_confidence_idx = np.argmax(np.max(probabilities, axis=1))
        most_confident_patch = coordinates[max_confidence_idx]
        most_confident_class = predictions[max_confidence_idx]
        most_confident_prob = np.max(probabilities[max_confidence_idx])

        return {
            'image_path': str(image_path),
            'image_shape': image.shape,
            'num_patches': len(patches),
            'predictions': predictions,
            'probabilities': probabilities,
            'coordinates': coordinates,
            'result_map': result_map,
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'most_confident_patch': most_confident_patch,
            'most_confident_class': most_confident_class,
            'most_confident_prob': most_confident_prob
        }


def create_visualization_overlay(
    image: np.ndarray,
    result_map: np.ndarray,
    class_names: List[str],
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create visualization overlay of classification results.

    Args:
        image: Original image
        result_map: Classification probability map (H, W, num_classes)
        class_names: List of class names
        alpha: Transparency for overlay

    Returns:
        Visualization image with overlay
    """
    height, width = image.shape[:2]

    # Get argmax class for each pixel
    class_map = np.argmax(result_map, axis=2)
    confidence_map = np.max(result_map, axis=2)

    # Create color map for classes
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 128],  # Purple
        [255, 165, 0],  # Orange
        [128, 128, 128] # Gray
    ]

    # Create overlay
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(len(class_names)):
        mask = class_map == i
        for c in range(3):
            overlay[:, :, c][mask] = colors[i][c]

    # Blend with original image
    overlay = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)

    # Add confidence as brightness
    overlay = cv2.convertScaleAbs(overlay, alpha=confidence_map)

    return overlay


@app.command()
def classify_single(
    model_path: Path = typer.Option(..., help="Path to trained model checkpoint"),
    image_path: Path = typer.Option(..., help="Path to image to classify"),
    output_dir: Path = typer.Option("classification_results", help="Output directory"),
    patch_size: int = typer.Option(224, help="Patch size for sliding window"),
    stride: int = typer.Option(112, help="Stride for sliding window"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    device: str = typer.Option("auto", help="Device to use"),
    save_overlay: bool = typer.Option(True, help="Save visualization overlay"),
):
    """
    Classify a single image using the trained ResNet50 model.
    """
    # Initialize classifier
    classifier = WholeSlideClassifier(
        model_path=model_path,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        device=device
    )

    # Classify image
    results = classifier.classify_image(image_path)

    if not results:
        print("❌ Classification failed")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print results
    print(f"\n=== Classification Results ===")
    print(f"Image: {image_path.name}")
    print(f"Total patches analyzed: {results['num_patches']}")

    # Class names for tissue classification
    class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    print(f"\nClass distribution:")
    for i, (class_name, count, percentage) in enumerate(
        zip(class_names, results['class_counts'], results['class_percentages'])
    ):
        print(f"{class_name:4s}: {count:6d} patches ({percentage:5.1f}%)")

    print(f"\nMost confident prediction:")
    print(f"Class: {class_names[results['most_confident_class']]} "
          f"(confidence: {results['most_confident_prob']:.3f})")
    print(f"Patch location: {results['most_confident_patch']}")

    # Save results
    try:
        # Save classification report
        report_path = output_dir / f"{image_path.stem}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Image Classification Report\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Model: {model_path.name}\n")
            f.write(f"Total patches: {results['num_patches']}\n\n")

            f.write("Class distribution:\n")
            for i, (class_name, count, percentage) in enumerate(
                zip(class_names, results['class_counts'], results['class_percentages'])
            ):
                f.write(f"{class_name:4s}: {count:6d} patches ({percentage:5.1f}%)\n")

            f.write(f"\nMost confident prediction:\n")
            f.write(f"Class: {class_names[results['most_confident_class']]}\n")
            f.write(f"Confidence: {results['most_confident_prob']:.3f}\n")
            f.write(f"Patch location: {results['most_confident_patch']}\n")

        print(f"📄 Report saved to: {report_path}")

        # Save probability maps
        prob_path = output_dir / f"{image_path.stem}_probabilities.npz"
        np.savez(
            prob_path,
            result_map=results['result_map'],
            predictions=results['predictions'],
            coordinates=results['coordinates'],
            class_names=class_names
        )
        print(f"💾 Probability maps saved to: {prob_path}")

        # Save visualization overlay
        if save_overlay:
            try:
                # Load original image
                if image_path.suffix.lower() in ['.tif', '.tiff']:
                    original_image = tifffile.imread(image_path)
                else:
                    original_image = cv2.imread(str(image_path))
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # Create overlay
                overlay = create_visualization_overlay(
                    original_image,
                    results['result_map'],
                    class_names
                )

                # Save overlay
                overlay_path = output_dir / f"{image_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"🖼️  Overlay saved to: {overlay_path}")

            except Exception as e:
                print(f"⚠️  Could not save overlay: {e}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")

    print(f"\n✅ Classification completed! Results saved to: {output_dir}")


@app.command()
def batch_classify(
    model_path: Path = typer.Option(..., help="Path to trained model checkpoint"),
    input_dir: Path = typer.Option(..., help="Directory containing images to classify"),
    output_dir: Path = typer.Option("batch_classification", help="Output directory"),
    file_pattern: str = typer.Option("*.tif", help="File pattern to match"),
    patch_size: int = typer.Option(224, help="Patch size for sliding window"),
    stride: int = typer.Option(112, help="Stride for sliding window"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    device: str = typer.Option("auto", help="Device to use"),
):
    """
    Classify multiple images in a directory.
    """
    # Initialize classifier once
    classifier = WholeSlideClassifier(
        model_path=model_path,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
        device=device
    )

    # Find images
    image_paths = list(input_dir.glob(file_pattern))
    if not image_paths:
        print(f"No images found in {input_dir} with pattern {file_pattern}")
        return

    print(f"Found {len(image_paths)} images to classify")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for image_path in image_paths:
        print(f"\n{'='*60}")
        classify_single(
            model_path=model_path,
            image_path=image_path,
            output_dir=output_dir,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            save_overlay=True
        )

    print(f"\n✅ Batch classification completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    app()