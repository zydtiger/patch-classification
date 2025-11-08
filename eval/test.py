import sys
from pathlib import Path
from typing import Literal

import torch
import typer

# Add parent directory to path for local imports
sys.path.append("..")
from model import MulticlassMLP, PatchMIL
from utils import create_data_loaders, get_latest_checkpoint
from vis_utils import visualize_model_performance

app = typer.Typer()


@app.command()
def main(
    embed_dir: Path = typer.Option(
        ...,
        help="Input path of the embeddings generated from the patches.",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    dataset_type: Literal["cls", "patch"] = typer.Option(
        "patch",
        help="Type of dataset: 'cls' for CLS token classification, 'patch' for patch-based MIL.",
    ),
    embed_dim: int = typer.Option(
        1280, help="Embedding dimension of the foundation model."
    ),
    model_pth: Path = typer.Option(
        None,
        help="Path of the model to be evaluated.",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    tiles_per_patch: int = typer.Option(
        256, "--tpp", help="The number of ViT tiles per image patch."
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device to inference the model on.",
    ),
    num_classes: int = typer.Option(9, help="Number of output classes."),
    batch_size: int = typer.Option(256, help="Batch size for evaluation."),
):
    """Evaluate model performance and generate confusion matrix and ROC curves."""
    print(f"Evaluating model with dataset type: {dataset_type}")

    # Validate dataset type
    if dataset_type not in ["cls", "patch"]:
        raise ValueError("dataset_type must be either 'cls' or 'patch'")

    # Set default model path if not provided
    if model_pth is None:
        model_pth = get_latest_checkpoint(Path("../checkpoints"))

    # Initialize model based on dataset type
    if dataset_type == "cls":
        print(f"Creating CLS token classification model (MulticlassMLP)")
        model = MulticlassMLP(embed_dim, num_classes)
    else:  # patch
        print(f"Creating patch-based MIL model (PatchMIL)")
        model = PatchMIL(tiles_per_patch, embed_dim, num_classes)

    # Load checkpoint with proper error handling
    checkpoint = torch.load(model_pth, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get data loaders using the new function signature
    _, val_loader = create_data_loaders(
        batch_size=batch_size,
        dataset_type=dataset_type,
        embed_dim=embed_dim,
        num_classes=num_classes,
        test_size=0.2,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=(device != "cpu"),
        persistent_workers=True,
        drop_last=False,
        embed_dir=embed_dir,
    )

    # Generate visualizations
    cm_fig, roc_fig = visualize_model_performance(
        model, val_loader, device, num_classes
    )

    # Save figures
    figs_dir = Path("./figs")
    figs_dir.mkdir(exist_ok=True)
    cm_fig.savefig(figs_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    roc_fig.savefig(figs_dir / "ROC_curve.png", dpi=300, bbox_inches="tight")

    print(f"Figures saved to {figs_dir.absolute()}")


if __name__ == "__main__":
    app()
