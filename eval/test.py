import os
import sys
import torch
import typer
import numpy as np

sys.path.append("..")
from model import PatchMIL
from dataloader import get_loaders, get_latest_checkpoint
from vis_utils import visualize_model_performance

latest_checkpoint = get_latest_checkpoint("../checkpoints")
app = typer.Typer()


@app.command()
def main(
    embed_folder: str = typer.Argument(
        ...,
        help="Input path of the embeddings generated from the patches.",
    ),
    embed_dim: int = typer.Option(
        ..., "--embed_dim", help="Embedding dimension of the foundation model."
    ),
    model_pth: str = typer.Option(
        latest_checkpoint,
        "--model_pth",
        help="Path of the model to be evaluated.",
    ),
    tiles_per_patch: int = typer.Option(
        256, "--tpp", help="The number of ViT tiles per image patch."
    ),
    device: str = typer.Option(
        "cpu", "--device", help="Device to inference the model on."
    ),
    num_classes: int = typer.Option(
        11, "--num_classes", help="Number of output classes."
    ),
    batch_size: int = typer.Option(
        256, "--batch_size", help="Batch size of training."
    ),
):
    model = PatchMIL(tiles_per_patch, embed_dim, num_classes)

    checkpoint = torch.load(model_pth)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _, val_loader = get_loaders(
        batch_size,
        embed_folder,
        "patch",
        embed_dim,
        num_classes,
        dtype=np.float16,
        test_size=0.2,
    )

    cm_fig, roc_fig = visualize_model_performance(
        model, val_loader, device, num_classes
    )
    os.makedirs("./figs", exist_ok=True)
    cm_fig.savefig("./figs/confusion_matrix.png")
    roc_fig.savefig("./figs/ROC_curve.png")


if __name__ == "__main__":
    app()
