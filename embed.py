import os
import glob
import numpy as np
import typer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform

from patch_loader import HDF5PatchDataset

device = "cuda"
app = typer.Typer()


@app.command()
def main(
    patch_folder: str = typer.Argument(
        ..., help="Path to the folder containing all the input patch files."
    ),
    embed_folder: str = typer.Argument(
        ..., help="Output path of the embeddings generated from the patches."
    ),
    start_index: int = typer.Option(
        0,
        "--start-index",
        help="Starting index of the patch file to start processing.",
    ),
):
    # Load processor and model
    batch_size = 256  # Adjust based on your memory
    np_dtype = np.float16
    pt_dtype = torch.float16
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.mlp.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    transforms = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model = model.to(device)
    model.eval()

    patch_files = sorted(glob.glob(os.path.join(patch_folder, "*.mat")))
    if not patch_files:
        raise FileNotFoundError(f"No .mat files found in {patch_folder}")

    for patch_file in patch_files:
        # Extract the index from the patch file name (e.g., "patches_0000.mat" -> "0000")
        patch_index = int(
            os.path.basename(patch_file).split("_")[1].split(".")[0]
        )
        if patch_index < start_index:
            continue

        patch_dataset = HDF5PatchDataset(patch_file)
        dataloader = DataLoader(
            patch_dataset,
            batch_size=batch_size,
            num_workers=12,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
        )

        patch_parent = os.path.basename(os.path.dirname(patch_file))
        output_dir = os.path.join(embed_folder, patch_parent)
        os.makedirs(output_dir, exist_ok=True)
        patch_suffix = os.path.splitext(patch_file)[0].split("_")[1]
        output_base = os.path.join(output_dir, f"embeddings_{patch_suffix}")

        # Define memmap file paths
        cls_path = f"{output_base}_cls.npy"
        patch_path = f"{output_base}_patch.npy"
        label_path = f"{output_base}_labels.npy"

        cls_memmap = None
        patch_memmap = None
        label_memmap = None
        processed_count = 0

        try:
            with torch.inference_mode(), torch.autocast(
                device_type=device, dtype=pt_dtype
            ):
                # First batch to determine dimensions
                dataloader_iter = iter(dataloader)
                first_batch = next(dataloader_iter)
                patches, labels = first_batch
                patches = patches.to(device, non_blocking=True)
                patches = transforms(patches)  # type: ignore
                outputs = model(patches).to(pt_dtype)
                print(outputs.dtype, outputs.shape)

                # Get CLS token embeddings (pooler_output)
                cls_emb = outputs[:, 0].cpu().numpy()
                # Get patch token embeddings (last_hidden_state[5:])
                patch_tokens = outputs[:, 5:].cpu().numpy()
                labels = labels.numpy()

                # Get dimensions
                cls_emb_dim = cls_emb.shape[1]
                patch_tokens_shape = patch_tokens.shape[
                    1:
                ]  # (num_patches, hidden_dim)

                # Create memory-mapped files
                cls_memmap = np.memmap(
                    cls_path,
                    dtype=np_dtype,
                    mode="w+",
                    shape=(len(patch_dataset), cls_emb_dim),
                )

                patch_memmap = np.memmap(
                    patch_path,
                    dtype=np_dtype,
                    mode="w+",
                    shape=(len(patch_dataset), *patch_tokens_shape),
                )

                label_memmap = np.memmap(
                    label_path,
                    dtype="int8",
                    mode="w+",
                    shape=(len(patch_dataset), 1),
                )

                # Save first batch
                cls_memmap[0 : cls_emb.shape[0]] = cls_emb
                patch_memmap[0 : cls_emb.shape[0]] = patch_tokens
                label_memmap[0 : cls_emb.shape[0]] = labels.reshape(-1, 1)

                processed_count = cls_emb.shape[0]

                # Process remaining batches
                for batch_data in tqdm(
                    dataloader_iter,
                    desc=f"Processing {os.path.basename(patch_file)}",
                    initial=1,
                ):
                    patches, labels = batch_data
                    patches = patches.to(device, non_blocking=True)
                    patches = transforms(patches)  # type: ignore
                    outputs = model(patches).to(pt_dtype)

                    cls_emb = outputs[:, 0].cpu().numpy()
                    patch_tokens = outputs[:, 5:].cpu().numpy()
                    labels = labels.numpy()

                    # Save CLS embeddings
                    cls_memmap[
                        processed_count : processed_count + cls_emb.shape[0]
                    ] = cls_emb

                    # Save patch token embeddings
                    patch_memmap[
                        processed_count : processed_count + cls_emb.shape[0]
                    ] = patch_tokens

                    # Save labels
                    label_memmap[
                        processed_count : processed_count + cls_emb.shape[0]
                    ] = labels.reshape(-1, 1)

                    processed_count += cls_emb.shape[0]

            print(f"Embeddings and labels saved to {output_base}_*.npy files")
            print(f"CLS Embeddings shape: {cls_memmap.shape}")
            print(f"Patch Tokens shape: {patch_memmap.shape}")
            print(f"Labels shape: {label_memmap.shape}")
            print()

        finally:
            # Flush changes to disk
            if cls_memmap is not None:
                cls_memmap.flush()
            if patch_memmap is not None:
                patch_memmap.flush()
            if label_memmap is not None:
                label_memmap.flush()


if __name__ == "__main__":
    app()
