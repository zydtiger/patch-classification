import h5py
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import convert_image_dtype


class HDF5PatchDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Open once to get the number of patches (X)
        with h5py.File(h5_path, "r") as f:
            # MATLAB format: (3, h, w, X)
            self.length = f["patches"].shape[3]  # type: ignore

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy open per worker
        if not hasattr(self, "h5_file"):
            self.h5_file = h5py.File(self.h5_path, "r")
        patch = self.h5_file["patches"][..., idx]  # type: ignore (3, h, w)
        patch = torch.from_numpy(patch)
        patch = patch.contiguous()
        patch = convert_image_dtype(patch, torch.float32)
        label = self.h5_file["labels"][..., idx]  # type: ignore (1)
        return patch, label

    def __del__(self):
        if hasattr(self, "h5_file"):
            self.h5_file.close()
