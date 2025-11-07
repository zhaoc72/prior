"""Common dataset utilities for Gaussian category priors."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class GCPSamples(Dataset):
    """Dataset wrapper around precomputed index entries."""

    def __init__(self, index_file: str, categories: list[str] | None = None) -> None:
        data = np.load(index_file, allow_pickle=True).item()
        items = data["items"]
        if categories is not None:
            items = [item for item in items if item["cat"] in categories]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        mask_path = Path(item["mask_npy"])
        if mask_path.suffix.lower() == ".npy":
            mask_array = np.load(mask_path).astype(np.float32)
        else:
            image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask_array = (image > 127).astype(np.float32)
        occ = np.load(item["occ_npz"])
        sample = {
            "mask": torch.from_numpy(mask_array[None]),
            "K": torch.tensor(np.array(item["K"], dtype=np.float32)),
            "Rt": torch.tensor(np.array(item["Rt"], dtype=np.float32)),
            "occ_pts": torch.tensor(occ["xyz"].astype(np.float32)),
            "occ_lbl": torch.tensor(occ["lbl"].astype(np.float32)),
            "cat": item["cat"],
            "cat_id": torch.tensor(int(item["cat_id"]), dtype=torch.long),
        }
        surface = occ["surface"].astype(np.float32) if "surface" in occ else np.zeros((0, 3), dtype=np.float32)
        sample["surface_pts"] = torch.tensor(surface)
        return sample

__all__ = ["GCPSamples"]
