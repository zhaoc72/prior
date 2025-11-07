"""Camera helper structures."""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class Camera:
    """Simple pinhole camera representation."""

    intrinsics: torch.Tensor
    extrinsics: torch.Tensor

    def to(self, device: torch.device | str) -> "Camera":
        return Camera(self.intrinsics.to(device), self.extrinsics.to(device))


__all__ = ["Camera"]
