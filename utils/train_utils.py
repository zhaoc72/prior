"""General training utilities and monitoring helpers."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import torch.nn.functional as F


def to_device(batch: dict, device: torch.device | str) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def binary_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute mean IoU between probabilistic predictions and binary targets."""

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred_mask = (pred >= threshold)
    target_mask = (target >= threshold)

    intersection = (pred_mask & target_mask).flatten(1).sum(dim=-1).float()
    union = (pred_mask | target_mask).flatten(1).sum(dim=-1).float().clamp_min(1.0)
    iou = (intersection / union).mean().item()
    return float(iou)


def _morphological_kernel(device: torch.device) -> torch.Tensor:
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
    return kernel


def _erode(mask: torch.Tensor) -> torch.Tensor:
    kernel = _morphological_kernel(mask.device)
    conv = F.conv2d(mask, kernel, padding=1)
    return (conv == kernel.numel()).float()


def _dilate(mask: torch.Tensor) -> torch.Tensor:
    kernel = _morphological_kernel(mask.device)
    conv = F.conv2d(mask, kernel, padding=1)
    return (conv > 0).float()


def boundary_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Boundary F1 score with a one-pixel tolerance."""

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()

    pred_boundary = (pred_bin - _erode(pred_bin)).clamp_min(0.0)
    target_boundary = (target_bin - _erode(target_bin)).clamp_min(0.0)

    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return 0.0

    pred_dilate = _dilate(pred_boundary)
    target_dilate = _dilate(target_boundary)

    true_positive = (pred_boundary * target_dilate).sum(dim=(1, 2, 3))
    pred_total = pred_boundary.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    target_true = (target_boundary * pred_dilate).sum(dim=(1, 2, 3))
    target_total = target_boundary.sum(dim=(1, 2, 3)).clamp_min(1e-6)

    precision = (true_positive / pred_total).mean()
    recall = (target_true / target_total).mean()
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return float(f1.item())


def alpha_statistics(
    alpha: torch.Tensor, low_threshold: float = 0.1, high_threshold: float = 0.95
) -> dict[str, float]:
    """Summarise opacity distribution for monitoring."""

    flat = alpha.detach().flatten().float()
    if flat.numel() == 0:
        return {
            "alpha_mean": 0.0,
            "alpha_median": 0.0,
            "alpha_low_frac": 0.0,
            "alpha_high_frac": 0.0,
        }

    alpha_mean = flat.mean().item()
    alpha_median = flat.median().item()
    alpha_low = (flat < low_threshold).float().mean().item()
    alpha_high = (flat > high_threshold).float().mean().item()
    return {
        "alpha_mean": float(alpha_mean),
        "alpha_median": float(alpha_median),
        "alpha_low_frac": float(alpha_low),
        "alpha_high_frac": float(alpha_high),
    }


def scale_statistics(scales: torch.Tensor) -> dict[str, float]:
    """Compute range statistics for Gaussian scales."""

    flat = scales.detach().flatten().float()
    if flat.numel() == 0:
        return {"scale_min": 0.0, "scale_max": 0.0, "scale_mean": 0.0, "scale_p95": 0.0}

    scale_min = flat.min().item()
    scale_max = flat.max().item()
    scale_mean = flat.mean().item()
    scale_p95 = torch.quantile(flat, 0.95).item()
    return {
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "scale_mean": float(scale_mean),
        "scale_p95": float(scale_p95),
    }


def update_center_movement(
    cache: dict[int, torch.Tensor], cat_ids: torch.Tensor, centers: torch.Tensor
) -> float | None:
    """Update centre cache and return average movement for present categories."""

    cat_ids = cat_ids.detach().cpu().tolist()
    centers_cpu = centers.detach().cpu()
    movements: list[float] = []

    for idx, cat in enumerate(cat_ids):
        current = centers_cpu[idx]
        if cat in cache:
            prev = cache[cat]
            displacement = torch.norm(current - prev, dim=-1).mean().item()
            movements.append(displacement)
        cache[cat] = current.clone()

    if movements:
        return float(sum(movements) / len(movements))
    return None


@dataclass
class MetricLogger:
    """Aggregate metrics across iterations for stable logging."""

    storage: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def update(self, **metrics: float | None) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            self.storage[key].append(float(value))

    def summary(self) -> dict[str, float]:
        return {key: float(fmean(values)) for key, values in self.storage.items() if values}

    def reset(self) -> None:
        self.storage.clear()

    def dump(self, path: Path | None, meta: dict[str, Any]) -> None:
        if path is None:
            return
        summary = self.summary()
        payload = {**meta, **summary}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


__all__ = [
    "MetricLogger",
    "alpha_statistics",
    "binary_iou",
    "boundary_f1",
    "scale_statistics",
    "to_device",
    "update_center_movement",
]
