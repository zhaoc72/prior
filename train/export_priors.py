"""Export Gaussian templates from trained checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from train.train_gcp import GlobalTemplate

def export_templates(checkpoint: Path, num_classes: int, K: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu")
    model = GlobalTemplate(K=K, num_classes=num_classes)
    model.load_state_dict(state["model"])
    model.eval()

    mu = model.mu.detach().cpu().numpy()
    sigma = model.sigma.detach().cpu().numpy()
    alpha = model.alpha.detach().cpu().numpy()

    for idx in range(num_classes):
        np.savez(out_dir / f"cat_{idx:02d}.npz", mu=mu[idx], scale=sigma[idx], alpha=alpha[idx])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    export_templates(args.ckpt, cfg["model"]["num_classes"], cfg["model"]["K"], Path(args.out))
