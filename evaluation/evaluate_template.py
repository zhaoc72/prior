"""Offline evaluation script for Gaussian category priors."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.common import GCPSamples
from evaluation.metrics import (
    aggregate,
    chamfer_L2,
    fscore_at_tau,
    sample_gaussian_template,
    volume_iou,
)
from losses.geometry import gaussian_occupancy
from models.gaussian_core import gaussian_2d_params
from train.train_gcp import GlobalTemplate
from utils.render import render_silhouette
from utils.train_utils import binary_iou, boundary_f1

CHAMFER_TARGET = 0.005
FSCORE_TARGET = 0.60
SILHOUETTE_TARGET = 0.70
BOUNDARY_TARGET = 0.60
VOLUME_IOU_TARGET = 0.50


@torch.no_grad()
def evaluate(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dataset = GCPSamples(args.index_file, categories=cfg["data"].get("categories"))
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = GlobalTemplate(K=cfg["model"]["K"], num_classes=cfg["model"]["num_classes"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = defaultdict(list)
    metrics_by_cat: dict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    evaluation_cfg = cfg.get("evaluation", {})
    num_samples = args.num_template_samples or evaluation_cfg.get("template_samples", 5000)
    fscore_tau = args.fscore_tau or evaluation_cfg.get("fscore_tau", 0.02)
    occ_threshold = args.occ_threshold or evaluation_cfg.get("occ_threshold", 0.5)
    chunk_size = args.chunk_size or evaluation_cfg.get("chunk_size", 2048)

    vis_written: set[str] = set()
    if args.vis_dir is not None:
        args.vis_dir.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        mask = batch["mask"].to(device)
        intr = batch["K"].to(device)
        extr = batch["Rt"].to(device)
        occ_pts = batch["occ_pts"].to(device)
        occ_lbl = batch["occ_lbl"].to(device)
        cat_id = batch["cat_id"].to(device)
        cat_name = batch.get("cat", [f"cat_{int(cat_id.item())}"])[0]

        centers, scales, alpha = model(cat_id)
        centers_2d, scales_2d, depth = gaussian_2d_params(centers, scales, None, intr, extr)
        silhouette = render_silhouette(centers_2d, scales_2d, depth, mask.size(2), mask.size(3), alpha)

        iou = binary_iou(silhouette, mask)
        bf1 = boundary_f1(silhouette, mask)
        inv_scales = 1.0 / (scales**2 + 1e-6)
        occ = gaussian_occupancy(occ_pts, centers, inv_scales, alpha)
        occ_iou = volume_iou(occ, occ_lbl, threshold=occ_threshold)

        metrics["silhouette_iou"].append(iou)
        metrics["boundary_f1"].append(bf1)
        metrics["volume_iou"].append(occ_iou)
        metrics_by_cat[cat_name]["silhouette_iou"].append(iou)
        metrics_by_cat[cat_name]["boundary_f1"].append(bf1)
        metrics_by_cat[cat_name]["volume_iou"].append(occ_iou)

        if batch["surface_pts"].numel() > 0:
            surface = batch["surface_pts"].squeeze(0).to(device)
            samples = sample_gaussian_template(centers.squeeze(0), scales.squeeze(0), alpha.squeeze(0), num_samples)
            chamfer = chamfer_L2(samples, surface, chunk_size=chunk_size)
            fscore = fscore_at_tau(samples, surface, tau=fscore_tau, chunk_size=chunk_size)
            metrics["chamfer_l2"].append(chamfer)
            metrics["fscore"].append(fscore)
            metrics_by_cat[cat_name]["chamfer_l2"].append(chamfer)
            metrics_by_cat[cat_name]["fscore"].append(fscore)

        if args.vis_dir is not None and cat_name not in vis_written:
            pred_img = silhouette.squeeze().detach().cpu().numpy()
            gt_img = mask.squeeze().detach().cpu().numpy()
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(pred_img, cmap="gray")
            axes[0].set_title("Predicted")
            axes[0].axis("off")
            axes[1].imshow(gt_img, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            fig.tight_layout()
            fig.savefig(args.vis_dir / f"{cat_name}.png", dpi=150)
            plt.close(fig)
            vis_written.add(cat_name)

    aggregate_metrics = {name: aggregate(values) for name, values in metrics.items()}
    per_category = {
        cat: {metric: aggregate(values) for metric, values in metric_map.items()}
        for cat, metric_map in metrics_by_cat.items()
    }

    thresholds = {
        "silhouette_iou": {
            "value": aggregate_metrics.get("silhouette_iou"),
            "target": SILHOUETTE_TARGET,
            "status": aggregate_metrics.get("silhouette_iou", 0.0) >= SILHOUETTE_TARGET,
        },
        "boundary_f1": {
            "value": aggregate_metrics.get("boundary_f1"),
            "target": BOUNDARY_TARGET,
            "status": aggregate_metrics.get("boundary_f1", 0.0) >= BOUNDARY_TARGET,
        },
        "volume_iou": {
            "value": aggregate_metrics.get("volume_iou"),
            "target": VOLUME_IOU_TARGET,
            "status": aggregate_metrics.get("volume_iou", 0.0) >= VOLUME_IOU_TARGET,
        },
    }

    if "chamfer_l2" in aggregate_metrics:
        thresholds["chamfer_l2"] = {
            "value": aggregate_metrics["chamfer_l2"],
            "target": CHAMFER_TARGET,
            "status": aggregate_metrics["chamfer_l2"] <= CHAMFER_TARGET,
        }
    if "fscore" in aggregate_metrics:
        thresholds["fscore"] = {
            "value": aggregate_metrics["fscore"],
            "target": FSCORE_TARGET,
            "status": aggregate_metrics["fscore"] >= FSCORE_TARGET,
        }

    report = {
        "aggregate": aggregate_metrics,
        "per_category": per_category,
        "thresholds": thresholds,
    }

    return report


def save_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Training config used for the template")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint")
    parser.add_argument("--index_file", type=Path, help="Override dataset index for evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_template_samples", type=int, default=0)
    parser.add_argument("--fscore_tau", type=float, default=0.0)
    parser.add_argument("--occ_threshold", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--vis_dir", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.index_file is None:
        args.index_file = Path(cfg["data"]["index_file"])
    args.index_file = Path(args.index_file)

    report = evaluate(cfg, args)

    out_path = args.out
    if out_path is None:
        out_path = Path(cfg["train"]["out_dir"]) / "evaluation_report.json"
    save_report(report, out_path)

    print("=== Aggregate Metrics ===")
    for key, value in report["aggregate"].items():
        print(f"{key:16s}: {value:.4f}")

    print("\n=== Threshold Checks ===")
    for key, meta in report["thresholds"].items():
        value = meta["value"]
        target = meta["target"]
        status = "PASS" if meta["status"] else "ALERT"
        if value is None or value != value:  # NaN check
            print(f"{key:16s}: n/a (target {target:.3f}) [{status}]")
        else:
            comp = ">=" if key not in {"chamfer_l2"} else "<="
            print(f"{key:16s}: {value:.4f} ({comp} {target:.3f}) [{status}]")

    print("\n=== Per-category summary (top 5) ===")
    for cat, metrics_map in list(report["per_category"].items())[:5]:
        formatted = ", ".join(f"{k}:{v:.3f}" for k, v in metrics_map.items())
        print(f"{cat:16s}: {formatted}")


if __name__ == "__main__":
    main()
