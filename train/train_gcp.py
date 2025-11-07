"""Training loop for Gaussian category priors with rich monitoring."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.common import GCPSamples
from losses.geometry import occupancy_loss
from losses.regularizers import repulsion_regularizer, scale_regularizer, volume_regularizer
from losses.silhouette import silhouette_loss
from models.gaussian_core import gaussian_2d_params
from utils.render import render_silhouette
from utils.train_utils import (
    MetricLogger,
    alpha_statistics,
    binary_iou,
    scale_statistics,
    update_center_movement,
)


def _ensure_chw_image(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure input is a (C, H, W) tensor on CPU for TensorBoard."""

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        return tensor.cpu()
    if tensor.dim() == 3 and tensor.size(-1) in {1, 3}:
        tensor = tensor.permute(2, 0, 1)
    return tensor.cpu()


def _prepare_overlay(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Create a simple red/green overlay for predicted vs. target masks."""

    pred_gray = pred.squeeze(0).clamp(0.0, 1.0)
    target_gray = target.squeeze(0).clamp(0.0, 1.0)
    overlay = torch.zeros(3, pred_gray.shape[0], pred_gray.shape[1], device=pred.device)
    overlay[0] = target_gray  # red channel for GT
    overlay[1] = pred_gray    # green channel for prediction
    return overlay.cpu()

class GlobalTemplate(torch.nn.Module):
    def __init__(self, K: int, num_classes: int) -> None:
        super().__init__()
        self.mu = torch.nn.Parameter(torch.randn(num_classes, K, 3) * 0.1)
        self.sigma = torch.nn.Parameter(torch.ones(num_classes, K, 3) * 0.05)
        self.alpha = torch.nn.Parameter(torch.ones(num_classes, K, 1) * 0.9)

    def forward(self, category_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mu[category_ids], self.sigma[category_ids], self.alpha[category_ids]

def train(config_path: Path) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset_name")
    if not dataset_name:
        index_path = data_cfg.get("index_file")
        if index_path:
            dataset_name = Path(index_path).parent.name

    dataset = GCPSamples(cfg["data"]["index_file"], categories=data_cfg.get("categories"))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
    )

    model = GlobalTemplate(K=cfg["model"]["K"], num_classes=cfg["model"]["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])

    monitoring_cfg = cfg.get("monitoring", {})
    log_interval = monitoring_cfg.get("log_interval", 100)
    alpha_low = monitoring_cfg.get("alpha_low_threshold", 0.1)
    alpha_high = monitoring_cfg.get("alpha_high_threshold", 0.95)

    tb_cfg = monitoring_cfg.get("tensorboard", {})
    writer: SummaryWriter | None = None
    if tb_cfg.get("enabled", True):
        log_dir = Path(tb_cfg.get("log_dir", "logs/gaussian_category_prior"))
        if tb_cfg.get("unique_run_subdir", True):
            log_dir = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir.as_posix())
        if tb_cfg.get("write_config", True):
            writer.add_text("config", Path(config_path).read_text())

    scalar_interval = tb_cfg.get("scalar_interval", 20)
    histogram_interval = tb_cfg.get("histogram_interval", 100)
    image_interval = tb_cfg.get("image_interval", 200)
    multiview_interval = tb_cfg.get("multiview_interval", 2000)

    output_dir = Path(cfg["train"]["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_filename = "training_metrics.jsonl"
    if dataset_name:
        metrics_filename = f"training_metrics_{dataset_name}.jsonl"
    metrics_file = output_dir / metrics_filename
    if metrics_file.exists():
        metrics_file.unlink()

    metric_logger = MetricLogger()
    center_cache: dict[int, torch.Tensor] = {}
    global_step = 0

    try:
        for epoch in range(cfg["train"]["epochs"]):
            for batch in dataloader:
                global_step += 1

                mask = batch["mask"].to(device)
                intr = batch["K"].to(device)
                extr = batch["Rt"].to(device)
                occ_pts = batch["occ_pts"].to(device)
                occ_lbl = batch["occ_lbl"].to(device)
                cat_id = batch["cat_id"].to(device)

                centers, scales, alpha = model(cat_id)
                centers_2d, scales_2d, depth = gaussian_2d_params(centers, scales, None, intr, extr)
                silhouette = render_silhouette(
                    centers_2d, scales_2d, depth, mask.size(2), mask.size(3), alpha
                )

                inv_scales = 1.0 / (scales**2 + 1e-6)
                loss_2d = silhouette_loss(silhouette, mask)
                loss_3d = occupancy_loss(occ_pts, occ_lbl, centers, inv_scales, alpha)
                reg_scale = scale_regularizer(scales)
                reg_repulsion = repulsion_regularizer(centers)
                reg_volume = volume_regularizer(scales)
                scale_w = cfg["loss"].get("scale_reg_weight", 0.1)
                repulsion_w = cfg["loss"].get("repulsion_reg_weight", 0.1)
                volume_w = cfg["loss"].get("volume_reg_weight", 0.01)
                loss_reg = scale_w * reg_scale + repulsion_w * reg_repulsion + volume_w * reg_volume
                loss = cfg["loss"]["w_2d"] * loss_2d + cfg["loss"]["w_3d"] * loss_3d + loss_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    iou = binary_iou(silhouette, mask)
                    alpha_stats = alpha_statistics(alpha, alpha_low, alpha_high)
                    scale_stats = scale_statistics(scales)
                    movement = update_center_movement(center_cache, cat_id, centers)

                metric_logger.update(
                    loss=float(loss.item()),
                    loss_2d=float(loss_2d.item()),
                    loss_3d=float(loss_3d.item()),
                    loss_reg=float(loss_reg.item()),
                    loss_reg_scale=float((scale_w * reg_scale).item()),
                    loss_reg_repulsion=float((repulsion_w * reg_repulsion).item()),
                    loss_reg_volume=float((volume_w * reg_volume).item()),
                    silhouette_iou=iou,
                    center_movement=movement,
                    **alpha_stats,
                    **scale_stats,
                )

                if writer and global_step % scalar_interval == 0:
                    writer.add_scalar("loss/total", loss.item(), global_step)
                    writer.add_scalar("loss/silhouette", loss_2d.item(), global_step)
                    writer.add_scalar("loss/occupancy", loss_3d.item(), global_step)
                    writer.add_scalar("loss/reg_total", loss_reg.item(), global_step)
                    writer.add_scalar("loss/reg_scale", (scale_w * reg_scale).item(), global_step)
                    writer.add_scalar(
                        "loss/reg_repulsion", (repulsion_w * reg_repulsion).item(), global_step
                    )
                    writer.add_scalar("loss/reg_volume", (volume_w * reg_volume).item(), global_step)
                    writer.add_scalar("metrics/silhouette_iou", iou, global_step)
                    if movement is not None:
                        writer.add_scalar("gaussian/mu_update", movement, global_step)
                    writer.add_scalar("gaussian/alpha_mean", alpha_stats.get("alpha_mean", 0.0), global_step)
                    writer.add_scalar("gaussian/alpha_low_frac", alpha_stats.get("alpha_low_frac", 0.0), global_step)
                    writer.add_scalar(
                        "gaussian/alpha_high_frac", alpha_stats.get("alpha_high_frac", 0.0), global_step
                    )
                    writer.add_scalar("gaussian/scale_min", scale_stats.get("scale_min", 0.0), global_step)
                    writer.add_scalar("gaussian/scale_max", scale_stats.get("scale_max", 0.0), global_step)
                    writer.add_scalar("gaussian/scale_mean", scale_stats.get("scale_mean", 0.0), global_step)
                    writer.add_scalar("gaussian/scale_p95", scale_stats.get("scale_p95", 0.0), global_step)

                if writer and histogram_interval > 0 and global_step % histogram_interval == 0:
                    writer.add_histogram("gaussian/alpha", alpha.detach().cpu(), global_step)
                    writer.add_histogram("gaussian/scale", scales.detach().cpu(), global_step)

                if (
                    writer
                    and image_interval > 0
                    and global_step % image_interval == 0
                    and mask.size(0) > 0
                ):
                    sample_idx = 0
                    pred_img = _ensure_chw_image(silhouette[sample_idx].detach())
                    gt_img = _ensure_chw_image(mask[sample_idx].detach())
                    render_rgb = pred_img.repeat(3, 1, 1) if pred_img.size(0) == 1 else pred_img
                    overlay = _prepare_overlay(silhouette[sample_idx].detach(), mask[sample_idx].detach())
                    writer.add_image("vis/silhouette_pred", pred_img, global_step)
                    writer.add_image("vis/silhouette_gt", gt_img, global_step)
                    writer.add_image("vis/gaussian_render", render_rgb, global_step)
                    writer.add_image("vis/overlay", overlay, global_step)

                if global_step % log_interval == 0:
                    summary = metric_logger.summary()
                    log_entry = {
                        "epoch": epoch,
                        "iter": global_step,
                        **summary,
                    }
                    with open(metrics_file, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(log_entry) + "\n")

                    msg = (
                        f"[epoch {epoch:03d} iter {global_step:06d}] "
                        f"loss={summary.get('loss', 0.0):.4f} "
                        f"L2D={summary.get('loss_2d', 0.0):.4f} "
                        f"L3D={summary.get('loss_3d', 0.0):.4f} "
                        f"Lreg={summary.get('loss_reg', 0.0):.4f} "
                        f"IoU={summary.get('silhouette_iou', 0.0):.3f} "
                        f"alpha_mu={summary.get('alpha_mean', 0.0):.3f} "
                        f"alpha_low={summary.get('alpha_low_frac', 0.0):.3f} "
                        f"alpha_high={summary.get('alpha_high_frac', 0.0):.3f} "
                        f"scale_min={summary.get('scale_min', 0.0):.4f} "
                        f"scale_p95={summary.get('scale_p95', 0.0):.4f} "
                        f"scale_max={summary.get('scale_max', 0.0):.4f} "
                        f"move={summary.get('center_movement', 0.0):.4f}"
                    )
                    print(msg)
                    metric_logger.reset()

            if metric_logger.storage:
                summary = metric_logger.summary()
                msg = (
                    f"[epoch {epoch:03d} summary] loss={summary.get('loss', 0.0):.4f} "
                    f"L2D={summary.get('loss_2d', 0.0):.4f} "
                    f"L3D={summary.get('loss_3d', 0.0):.4f} "
                    f"Lreg={summary.get('loss_reg', 0.0):.4f} "
                    f"IoU={summary.get('silhouette_iou', 0.0):.3f}"
                )
                print(msg)
                log_entry = {"epoch": epoch, "iter": global_step, **summary}
                with open(metrics_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(log_entry) + "\n")
                metric_logger.reset()
    finally:
        ckpt_name = "gcp_final.pt"
        if dataset_name:
            ckpt_name = f"gcp_final_{dataset_name}.pt"
        torch.save({"model": model.state_dict()}, output_dir / ckpt_name)
        if writer:
            writer.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args.config)
