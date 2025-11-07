"""Conditional Gaussian template generator."""
from __future__ import annotations

import torch
import torch.nn as nn

class MaskEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        features = self.network(mask)
        return features.flatten(1)

class GCPGenerator(nn.Module):
    def __init__(self, num_classes: int, K: int = 2048, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = MaskEncoder(1, latent_dim)
        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.head = nn.Linear(latent_dim * 2, K * 7)
        self.K = K

    def forward(self, mask: torch.Tensor, category: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.embedding(category)
        encoded = self.encoder(mask)
        fused = torch.cat([encoded, embedding], dim=-1)
        output = self.head(fused).view(mask.size(0), self.K, 7)
        centers = output[..., :3]
        scales = output[..., 3:6].abs() + 1e-3
        alpha = output[..., -1:].sigmoid()
        return centers, scales, alpha

__all__ = ["MaskEncoder", "GCPGenerator"]
