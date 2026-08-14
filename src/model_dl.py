"""Deep-learning track: a lightweight Temporal-Attention U-Net.

Given only ~66 training patches, a full U-TAE (the original PASTIS
benchmark model) is likely to overfit and is expensive to tune; per the
assignment's preference for "a simpler baseline model with strong
reasoning ... over an unnecessarily complex model", this implements a
smaller model in the same spirit:

  1. A shared 2D conv "stem" extracts spatial features independently from
     every one of the T acquisitions.
  2. A per-pixel temporal-attention layer collapses the T frames into one
     feature map, learning *per pixel* how much to trust each acquisition.
     This is a deliberate, interpretable response to the EDA finding that
     several acquisition dates are likely cloud-contaminated (dataset-wide
     NDVI dips) — the model can learn to downweight those frames itself
     rather than requiring an explicit cloud mask.
  3. A small 2-level U-Net encoder/decoder turns the aggregated feature map
     into a per-pixel class-logit map.

Input: (B, T, C, H, W) float tensor (band-normalized reflectance).
Output: (B, n_classes, H, W) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalAttentionPool(nn.Module):
    """Collapses (B, T, C, H, W) -> (B, C, H, W) via per-pixel softmax attention over T."""

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        scores = self.score(x.reshape(b * t, c, h, w)).reshape(b, t, 1, h, w)
        weights = torch.softmax(scores, dim=1)  # softmax over T, per pixel
        pooled = (x * weights).sum(dim=1)  # (B, C, H, W)
        return pooled, weights.squeeze(2)  # also return (B, T, H, W) for inspection/viz


class TemporalAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        n_classes: int = 20,
        temporal_hidden_dim: int = 64,
        unet_base_channels: int = 32,
    ):
        super().__init__()
        base = unet_base_channels

        self.stem = ConvBNReLU(in_channels, base)
        self.temporal_pool = TemporalAttentionPool(base, hidden=temporal_hidden_dim)

        self.enc1 = ConvBNReLU(base, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBNReLU(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBNReLU(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBNReLU(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBNReLU(base * 2, base)

        self.classifier = nn.Conv2d(base, n_classes, 1)

    def forward(self, x, return_attention: bool = False):
        b, t, c, h, w = x.shape
        frame_features = self.stem(x.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w)
        pooled, attn_weights = self.temporal_pool(frame_features)  # (B, base, H, W)

        e1 = self.enc1(pooled)
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))

        d2 = self.up2(bn)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.classifier(d1)
        if return_attention:
            return logits, attn_weights
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TemporalAttentionUNet(in_channels=10, n_classes=20)
    print(f"parameters: {count_parameters(m):,}")
    x = torch.randn(2, 46, 10, 128, 128)
    logits, attn = m(x, return_attention=True)
    print("logits:", logits.shape, "attn:", attn.shape)
    assert logits.shape == (2, 20, 128, 128)
    assert attn.shape == (2, 46, 128, 128)
    print("OK")
