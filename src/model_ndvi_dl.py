"""Track D: a 1D temporal CNN over the per-pixel NDVI time series.

This is the neural counterpart of Track C: it consumes the *same* 46-value
ordered NDVI sequence per pixel, so comparing the two isolates model class
(tree ensemble vs. convolutional net) with the feature representation held
fixed. Architecture follows the TempCNN family (Pelletier et al., 2019),
which was designed specifically for pixel-wise satellite image time series.

Why a CNN rather than more trees: a Random Forest can only threshold
individual dates ("NDVI at t28 > 0.6"). A 1D convolution slides a learned
kernel along the time axis, so it can respond to local *shapes* -- the rate
of green-up, the width of a plateau, the sharpness of senescence -- which
are the features an agronomist would actually describe a crop calendar with.

Design note on pooling: the convolution stack uses stride-2 max-pooling but
deliberately ends in a flatten rather than a global pool. Global pooling
would make the representation translation-invariant along time, which would
discard absolute date information -- exactly the property that made Track A's
summary statistics fail. Pooling to a coarse time axis (46 -> 11 steps) keeps
position while cutting parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TempCNN(nn.Module):
    def __init__(
        self,
        n_timesteps: int = 46,
        n_classes: int = 20,
        channels: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.features = nn.Sequential(
            conv_block(1, channels),
            conv_block(channels, channels),
            nn.MaxPool1d(2),                       # 46 -> 23
            conv_block(channels, channels),
            conv_block(channels, channels),
            nn.MaxPool1d(2),                       # 23 -> 11
        )
        pooled_len = n_timesteps // 4              # 46 -> 11
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # keeps (coarse) date position
            nn.Linear(channels * pooled_len, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        """x: (B, T) raw NDVI sequence -> (B, n_classes) logits."""
        if x.dim() == 2:
            x = x.unsqueeze(1)                     # (B, 1, T)
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = TempCNN(n_timesteps=46, n_classes=20)
    print(f"parameters: {count_parameters(m):,}")
    out = m(torch.randn(8, 46))
    print("output:", out.shape)
    assert out.shape == (8, 20)
    print("OK")
