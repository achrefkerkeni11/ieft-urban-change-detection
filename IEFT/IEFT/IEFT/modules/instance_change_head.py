"""Optional center-and-offset head for changed-building instances.

The head consumes the final 32-channel feature produced by the semantic pixel
decoder.  It is deliberately side-car only: no instance prediction is fed
back into the scientific semantic change map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in (8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


class InstanceChangeHead(nn.Module):
    """Predict a center heatmap and per-pixel ``(dy, dx)`` center offsets."""

    def __init__(
        self,
        in_ch: int = 32,
        hidden: int = 32,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        if in_ch <= 0 or hidden <= 0:
            raise ValueError("in_ch and hidden must be positive")

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
                nn.GroupNorm(_num_groups(cout), cout),
                nn.GELU(),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            )

        self.center_branch = nn.Sequential(
            block(in_ch, hidden),
            block(hidden, hidden),
        )
        self.center_head = nn.Conv2d(hidden, 1, 1)

        self.offset_branch = nn.Sequential(
            block(in_ch, hidden),
            block(hidden, hidden),
        )
        self.offset_head = nn.Conv2d(hidden, 2, 1)

    def forward(self, dense_feat: torch.Tensor):
        if dense_feat.ndim != 4:
            raise ValueError(
                "InstanceChangeHead expects [B,C,H,W], got "
                f"{tuple(dense_feat.shape)}"
            )
        center_logits = self.center_head(self.center_branch(dense_feat))
        offsets_yx = self.offset_head(self.offset_branch(dense_feat))
        return center_logits, offsets_yx
