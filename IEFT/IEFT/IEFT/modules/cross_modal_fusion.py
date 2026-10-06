"""Optional checkpoint-safe bidirectional task/visual interaction.

The historical FIE blocks remain the authoritative path.  This module adds a
separate residual companion rather than renaming/replacing their parameters.
Its final projection is exactly zero, so enabling it preserves the historical
output at optimizer step zero while still allowing the new path to learn.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BidirectionalFIEResidual(nn.Module):
    """Enrich task tokens from vision, then return a zero-init visual residual."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.20):
        super().__init__()
        self.task_from_visual = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_from_task = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_norm = nn.LayerNorm(dim)
        self.task_norm = nn.LayerNorm(dim)
        self.enriched_task_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim, bias=True)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_residual_parameters()

    def reset_residual_parameters(self) -> None:
        with torch.no_grad():
            self.residual_scale.fill_(1.0)
            self.output_projection.weight.zero_()
            self.output_projection.bias.zero_()

    def forward(self, visual: torch.Tensor, tasks: torch.Tensor) -> torch.Tensor:
        if visual.ndim != 3 or tasks.ndim != 3 or visual.shape[0] != tasks.shape[0]:
            raise ValueError(
                "Bidirectional FIE inputs must be [B,L,D] and [B,T,D], got "
                f"{tuple(visual.shape)} and {tuple(tasks.shape)}"
            )
        visual_norm = self.visual_norm(visual)
        task_update, _ = self.task_from_visual(
            self.task_norm(tasks), visual_norm, visual_norm, need_weights=False
        )
        enriched_tasks = self.enriched_task_norm(tasks + task_update)
        visual_update, _ = self.visual_from_task(
            visual_norm, enriched_tasks, enriched_tasks, need_weights=False
        )
        return self.output_projection(visual_update) * self.residual_scale


__all__ = ["BidirectionalFIEResidual"]
