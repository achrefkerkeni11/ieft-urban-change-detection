# IEFT/modules/safe_osm_guidance.py
# -*- coding: utf-8 -*-
"""
Conservative T2 OSM guidance for the final IEFT change-detection architecture.

Scientific intent
-----------------
The retained OSM cache is valid as timestamped structured T2 context, but the
available geometry is bbox-approximate rather than exact.  Therefore this module
does NOT rasterize or inject OSM as an early spatial feature map.

Instead, it consumes the tile-local 16-D structured OSM vector already produced
by LEVIRCDDataset and applies only a small, bounded late correction to uncertain
RGB semantic probabilities.

Safety properties
-----------------
* OSM is always part of the final guided architecture when available.
* The RGB/IEFT semantic trunk is never used as a trainable input to this module:
  RGB probability is detached for gating.
* Valid-but-empty OSM produces exactly zero correction.
* Missing OSM produces exactly zero correction.
* The correction is reliability-gated and RGB-uncertainty-gated.
* The signed probability correction is hard bounded.
* The correction head is initialized to exactly zero.
* Its dedicated loss is evaluated relative to a detached RGB prediction, so the
  OSM auxiliary loss cannot alter the RGB semantic trajectory.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeOSMLateGuidance(nn.Module):
    """Bounded late guidance from a tile-local 16-D T2 OSM structure vector."""

    def __init__(
        self,
        struct_dim: int = 16,
        hidden_dim: int = 32,
        max_probability_delta: float = 0.015,
        uncertainty_power: float = 2.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.struct_dim = int(struct_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_probability_delta = float(max_probability_delta)
        self.uncertainty_power = float(uncertainty_power)
        self.eps = float(eps)

        if self.struct_dim <= 0:
            raise ValueError("struct_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if not 0.0 <= self.max_probability_delta <= 0.05:
            raise ValueError("max_probability_delta must lie in [0, 0.05]")
        if self.uncertainty_power <= 0.0:
            raise ValueError("uncertainty_power must be > 0")

        self.struct_encoder = nn.Sequential(
            nn.Linear(self.struct_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(self.hidden_dim, 1)
        self.confidence_head = nn.Linear(self.hidden_dim, 1)
        self.reset_safe_parameters()

    def reset_safe_parameters(self) -> None:
        """Start with exactly zero OSM semantic correction."""
        with torch.no_grad():
            self.delta_head.weight.zero_()
            self.delta_head.bias.zero_()
            self.confidence_head.weight.zero_()
            # Conservative initial confidence ~= 0.18.  Since delta_head is
            # exactly zero this still yields exactly zero initial correction.
            self.confidence_head.bias.fill_(-1.5)

    @staticmethod
    def _as_batch_matrix(
        value: torch.Tensor,
        batch: int,
        width: int,
        name: str,
    ) -> torch.Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape != (batch, width):
            raise ValueError(
                f"{name} must be [B,{width}], got {tuple(value.shape)} "
                f"for batch={batch}"
            )
        return value

    @staticmethod
    def _as_batch_scalar(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
        value = value.reshape(-1)
        if value.numel() == 1 and batch > 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError(
                f"{name} must provide one value per batch item, got {tuple(value.shape)}"
            )
        return value

    def forward(
        self,
        osm_struct_t2: torch.Tensor,
        has_osm_t2: torch.Tensor,
        osm_reliability: torch.Tensor,
        rgb_probability: torch.Tensor,
        dense_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if rgb_probability.ndim == 4 and rgb_probability.shape[1] == 1:
            rgb_probability = rgb_probability.squeeze(1)
        if rgb_probability.ndim != 3:
            raise ValueError(
                "rgb_probability must be [B,H,W], got "
                f"{tuple(rgb_probability.shape)}"
            )

        batch = int(rgb_probability.shape[0])
        dense_h, dense_w = int(dense_shape[0]), int(dense_shape[1])

        osm_struct_t2 = self._as_batch_matrix(
            osm_struct_t2.float(), batch, self.struct_dim, "osm_struct_t2"
        )
        has_osm_t2 = self._as_batch_scalar(
            has_osm_t2.float(), batch, "has_osm_t2"
        )
        osm_reliability = self._as_batch_scalar(
            osm_reliability.float(), batch, "osm_reliability"
        )

        osm_struct_t2 = torch.nan_to_num(
            osm_struct_t2, nan=0.0, posinf=0.0, neginf=0.0
        )
        osm_reliability = torch.clamp(
            torch.nan_to_num(osm_reliability, nan=0.0), 0.0, 1.0
        )
        available = (has_osm_t2 > 0.5).float()

        # Empty OSM is valid context, but absence of mapped features is not a
        # physical no-change statement.  It therefore produces no correction
        # rather than a hard negative veto.
        content_strength = torch.clamp(
            osm_struct_t2.abs().amax(dim=1), 0.0, 1.0
        )
        source_gate = (
            available * osm_reliability * content_strength
        ).view(batch, 1, 1)

        encoded = self.struct_encoder(osm_struct_t2)
        raw_signed_scalar = torch.tanh(self.delta_head(encoded)).view(batch, 1, 1)
        confidence_scalar = torch.sigmoid(
            self.confidence_head(encoded)
        ).view(batch, 1, 1)

        # OSM has no trusted exact spatial geometry in the retained cache.  It
        # can only guide *where RGB is uncertain*, never paint its own footprint.
        rgb_gate_prob = torch.clamp(
            rgb_probability.detach(), self.eps, 1.0 - self.eps
        )
        uncertainty = 4.0 * rgb_gate_prob * (1.0 - rgb_gate_prob)
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0).pow(
            self.uncertainty_power
        )

        effective_gate = confidence_scalar * source_gate * uncertainty
        delta_probability = (
            self.max_probability_delta * raw_signed_scalar * effective_gate
        )

        final_probability = torch.clamp(
            rgb_probability + delta_probability,
            self.eps,
            1.0 - self.eps,
        )
        final_logits = torch.log(final_probability) - torch.log1p(
            -final_probability
        )

        return {
            "delta_probability": delta_probability,
            "final_probability": final_probability,
            "final_logits": final_logits,
            "confidence": confidence_scalar.expand(batch, dense_h, dense_w),
            "uncertainty": uncertainty,
            "effective_gate": effective_gate,
            "source_gate": source_gate,
            "content_strength": content_strength.view(batch, 1, 1),
            "reliability": osm_reliability.view(batch, 1, 1),
        }

    def loss(
        self,
        rgb_probability_detached: torch.Tensor,
        delta_probability: torch.Tensor,
        target: torch.Tensor,
        safety_weight: float = 3.0,
        l1_weight: float = 0.10,
    ) -> Dict[str, torch.Tensor]:
        """Train the OSM residual locally against a detached RGB baseline.

        The objective is the BCE change relative to the protected RGB
        prediction.  Improvements are rewarded, worsening is penalized more
        strongly, and correction magnitude is regularized.
        """
        base = rgb_probability_detached.detach()
        if base.ndim == 4 and base.shape[1] == 1:
            base = base.squeeze(1)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if delta_probability.ndim == 4 and delta_probability.shape[1] == 1:
            delta_probability = delta_probability.squeeze(1)

        target = (target.float() > 0.5).float()
        base = torch.clamp(base.float(), self.eps, 1.0 - self.eps)
        final = torch.clamp(
            base + delta_probability.float(), self.eps, 1.0 - self.eps
        )

        # AMP-safe BCE: PyTorch deliberately rejects probability-space
        # binary_cross_entropy under autocast.  Convert the clamped probabilities
        # back to logits and use binary_cross_entropy_with_logits instead.  This
        # is mathematically equivalent here while remaining safe in fp16 AMP.
        base_logits = torch.log(base) - torch.log1p(-base)
        final_logits = torch.log(final) - torch.log1p(-final)
        base_bce = F.binary_cross_entropy_with_logits(
            base_logits, target, reduction="none"
        )
        final_bce = F.binary_cross_entropy_with_logits(
            final_logits, target, reduction="none"
        )
        delta_bce = final_bce - base_bce

        worsening = F.relu(delta_bce)
        magnitude = delta_probability.abs()

        loss_improvement = delta_bce.mean()
        loss_safety = worsening.mean()
        loss_magnitude = magnitude.mean()
        total = (
            loss_improvement
            + float(safety_weight) * loss_safety
            + float(l1_weight) * loss_magnitude
        )

        with torch.no_grad():
            helpful = (delta_bce < 0.0).float().mean()
            harmful = (delta_bce > 0.0).float().mean()

        return {
            "loss_safe_osm": total,
            "loss_safe_osm_improvement": loss_improvement,
            "loss_safe_osm_safety": loss_safety,
            "loss_safe_osm_magnitude": loss_magnitude,
            "safe_osm_helpful_fraction": helpful,
            "safe_osm_harmful_fraction": harmful,
        }