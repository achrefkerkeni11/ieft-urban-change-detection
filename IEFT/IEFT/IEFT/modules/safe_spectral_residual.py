# -*- coding: utf-8 -*-
"""Conservative late spectral correction for IEFT / LEVIR-CD.

This module is intentionally isolated from the RGB semantic trunk:

* it consumes only cached genuine NDVI/NDWI tensors and their validity masks;
* it never modifies ViT/FIE/multiscale features;
* its correction is bounded in probability space;
* it is reliability-gated and uncertainty-gated;
* it starts at exactly zero correction;
* its training loss can be computed from a detached RGB prediction so gradients
  remain local to this module.

The goal is not to let coarse Landsat context re-learn the segmentation.  It is
only allowed to make a small late correction where the RGB model is uncertain.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SafeSpectralLateResidual(nn.Module):
    """Small bounded NDVI/NDWI correction applied after RGB dense prediction.

    Inputs use two index channels per date: ``[NDVI, NDWI]``.  The branch builds
    absolute/signed temporal context, downsamples it to a coarse grid, predicts a
    signed correction and a confidence gate, and finally bounds the correction
    in *probability* space.

    ``max_probability_delta`` is a hard bound before the reliability/uncertainty
    gates, so the actual correction can only be smaller.
    """

    def __init__(
        self,
        hidden_dim: int = 24,
        coarse_grid_size: int = 16,
        max_probability_delta: float = 0.02,
        min_reliability: float = 0.50,
        uncertainty_power: float = 2.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.coarse_grid_size = int(coarse_grid_size)
        self.max_probability_delta = float(max_probability_delta)
        self.min_reliability = float(min_reliability)
        self.uncertainty_power = float(uncertainty_power)
        self.eps = float(eps)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.coarse_grid_size <= 0:
            raise ValueError("coarse_grid_size must be > 0")
        if not 0.0 <= self.max_probability_delta <= 0.10:
            raise ValueError("max_probability_delta must lie in [0, 0.10]")
        if not 0.0 <= self.min_reliability < 1.0:
            raise ValueError("min_reliability must lie in [0, 1)")
        if self.uncertainty_power <= 0.0:
            raise ValueError("uncertainty_power must be > 0")

        # 10 channels:
        # T1(2), T2(2), signed delta(2), absolute delta(2), valid_T1, valid_T2.
        self.encoder = nn.Sequential(
            nn.Conv2d(10, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(self.hidden_dim, 1, 1, bias=True)
        self.confidence_head = nn.Conv2d(self.hidden_dim, 1, 1, bias=True)
        self.reset_safe_parameters()

    def reset_safe_parameters(self) -> None:
        """Initialize the branch with exactly zero semantic correction."""
        with torch.no_grad():
            self.delta_head.weight.zero_()
            self.delta_head.bias.zero_()
            self.confidence_head.weight.zero_()
            # A conservative initial confidence (~0.18).  It has no semantic
            # effect at initialization because delta_head is exactly zero.
            self.confidence_head.bias.fill_(-1.5)

    @staticmethod
    def _as_bchw(x: torch.Tensor, channels: int, name: str) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4 or x.shape[1] != channels:
            raise ValueError(
                f"{name} must be [B,{channels},H,W], got {tuple(x.shape)}"
            )
        return x

    @staticmethod
    def _as_batch_scalar(x: torch.Tensor, batch: int, name: str) -> torch.Tensor:
        x = x.reshape(-1)
        if x.numel() == 1 and batch > 1:
            x = x.expand(batch)
        if x.numel() != batch:
            raise ValueError(
                f"{name} must provide one value per batch item, got {tuple(x.shape)}"
            )
        return x

    def _reliability_gate(
        self,
        reliability: torch.Tensor,
        has_t1: torch.Tensor,
        has_t2: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        reliability = self._as_batch_scalar(reliability.float(), batch, "reliability")
        has_t1 = self._as_batch_scalar(has_t1.float(), batch, "has_t1")
        has_t2 = self._as_batch_scalar(has_t2.float(), batch, "has_t2")

        denom = max(1e-6, 1.0 - self.min_reliability)
        rel = torch.clamp((reliability - self.min_reliability) / denom, 0.0, 1.0)
        pair = (has_t1 > 0.5).float() * (has_t2 > 0.5).float()
        return (rel * pair).view(batch, 1, 1)

    def forward(
        self,
        spectral_t1: torch.Tensor,
        spectral_t2: torch.Tensor,
        valid_t1: torch.Tensor,
        valid_t2: torch.Tensor,
        has_t1: torch.Tensor,
        has_t2: torch.Tensor,
        reliability: torch.Tensor,
        rgb_probability: torch.Tensor,
        dense_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        spectral_t1 = self._as_bchw(spectral_t1.float(), 2, "spectral_t1")
        spectral_t2 = self._as_bchw(spectral_t2.float(), 2, "spectral_t2")
        valid_t1 = self._as_bchw(valid_t1.float(), 1, "valid_t1")
        valid_t2 = self._as_bchw(valid_t2.float(), 1, "valid_t2")

        if spectral_t1.shape != spectral_t2.shape:
            raise ValueError("T1/T2 spectral index tensors must have the same shape")
        if valid_t1.shape[0] != spectral_t1.shape[0] or valid_t2.shape[0] != spectral_t1.shape[0]:
            raise ValueError("Spectral validity masks must match the spectral batch size")

        batch = spectral_t1.shape[0]
        dense_h, dense_w = int(dense_shape[0]), int(dense_shape[1])

        if rgb_probability.ndim == 4 and rgb_probability.shape[1] == 1:
            rgb_probability = rgb_probability.squeeze(1)
        if rgb_probability.ndim != 3 or rgb_probability.shape[0] != batch:
            raise ValueError(
                "rgb_probability must be [B,H,W], got "
                f"{tuple(rgb_probability.shape)}"
            )

        spectral_t1 = torch.nan_to_num(spectral_t1, nan=0.0, posinf=0.0, neginf=0.0)
        spectral_t2 = torch.nan_to_num(spectral_t2, nan=0.0, posinf=0.0, neginf=0.0)
        valid_t1 = torch.clamp(torch.nan_to_num(valid_t1, nan=0.0), 0.0, 1.0)
        valid_t2 = torch.clamp(torch.nan_to_num(valid_t2, nan=0.0), 0.0, 1.0)

        t1 = spectral_t1 * valid_t1
        t2 = spectral_t2 * valid_t2
        pair_valid = valid_t1 * valid_t2
        signed_delta = (t2 - t1) * pair_valid
        absolute_delta = signed_delta.abs()

        features = torch.cat(
            [t1, t2, signed_delta, absolute_delta, valid_t1, valid_t2], dim=1
        )
        coarse = F.adaptive_avg_pool2d(
            features, (self.coarse_grid_size, self.coarse_grid_size)
        )
        encoded = self.encoder(coarse)

        raw_signed = torch.tanh(self.delta_head(encoded))
        confidence = torch.sigmoid(self.confidence_head(encoded))
        raw_signed = F.interpolate(
            raw_signed,
            size=(dense_h, dense_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        confidence = F.interpolate(
            confidence,
            size=(dense_h, dense_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        pair_valid_coarse = F.adaptive_avg_pool2d(
            pair_valid, (self.coarse_grid_size, self.coarse_grid_size)
        )
        pair_valid_dense = F.interpolate(
            pair_valid_coarse,
            size=(dense_h, dense_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        pair_valid_dense = torch.clamp(pair_valid_dense, 0.0, 1.0)

        # RGB confidence is used only as a gate.  Detaching it prevents the
        # spectral branch from shaping the RGB trunk through this path.
        rgb_gate_prob = torch.clamp(rgb_probability.detach(), self.eps, 1.0 - self.eps)
        uncertainty = 4.0 * rgb_gate_prob * (1.0 - rgb_gate_prob)
        uncertainty = torch.clamp(uncertainty, 0.0, 1.0).pow(self.uncertainty_power)

        reliability_gate = self._reliability_gate(
            reliability, has_t1, has_t2, batch
        )
        effective_gate = (
            confidence
            * uncertainty
            * pair_valid_dense
            * reliability_gate
        )

        delta_probability = (
            self.max_probability_delta * raw_signed * effective_gate
        )
        final_probability = torch.clamp(
            rgb_probability + delta_probability,
            self.eps,
            1.0 - self.eps,
        )
        final_logits = torch.log(final_probability) - torch.log1p(-final_probability)

        return {
            "delta_probability": delta_probability,
            "final_probability": final_probability,
            "final_logits": final_logits,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "effective_gate": effective_gate,
            "reliability_gate": reliability_gate,
            "pair_valid": pair_valid_dense,
        }

    def loss(
        self,
        rgb_probability_detached: torch.Tensor,
        delta_probability: torch.Tensor,
        target: torch.Tensor,
        safety_weight: float = 2.0,
        l1_weight: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Train only the residual branch against a detached RGB prediction.

        The objective is expressed as *change in BCE relative to RGB*.  It is
        zero when the correction is zero, rewards improvements, penalizes local
        worsening more strongly, and adds a small magnitude regularizer.
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
            "loss_safe_spectral": total,
            "loss_safe_spectral_improvement": loss_improvement,
            "loss_safe_spectral_safety": loss_safety,
            "loss_safe_spectral_magnitude": loss_magnitude,
            "safe_spectral_helpful_fraction": helpful,
            "safe_spectral_harmful_fraction": harmful,
        }