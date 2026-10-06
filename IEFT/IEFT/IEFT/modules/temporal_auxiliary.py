"""Lightweight temporal fusion blocks for cached LEVIR auxiliary data.

The blocks in this module deliberately depend only on PyTorch.  In particular,
they do not import the training module, timm, Lightning, raster libraries, or
network clients.  This keeps the model-side contract independently testable and
ensures that auxiliary data access remains a dataset/preparation concern.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


def _availability_mask(
    value: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Return a finite availability tensor with shape ``[B, 1]``."""

    if value is None:
        return torch.ones((batch_size, 1), device=device, dtype=dtype)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    value = value.to(device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    elif value.ndim > 2:
        value = value.reshape(value.shape[0], -1)
    if value.shape != (batch_size, 1):
        raise ValueError(
            f"{name} must be scalar, [B], or [B,1]; got {tuple(value.shape)} "
            f"for batch size {batch_size}."
        )
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return value.clamp(0.0, 1.0)


class PairedOSMFusion(nn.Module):
    """Fuse two OSM embeddings while preserving temporal direction.

    The date-specific embeddings are expected to come from the *same* encoder.
    Missing dates are zeroed before comparison, and differences are only formed
    where both dates are available.  Availability indicators are part of the
    learned input, so an available-but-empty OSM snapshot remains distinct from
    an unavailable snapshot.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        residual_scale_init: float = 0.01,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale_init = float(residual_scale_init)
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")

        # Four temporal feature groups plus the two explicit availability bits.
        self.fusion = nn.Sequential(
            nn.Linear(4 * self.feature_dim + 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))

    def reset_residual_parameters(self) -> None:
        """Restore the intentionally small initial auxiliary contribution."""

        with torch.no_grad():
            self.residual_scale.fill_(self.residual_scale_init)

    def forward(
        self,
        z_t1: torch.Tensor,
        z_t2: torch.Tensor,
        has_t1: torch.Tensor,
        has_t2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if z_t1.ndim != 2 or z_t2.ndim != 2:
            raise ValueError(
                "Paired OSM embeddings must have shape [B,D], got "
                f"{tuple(z_t1.shape)} and {tuple(z_t2.shape)}."
            )
        if z_t1.shape != z_t2.shape or z_t1.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected matching [B,{self.feature_dim}] embeddings, got "
                f"{tuple(z_t1.shape)} and {tuple(z_t2.shape)}."
            )

        batch_size = z_t1.shape[0]
        device = z_t1.device
        dtype = z_t1.dtype
        z_t1 = torch.nan_to_num(z_t1, nan=0.0, posinf=0.0, neginf=0.0)
        z_t2 = torch.nan_to_num(z_t2.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        mask_t1 = _availability_mask(has_t1, batch_size, device, dtype, "has_t1")
        mask_t2 = _availability_mask(has_t2, batch_size, device, dtype, "has_t2")

        z_t1_valid = z_t1 * mask_t1
        z_t2_valid = z_t2 * mask_t2
        pair_available = mask_t1 * mask_t2
        any_available = torch.maximum(mask_t1, mask_t2)

        signed_delta = (z_t2_valid - z_t1_valid) * pair_available
        absolute_delta = signed_delta.abs()
        fusion_input = torch.cat(
            [
                z_t1_valid,
                z_t2_valid,
                absolute_delta,
                signed_delta,
                mask_t1,
                mask_t2,
            ],
            dim=-1,
        )
        fused = self.fusion(fusion_input)
        # Mask after affine layers: learned biases can never turn a completely
        # unavailable OSM pair into a real signal.
        fused = fused * any_available * self.residual_scale

        return {
            "fused": fused,
            "z_t1": z_t1_valid,
            "z_t2": z_t2_valid,
            "absolute_delta": absolute_delta,
            "signed_delta": signed_delta,
            "has_t1": mask_t1,
            "has_t2": mask_t2,
            "pair_available": pair_available,
            "any_available": any_available,
        }


class TemporalSpectralAdapter(nn.Module):
    """Patch-scale adapter for paired physical/index spectral rasters.

    Each date contains Green/Red/NIR surface reflectance followed by NDVI and
    McFeeters NDWI.  Input values and masks remain date-specific.  A shared
    encoder produces comparable T1/T2 features, after which absolute and signed
    ``T2 - T1`` differences are fused.  The injection projections are exactly
    zero-initialized, so enabling the adapter preserves the v20d RGB behaviour
    until the new branch receives training updates.
    """

    def __init__(
        self,
        output_dim: int,
        grid_size: int,
        hidden_dim: int = 32,
        num_indices: int = 5,
        num_channels: Optional[int] = None,
        residual_scale_init: float = 0.01,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        self.hidden_dim = int(hidden_dim)
        self.num_channels = int(num_indices if num_channels is None else num_channels)
        # Retain the historical attribute name so auxiliary-checkpoint reports
        # and external diagnostics do not fail merely because the input now
        # contains physical bands in addition to indices.
        self.num_indices = self.num_channels
        self.residual_scale_init = float(residual_scale_init)
        if min(self.output_dim, self.grid_size, self.hidden_dim, self.num_channels) <= 0:
            raise ValueError("output_dim, grid_size, hidden_dim, and num_channels must be positive")

        encoder_in = 2 * self.num_channels  # masked values plus per-channel validity
        groups = _group_count(self.hidden_dim)
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(encoder_in, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.date_projection = nn.Conv2d(self.hidden_dim, self.output_dim, 1, bias=False)
        self.temporal_projection = nn.Sequential(
            nn.Conv2d(4 * self.hidden_dim + 2, self.hidden_dim * 2, 1, bias=False),
            nn.GroupNorm(_group_count(self.hidden_dim * 2), self.hidden_dim * 2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim * 2, self.output_dim, 1, bias=False),
        )

        # Non-zero gates let the zero output projections learn on the first
        # optimization step; zeroing both a projection and its gate would make
        # the branch unable to leave the origin.
        self.date_residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self.temporal_residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self.reset_residual_parameters()

    def reset_residual_parameters(self) -> None:
        """Restore exact zero injection after the model's generic initializer."""

        with torch.no_grad():
            self.date_residual_scale.fill_(self.residual_scale_init)
            self.temporal_residual_scale.fill_(self.residual_scale_init)
            self.date_projection.weight.zero_()
            self.temporal_projection[-1].weight.zero_()

    def _validate_date_inputs(
        self,
        indices: torch.Tensor,
        valid: torch.Tensor,
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(indices) or not torch.is_tensor(valid):
            raise TypeError(f"{name} indices and validity mask must be torch tensors")
        if indices.ndim != 4:
            raise ValueError(f"{name} indices must have shape [B,C,H,W], got {tuple(indices.shape)}")
        if indices.shape[1] != self.num_channels:
            raise ValueError(
                f"{name} must contain {self.num_channels} spectral channels, "
                f"got {indices.shape[1]}"
            )
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.ndim != 4 or valid.shape[0] != indices.shape[0] or valid.shape[-2:] != indices.shape[-2:]:
            raise ValueError(
                f"{name} validity must be [B,1,H,W] or [B,C,H,W], got {tuple(valid.shape)} "
                f"for indices {tuple(indices.shape)}"
            )
        if valid.shape[1] == 1:
            valid = valid.expand(-1, self.num_channels, -1, -1)
        elif valid.shape[1] != self.num_channels:
            raise ValueError(
                f"{name} validity must have 1 or {self.num_channels} channels, "
                f"got {valid.shape[1]}"
            )

        indices = indices.float()
        valid = valid.to(device=indices.device, dtype=indices.dtype)
        # Scientific range/formula validation belongs to the cache loader,
        # before optional train-only standardization.  Physical reflectance and
        # standardized channels must not be clamped here.
        indices = torch.nan_to_num(indices, nan=0.0, posinf=0.0, neginf=0.0)
        valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        return indices, valid

    def _encode_date(
        self,
        indices: torch.Tensor,
        valid: torch.Tensor,
        availability: torch.Tensor,
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, valid = self._validate_date_inputs(indices, valid, name)
        batch_size = indices.shape[0]
        has_date = _availability_mask(
            availability,
            batch_size,
            indices.device,
            indices.dtype,
            f"has_{name}",
        )
        has_map = has_date.view(batch_size, 1, 1, 1)
        valid = valid * has_map

        # Average values over valid pixels only.  Coverage remains an explicit
        # channel, so a true index value of zero is never confused with nodata.
        pooled_valid = F.adaptive_avg_pool2d(valid, (self.grid_size, self.grid_size))
        pooled_sum = F.adaptive_avg_pool2d(indices * valid, (self.grid_size, self.grid_size))
        pooled_values = pooled_sum / pooled_valid.clamp_min(1.0e-6)
        pooled_values = torch.where(pooled_valid > 0, pooled_values, torch.zeros_like(pooled_values))

        coverage = pooled_valid.mean(dim=1, keepdim=True)
        encoded = self.shared_encoder(torch.cat([pooled_values, pooled_valid], dim=1))
        # Apply coverage after GroupNorm/affine layers for exact zero output in
        # invalid cells, including the all-invalid case.
        encoded = encoded * coverage
        return encoded, coverage, has_date

    def forward(
        self,
        indices_t1: torch.Tensor,
        indices_t2: torch.Tensor,
        valid_t1: torch.Tensor,
        valid_t2: torch.Tensor,
        has_t1: torch.Tensor,
        has_t2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_t1, coverage_t1, has_t1_mask = self._encode_date(
            indices_t1, valid_t1, has_t1, "spectral_t1"
        )
        z_t2, coverage_t2, has_t2_mask = self._encode_date(
            indices_t2, valid_t2, has_t2, "spectral_t2"
        )
        if z_t1.shape != z_t2.shape:
            raise ValueError(f"T1/T2 spectral encodings differ: {tuple(z_t1.shape)} vs {tuple(z_t2.shape)}")

        pair_coverage = torch.minimum(coverage_t1, coverage_t2)
        signed_delta = (z_t2 - z_t1) * pair_coverage
        absolute_delta = signed_delta.abs()
        temporal_input = torch.cat(
            [
                z_t1,
                z_t2,
                absolute_delta,
                signed_delta,
                coverage_t1,
                coverage_t2,
            ],
            dim=1,
        )

        spatial_t1 = self.date_projection(z_t1) * coverage_t1 * self.date_residual_scale
        spatial_t2 = self.date_projection(z_t2) * coverage_t2 * self.date_residual_scale
        temporal_spatial = self.temporal_projection(temporal_input)
        temporal_spatial = temporal_spatial * pair_coverage * self.temporal_residual_scale

        # Weighted pooling avoids treating invalid grid cells as measurements.
        denom = pair_coverage.sum(dim=(2, 3)).clamp_min(1.0e-6)
        pooled_temporal = temporal_spatial.sum(dim=(2, 3)) / denom
        pair_available = (pair_coverage.flatten(1).amax(dim=1, keepdim=True) > 0).to(z_t1.dtype)
        any_available = torch.maximum(has_t1_mask, has_t2_mask)
        pooled_temporal = pooled_temporal * pair_available

        return {
            "spatial_t1": spatial_t1,
            "spatial_t2": spatial_t2,
            "temporal_spatial": temporal_spatial,
            "pooled_temporal": pooled_temporal,
            "z_t1": z_t1,
            "z_t2": z_t2,
            "absolute_delta": absolute_delta,
            "signed_delta": signed_delta,
            "coverage_t1": coverage_t1,
            "coverage_t2": coverage_t2,
            "has_t1": has_t1_mask,
            "has_t2": has_t2_mask,
            "pair_available": pair_available,
            "any_available": any_available,
        }


class TemporalIndexAdapter(nn.Module):
    """Checkpoint-safe coarse Landsat NDVI/NDWI temporal residual.

    The transport tensor may be canonical five-channel
    ``[Green, Red, NIR, NDVI, NDWI]`` or the retained two-channel index cache.
    Only genuine cached indices enter this adapter.  It explicitly constructs
    ``NDVI_T1, NDVI_T2, dNDVI, NDWI_T1, NDWI_T2, dNDWI`` and never derives an
    index from LEVIR RGB.  Sensor family and reliability remain explicit.
    """

    def __init__(
        self,
        output_dim: int,
        grid_size: int,
        hidden_dim: int = 32,
        input_channels: int = 5,
        num_sensor_ids: int = 4,
        residual_scale_init: float = 0.01,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        self.hidden_dim = int(hidden_dim)
        self.input_channels = int(input_channels)
        self.num_channels = self.input_channels
        self.num_indices = 2
        self.num_sensor_ids = int(num_sensor_ids)
        self.residual_scale_init = float(residual_scale_init)
        if self.input_channels not in {2, 5}:
            raise ValueError("TemporalIndexAdapter input_channels must be 2 or 5")

        groups = _group_count(self.hidden_dim)
        # Two values plus two explicit validity channels for each date.
        self.shared_date_encoder = nn.Sequential(
            nn.Conv2d(4, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        # Six temporal values plus their six validity channels.
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(12, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.sensor_embedding = nn.Embedding(
            self.num_sensor_ids,
            self.hidden_dim,
            padding_idx=0,
        )
        self.date_projection = nn.Conv2d(self.hidden_dim, self.output_dim, 1, bias=False)
        temporal_in = 5 * self.hidden_dim + 3
        self.temporal_projection = nn.Sequential(
            nn.Conv2d(temporal_in, self.hidden_dim * 2, 1, bias=False),
            nn.GroupNorm(_group_count(self.hidden_dim * 2), self.hidden_dim * 2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim * 2, self.output_dim, 1, bias=False),
        )
        self.date_residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self.temporal_residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self.reset_residual_parameters()

    def reset_residual_parameters(self) -> None:
        with torch.no_grad():
            self.date_residual_scale.fill_(self.residual_scale_init)
            self.temporal_residual_scale.fill_(self.residual_scale_init)
            self.sensor_embedding.weight[0].zero_()
            self.date_projection.weight.zero_()
            self.temporal_projection[-1].weight.zero_()

    def _indices_and_valid(
        self, channels: torch.Tensor, valid: torch.Tensor, name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(channels) or channels.ndim != 4:
            raise ValueError(f"{name} channels must be [B,C,H,W]")
        if channels.shape[1] != self.input_channels:
            raise ValueError(
                f"{name} expected {self.input_channels} transport channels, "
                f"got {channels.shape[1]}"
            )
        indices = channels[:, 3:5] if self.input_channels == 5 else channels
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.ndim != 4 or valid.shape[0] != indices.shape[0] or valid.shape[-2:] != indices.shape[-2:]:
            raise ValueError(
                f"{name} validity must align with indices, got {tuple(valid.shape)} "
                f"and {tuple(indices.shape)}"
            )
        if valid.shape[1] == 1:
            valid = valid.expand(-1, 2, -1, -1)
        elif valid.shape[1] == self.input_channels and self.input_channels == 5:
            valid = valid[:, 3:5]
        elif valid.shape[1] != 2:
            raise ValueError(f"{name} validity must have 1, 2, or transport-channel count")
        indices = torch.nan_to_num(indices.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = torch.nan_to_num(valid.to(indices).float(), nan=0.0).clamp(0.0, 1.0)
        return indices, valid

    def _pool(self, values: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled_valid = F.adaptive_avg_pool2d(valid, (self.grid_size, self.grid_size))
        pooled_sum = F.adaptive_avg_pool2d(values * valid, (self.grid_size, self.grid_size))
        pooled = pooled_sum / pooled_valid.clamp_min(1.0e-6)
        pooled = torch.where(pooled_valid > 0, pooled, torch.zeros_like(pooled))
        return pooled, pooled_valid

    def _sensor_ids(self, value: Optional[torch.Tensor], batch: int, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch, dtype=torch.long, device=device)
        value = torch.as_tensor(value, device=device).long().reshape(-1)
        if value.numel() == 1 and batch > 1:
            value = value.expand(batch)
        if value.numel() != batch or torch.any(value < 0) or torch.any(value >= self.num_sensor_ids):
            raise ValueError(
                f"sensor IDs must contain {batch} values in [0,{self.num_sensor_ids - 1}]"
            )
        return value

    def forward(
        self,
        channels_t1: torch.Tensor,
        channels_t2: torch.Tensor,
        valid_t1: torch.Tensor,
        valid_t2: torch.Tensor,
        has_t1: torch.Tensor,
        has_t2: torch.Tensor,
        sensor_id_t1: Optional[torch.Tensor] = None,
        sensor_id_t2: Optional[torch.Tensor] = None,
        reliability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        indices_t1, valid_t1 = self._indices_and_valid(channels_t1, valid_t1, "spectral_t1")
        indices_t2, valid_t2 = self._indices_and_valid(channels_t2, valid_t2, "spectral_t2")
        batch = indices_t1.shape[0]
        if indices_t2.shape[0] != batch:
            raise ValueError("T1/T2 spectral batch sizes differ")
        has_t1_mask = _availability_mask(has_t1, batch, indices_t1.device, indices_t1.dtype, "has_spectral_t1")
        has_t2_mask = _availability_mask(has_t2, batch, indices_t1.device, indices_t1.dtype, "has_spectral_t2")
        valid_t1 = valid_t1 * has_t1_mask.view(batch, 1, 1, 1)
        valid_t2 = valid_t2 * has_t2_mask.view(batch, 1, 1, 1)
        pooled_t1, pooled_valid_t1 = self._pool(indices_t1, valid_t1)
        pooled_t2, pooled_valid_t2 = self._pool(indices_t2, valid_t2)

        coverage_t1 = pooled_valid_t1.mean(dim=1, keepdim=True)
        coverage_t2 = pooled_valid_t2.mean(dim=1, keepdim=True)
        z_t1 = self.shared_date_encoder(torch.cat([pooled_t1, pooled_valid_t1], dim=1))
        z_t2 = self.shared_date_encoder(torch.cat([pooled_t2, pooled_valid_t2], dim=1))
        sid_t1 = self._sensor_ids(sensor_id_t1, batch, indices_t1.device)
        sid_t2 = self._sensor_ids(sensor_id_t2, batch, indices_t1.device)
        z_t1 = (z_t1 + self.sensor_embedding(sid_t1).view(batch, -1, 1, 1)) * coverage_t1
        z_t2 = (z_t2 + self.sensor_embedding(sid_t2).view(batch, -1, 1, 1)) * coverage_t2

        pair_valid = torch.minimum(pooled_valid_t1, pooled_valid_t2)
        # Required order: NDVI_T1, NDVI_T2, dNDVI, NDWI_T1,
        # NDWI_T2, dNDWI.  Pair deltas are zero outside joint validity.
        raw_temporal = torch.cat(
            [
                pooled_t1[:, 0:1],
                pooled_t2[:, 0:1],
                (pooled_t2[:, 0:1] - pooled_t1[:, 0:1]) * pair_valid[:, 0:1],
                pooled_t1[:, 1:2],
                pooled_t2[:, 1:2],
                (pooled_t2[:, 1:2] - pooled_t1[:, 1:2]) * pair_valid[:, 1:2],
            ],
            dim=1,
        )
        raw_valid = torch.cat(
            [
                pooled_valid_t1[:, 0:1],
                pooled_valid_t2[:, 0:1],
                pair_valid[:, 0:1],
                pooled_valid_t1[:, 1:2],
                pooled_valid_t2[:, 1:2],
                pair_valid[:, 1:2],
            ],
            dim=1,
        )
        z_temporal = self.temporal_encoder(torch.cat([raw_temporal, raw_valid], dim=1))
        pair_coverage = pair_valid.mean(dim=1, keepdim=True)
        signed_delta = (z_t2 - z_t1) * pair_coverage
        absolute_delta = signed_delta.abs()

        if reliability is None:
            reliability_mask = torch.minimum(has_t1_mask, has_t2_mask)
        else:
            reliability_mask = _availability_mask(
                reliability,
                batch,
                indices_t1.device,
                indices_t1.dtype,
                "spectral_reliability",
            )
        reliability_map = reliability_mask.view(batch, 1, 1, 1)
        temporal_input = torch.cat(
            [
                z_t1,
                z_t2,
                absolute_delta,
                signed_delta,
                z_temporal,
                coverage_t1,
                coverage_t2,
                reliability_map.expand(-1, -1, self.grid_size, self.grid_size),
            ],
            dim=1,
        )
        spatial_t1 = self.date_projection(z_t1) * coverage_t1 * reliability_map * self.date_residual_scale
        spatial_t2 = self.date_projection(z_t2) * coverage_t2 * reliability_map * self.date_residual_scale
        temporal_spatial = self.temporal_projection(temporal_input)
        temporal_spatial = (
            temporal_spatial
            * pair_coverage
            * reliability_map
            * self.temporal_residual_scale
        )
        denom = (pair_coverage * reliability_map).sum(dim=(2, 3)).clamp_min(1.0e-6)
        pooled_temporal = temporal_spatial.sum(dim=(2, 3)) / denom
        pair_available = (
            (pair_coverage.flatten(1).amax(dim=1, keepdim=True) > 0).to(indices_t1.dtype)
            * (reliability_mask > 0).to(indices_t1.dtype)
        )
        pooled_temporal = pooled_temporal * pair_available
        return {
            "spatial_t1": spatial_t1,
            "spatial_t2": spatial_t2,
            "temporal_spatial": temporal_spatial,
            "pooled_temporal": pooled_temporal,
            "z_t1": z_t1,
            "z_t2": z_t2,
            "absolute_delta": absolute_delta,
            "signed_delta": signed_delta,
            "coverage_t1": coverage_t1,
            "coverage_t2": coverage_t2,
            "has_t1": has_t1_mask,
            "has_t2": has_t2_mask,
            "pair_available": pair_available,
            "any_available": torch.maximum(has_t1_mask, has_t2_mask),
            "spectral_reliability": reliability_mask,
            "temporal_index_context": raw_temporal,
        }


class OSMT2EarlyAdapter(nn.Module):
    """Five-channel T2 OSM residual aligned to the ViT 16x16 token grid."""

    def __init__(
        self,
        output_dim: int,
        grid_size: int,
        struct_dim: int = 16,
        hidden_dim: int = 32,
        map_channels: int = 5,
        residual_scale_init: float = 0.01,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        self.struct_dim = int(struct_dim)
        self.hidden_dim = int(hidden_dim)
        self.map_channels = int(map_channels)
        self.residual_scale_init = float(residual_scale_init)
        groups = _group_count(self.hidden_dim)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(self.map_channels, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden_dim),
            nn.GELU(),
        )
        self.struct_projection = nn.Sequential(
            nn.Linear(self.struct_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.output_projection = nn.Conv2d(self.hidden_dim, self.output_dim, 1, bias=True)
        self.residual_scale = nn.Parameter(torch.tensor(self.residual_scale_init))
        self.reset_residual_parameters()

    def reset_residual_parameters(self) -> None:
        with torch.no_grad():
            self.residual_scale.fill_(self.residual_scale_init)
            self.output_projection.weight.zero_()
            self.output_projection.bias.zero_()

    def forward(
        self,
        osm_maps: torch.Tensor,
        osm_t2_struct: torch.Tensor,
        osm_reliability: torch.Tensor,
        has_osm_t2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if osm_maps.ndim != 4 or osm_maps.shape[1] != self.map_channels:
            raise ValueError(
                f"osm_maps must be [B,{self.map_channels},H,W], got {tuple(osm_maps.shape)}"
            )
        batch = osm_maps.shape[0]
        if osm_t2_struct.ndim == 1:
            osm_t2_struct = osm_t2_struct.unsqueeze(0)
        if osm_t2_struct.shape != (batch, self.struct_dim):
            raise ValueError(
                f"osm_t2_struct must be [B,{self.struct_dim}], got {tuple(osm_t2_struct.shape)}"
            )
        reliability = _availability_mask(
            osm_reliability,
            batch,
            osm_maps.device,
            osm_maps.dtype,
            "osm_reliability",
        )
        if has_osm_t2 is not None:
            reliability = reliability * _availability_mask(
                has_osm_t2,
                batch,
                osm_maps.device,
                osm_maps.dtype,
                "has_osm_t2",
            )
        maps = torch.nan_to_num(osm_maps.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        maps = F.adaptive_max_pool2d(maps, (self.grid_size, self.grid_size))
        encoded = self.map_encoder(maps)
        struct = self.struct_projection(osm_t2_struct.float()).view(batch, -1, 1, 1)
        fused = encoded + struct
        residual = self.output_projection(fused) * self.residual_scale
        residual = residual * reliability.view(batch, 1, 1, 1)
        return residual.flatten(2).transpose(1, 2)
