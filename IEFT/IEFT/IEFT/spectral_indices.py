"""Scientific helpers and offline providers for LEVIR spectral preparation.

The canonical cache keeps the physical Green/Red/NIR surface-reflectance
measurements as well as the two indices derived from them.  Keeping those five
channels together makes the source data auditable and prevents the runtime
adapter from receiving indices whose physical inputs have been discarded.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests

from IEFT.levir_metadata import (
    LEVIRValidationError,
    atomic_write_json,
    calendar_month_window,
    non_overlapping_windows,
    path_relative_to,
    sha256_file,
)


SPECTRAL_SCHEMA_VERSION = "levir-spectral-reflectance-indices-v2"
SURFACE_REFLECTANCE_ORDER = ("green", "red", "nir")
INDEX_ORDER = ("ndvi", "ndwi")
SPECTRAL_CHANNEL_ORDER = SURFACE_REFLECTANCE_ORDER + (
    "ndvi",
    "ndwi_mcfeeters",
)
NDWI_VARIANT = "mcfeeters"
FORMULAE = {
    "ndvi": "(nir - red) / (nir + red)",
    "ndwi": "(green - nir) / (green + nir)",
}
DENOMINATOR_POLICY = (
    "divide only where inputs and denominator are finite and "
    "abs(denominator) > epsilon; do not add epsilon to a valid denominator"
)
LANDSAT_SCALE = 2.75e-5
LANDSAT_OFFSET = -0.2
LANDSAT_NATIVE_RESOLUTION_M = 30
SURFACE_REFLECTANCE_VALID_RANGE = (-0.2, 1.6)
LANDSAT_7_SLC_FAILURE_DATE = date(2003, 5, 31)
EARTH_ENGINE_REGISTRATION_URL = "https://code.earthengine.google.com/register?project={project}"


LANDSAT_SENSORS: Dict[str, Dict[str, Any]] = {
    "LANDSAT_5": {
        "collection": "LANDSAT/LT05/C02/T1_L2",
        "sensor": "TM",
        "bands": {"green": "SR_B2", "red": "SR_B3", "nir": "SR_B4"},
        "qa_pixel_mask_bits": [0, 1, 3, 4, 5],
        "qa_radsat_bits": [1, 2, 3, 9],
    },
    "LANDSAT_7": {
        "collection": "LANDSAT/LE07/C02/T1_L2",
        "sensor": "ETM+",
        "bands": {"green": "SR_B2", "red": "SR_B3", "nir": "SR_B4"},
        "qa_pixel_mask_bits": [0, 1, 3, 4, 5],
        "qa_radsat_bits": [1, 2, 3, 9],
    },
    "LANDSAT_8": {
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "sensor": "OLI/TIRS",
        "bands": {"green": "SR_B3", "red": "SR_B4", "nir": "SR_B5"},
        "qa_pixel_mask_bits": [0, 1, 2, 3, 4, 5],
        "qa_radsat_bits": [2, 3, 4],
    },
}


# LEVIR-CD spans 2002--2018. Landsat 7 is the one Collection 2 Level-2
# sensor family that covers every published LEVIR month, so the production
# default deliberately uses it for both dates instead of silently mixing
# instruments with different spectral response functions. The scan-line
# corrector failure is handled by QA masking, compositing, and the documented
# temporal-window expansion policy.
LANDSAT_SENSOR_POLICIES: Dict[str, Tuple[str, ...]] = {
    "landsat7_consistent": ("LANDSAT_7",),
}
LANDSAT_SENSOR_POLICY_DESCRIPTIONS: Dict[str, str] = {
    "landsat7_consistent": (
        "Use Landsat 7 ETM+ Collection 2 Tier 1 Level-2 surface reflectance "
        "for both T1 and T2; do not mix sensors and apply no cross-sensor "
        "empirical harmonization."
    ),
}


def sensors_for_policy(sensor_policy: str) -> Tuple[str, ...]:
    """Return the deterministic spacecraft selection for an EE policy."""

    try:
        return LANDSAT_SENSOR_POLICIES[str(sensor_policy)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Landsat sensor policy {sensor_policy!r}; expected one of "
            f"{sorted(LANDSAT_SENSOR_POLICIES)}"
        ) from exc


ACCEPTABLE_STATUSES = {"ok", "ok_partial"}
FAILED_STATUSES = {
    "insufficient_valid_pixels",
    "source_unavailable",
    "authentication_required",
    "failed_retryable",
    "failed_permanent",
}


class ProviderUnavailable(RuntimeError):
    pass


class AuthenticationRequired(RuntimeError):
    pass


def earth_engine_access_guidance(project: str, original_error: BaseException | str) -> str:
    """Return actionable, token-free guidance for Earth Engine access failures."""

    project = str(project or "YOUR_GOOGLE_CLOUD_PROJECT").strip()
    registration_url = EARTH_ENGINE_REGISTRATION_URL.format(project=project)
    return (
        f"Earth Engine initialization failed for project {project}. "
        "Complete these user-only access steps, then retry: "
        f"`gcloud services enable earthengine.googleapis.com --project={project}`; "
        "`earthengine authenticate`; "
        f"`earthengine set_project {project}`. "
        f"If the project is not registered for Earth Engine, register it at {registration_url}. "
        f"Original error: {original_error}"
    )


def landsat7_slc_mode(acquisition_date: Optional[str]) -> Optional[str]:
    """Classify a Landsat-7 acquisition as SLC-on/off from its UTC date."""

    if not acquisition_date:
        return None
    parsed = date.fromisoformat(str(acquisition_date)[:10])
    return "off" if parsed >= LANDSAT_7_SLC_FAILURE_DATE else "on"


def apply_scale_offset(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.float32(scale) + np.float32(offset)


def _safe_normalized_difference(
    first: np.ndarray,
    second: np.ndarray,
    valid: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    gross_tolerance: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"Normalized-difference shapes differ: {first.shape}, {second.shape}")
    denominator = first + second
    mask = np.isfinite(first) & np.isfinite(second) & np.isfinite(denominator)
    mask &= np.abs(denominator) > float(epsilon)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    output = np.zeros(first.shape, dtype=np.float32)
    # The epsilon is a validity threshold, not an additive bias.  This also
    # preserves the intended behaviour for finite negative reflectance values,
    # unlike Earth Engine's normalizedDifference convenience method.
    np.divide(first - second, denominator, out=output, where=mask)
    mask &= np.isfinite(output) & (np.abs(output) <= float(gross_tolerance))
    output = np.where(mask, np.clip(output, -1.0, 1.0), 0.0).astype(np.float32)
    return output, mask


def compute_ndvi_ndwi(
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    valid: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return float32 [NDVI, McFeeters NDWI] and a per-band valid mask."""

    ndvi, ndvi_valid = _safe_normalized_difference(nir, red, valid, epsilon)
    ndwi, ndwi_valid = _safe_normalized_difference(green, nir, valid, epsilon)
    indices = np.stack([ndvi, ndwi], axis=0).astype(np.float32)
    masks = np.stack([ndvi_valid, ndwi_valid], axis=0).astype(bool)
    return indices, masks


def compose_spectral_channels(
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    valid: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return canonical ``[Green, Red, NIR, NDVI, NDWI]`` and validity.

    Surface reflectance is deliberately retained in physical scaled units; it
    is never reconstructed from the LEVIR RGB images.  One common validity mask
    is used for all five channels so that every runtime pixel is a coherent
    multispectral observation and both indices can be verified from the cached
    physical bands.
    """

    physical = np.stack(
        [
            np.asarray(green, dtype=np.float32),
            np.asarray(red, dtype=np.float32),
            np.asarray(nir, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if physical.shape[1:] != physical[0].shape:
        raise ValueError(f"Surface-reflectance band shapes differ: {physical.shape}")
    base_valid = np.isfinite(physical).all(axis=0)
    if valid is not None:
        supplied_valid = np.asarray(valid, dtype=bool)
        if supplied_valid.ndim == 3:
            supplied_valid = supplied_valid.all(axis=0)
        if supplied_valid.shape != physical.shape[1:]:
            raise ValueError(
                f"Surface-reflectance validity shape {supplied_valid.shape} does not match "
                f"bands {physical.shape[1:]}"
            )
        base_valid &= supplied_valid
    indices, index_valid = compute_ndvi_ndwi(
        physical[0], physical[1], physical[2], base_valid, epsilon
    )
    common_valid = base_valid & index_valid.all(axis=0)
    channels = np.concatenate([physical, indices], axis=0).astype(np.float32, copy=False)
    channels[:, ~common_valid] = 0.0
    return channels, common_valid


def validate_spectral_arrays(
    channels: np.ndarray,
    valid: np.ndarray,
    *,
    epsilon: float = 1e-6,
    index_tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Validate a canonical five-channel cache, including formula consistency."""

    channels = np.asarray(channels)
    if channels.ndim != 3 or channels.shape[0] != len(SPECTRAL_CHANNEL_ORDER):
        raise ValueError(
            "Spectral raster must have shape [5,H,W] in Green/Red/NIR/NDVI/NDWI "
            f"order, got {channels.shape}"
        )
    valid_array = np.asarray(valid)
    if valid_array.ndim == 3:
        valid_array = valid_array.all(axis=0)
    if valid_array.shape != channels.shape[1:]:
        raise ValueError(
            f"Spectral validity shape {valid_array.shape} does not match {channels.shape[1:]}"
        )
    valid_bool = valid_array.astype(bool)
    if not np.isfinite(channels[:, valid_bool]).all():
        raise ValueError("Valid spectral pixels contain NaN or infinity")

    index_report = validate_index_arrays(channels[3:5], valid_bool)
    recomputed, recomputed_valid = compute_ndvi_ndwi(
        channels[0], channels[1], channels[2], valid_bool, epsilon
    )
    if np.any(valid_bool & ~recomputed_valid.all(axis=0)):
        raise ValueError("Valid spectral pixels violate the normalized-difference denominator policy")
    if valid_bool.any():
        maximum_formula_error = float(
            np.max(np.abs(recomputed[:, valid_bool] - channels[3:5, valid_bool]))
        )
        if maximum_formula_error > float(index_tolerance):
            raise ValueError(
                "Cached NDVI/NDWI do not match Green/Red/NIR: maximum absolute "
                f"error {maximum_formula_error:.6g} exceeds {index_tolerance:.6g}"
            )
        physical_min = [float(channels[i, valid_bool].min()) for i in range(3)]
        physical_max = [float(channels[i, valid_bool].max()) for i in range(3)]
        allowed_min, allowed_max = SURFACE_REFLECTANCE_VALID_RANGE
        if min(physical_min) < allowed_min - 1e-3 or max(physical_max) > allowed_max + 1e-3:
            raise ValueError(
                "Valid surface reflectance lies outside the scaled physical range "
                f"{SURFACE_REFLECTANCE_VALID_RANGE}: min={min(physical_min):.6g}, "
                f"max={max(physical_max):.6g}"
            )
    else:
        maximum_formula_error = None
        physical_min = [None, None, None]
        physical_max = [None, None, None]
    invalid_nonzero = int(np.count_nonzero(channels[:, ~valid_bool]))
    return {
        **index_report,
        "shape": list(channels.shape),
        "channel_order": list(SPECTRAL_CHANNEL_ORDER),
        "per_band_valid_fraction": [float(valid_bool.mean())] * len(SPECTRAL_CHANNEL_ORDER),
        "index_valid_min": index_report["valid_min"],
        "index_valid_max": index_report["valid_max"],
        "surface_reflectance_order": list(SURFACE_REFLECTANCE_ORDER),
        "surface_reflectance_valid_min": physical_min,
        "surface_reflectance_valid_max": physical_max,
        "maximum_formula_error": maximum_formula_error,
        "invalid_nonzero_count": invalid_nonzero,
    }


def landsat_qa_valid_mask(
    qa_pixel: np.ndarray, qa_radsat: np.ndarray, spacecraft: str
) -> np.ndarray:
    """Apply documented Landsat C2 fill/cloud/shadow/snow/cirrus/saturation bits."""

    if spacecraft not in LANDSAT_SENSORS:
        raise ValueError(f"Unsupported spacecraft {spacecraft!r}")
    qa_pixel = np.asarray(qa_pixel, dtype=np.uint16)
    qa_radsat = np.asarray(qa_radsat, dtype=np.uint16)
    if qa_pixel.shape != qa_radsat.shape:
        raise ValueError("QA_PIXEL and QA_RADSAT shapes differ")
    invalid_pixel = np.zeros(qa_pixel.shape, dtype=bool)
    for bit in LANDSAT_SENSORS[spacecraft]["qa_pixel_mask_bits"]:
        invalid_pixel |= (qa_pixel & np.uint16(1 << bit)) != 0
    saturated = np.zeros(qa_radsat.shape, dtype=bool)
    for bit in LANDSAT_SENSORS[spacecraft]["qa_radsat_bits"]:
        saturated |= (qa_radsat & np.uint16(1 << bit)) != 0
    return ~(invalid_pixel | saturated)


def deterministic_nanmedian(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(stack, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("Composite stack needs a scene dimension")
    observations = np.isfinite(values).sum(axis=0).astype(np.uint16)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(values, axis=0).astype(np.float32)
    composite[observations == 0] = np.nan
    return composite, observations


def validate_index_arrays(
    indices: np.ndarray, valid: np.ndarray, tolerance: float = 1e-5
) -> Dict[str, Any]:
    indices = np.asarray(indices)
    valid = np.asarray(valid)
    if indices.ndim != 3 or indices.shape[0] != 2:
        raise ValueError(f"Index raster must have shape [2,H,W], got {indices.shape}")
    if valid.ndim == 2:
        valid = np.broadcast_to(valid[None], indices.shape)
    elif valid.ndim == 3 and valid.shape[0] == 1:
        valid = np.broadcast_to(valid, indices.shape)
    if valid.shape != indices.shape:
        raise ValueError(f"Validity shape {valid.shape} is not broadcastable to {indices.shape}")
    valid_bool = valid.astype(bool)
    if not np.isfinite(indices[valid_bool]).all():
        raise ValueError("Valid index pixels contain NaN or infinity")
    if valid_bool.any():
        minimum = float(indices[valid_bool].min())
        maximum = float(indices[valid_bool].max())
        if minimum < -1.0 - tolerance or maximum > 1.0 + tolerance:
            raise ValueError(f"Valid index range [{minimum}, {maximum}] exceeds [-1,1]")
    else:
        minimum = maximum = None
    invalid_nonzero = int(np.count_nonzero(indices[~valid_bool]))
    return {
        "shape": list(indices.shape),
        "valid_fraction": float(valid_bool.all(axis=0).mean()),
        "invalid_fraction": float(1.0 - valid_bool.all(axis=0).mean()),
        "per_band_valid_fraction": [float(valid_bool[i].mean()) for i in range(2)],
        "valid_min": minimum,
        "valid_max": maximum,
        "invalid_nonzero_count": invalid_nonzero,
    }


def expansion_steps(step_days: int, max_days: int) -> List[int]:
    step = int(step_days)
    maximum = int(max_days)
    if step <= 0 and maximum > 0:
        raise ValueError("expand_step_days must be positive when max_window_days > 0")
    values = [0]
    current = step
    while current <= maximum and current > 0:
        values.append(current)
        current += step
    if maximum > 0 and values[-1] != maximum:
        values.append(maximum)
    return values


def atomic_raster_path(destination: Path) -> Tuple[Path, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=str(destination.parent)
    )
    os.close(handle)
    temporary = Path(name)
    temporary.unlink(missing_ok=True)
    return temporary, destination


def _rasterio():
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject

        return rasterio, Resampling, from_bounds, reproject
    except ImportError as exc:
        raise ProviderUnavailable(
            "rasterio is required for georeferenced spectral preparation; "
            "install requirements-auxiliary.txt"
        ) from exc


def write_spectral_rasters(
    spectral_path: Path,
    mask_path: Path,
    channels: np.ndarray,
    valid: np.ndarray,
    crs: Any,
    transform: Any,
    metadata_tags: Mapping[str, Any],
) -> None:
    """Atomically write the canonical five-channel cache and binary validity."""

    rasterio, _, _, _ = _rasterio()
    channels = np.asarray(channels, dtype=np.float32)
    valid_2d = np.asarray(valid, dtype=bool)
    if valid_2d.ndim == 3:
        valid_2d = valid_2d.all(axis=0)
    validate_spectral_arrays(
        channels,
        valid_2d,
        epsilon=float(metadata_tags.get("epsilon", 1e-6)),
    )
    clean = np.where(valid_2d[None], channels, 0.0).astype(np.float32)
    tags = {
        str(key): json.dumps(value) if isinstance(value, (dict, list, tuple)) else str(value)
        for key, value in metadata_tags.items()
    }
    tags.setdefault("spectral_channel_order", json.dumps(list(SPECTRAL_CHANNEL_ORDER)))
    tags.setdefault("surface_reflectance_units", "scaled unitless surface reflectance")
    tmp_spectral, final_spectral = atomic_raster_path(spectral_path)
    tmp_mask, final_mask = atomic_raster_path(mask_path)
    try:
        with rasterio.open(
            tmp_spectral,
            "w",
            driver="GTiff",
            height=clean.shape[1],
            width=clean.shape[2],
            count=len(SPECTRAL_CHANNEL_ORDER),
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=None,
            compress="deflate",
            predictor=3,
        ) as dataset:
            dataset.write(clean)
            for band, description in enumerate(SPECTRAL_CHANNEL_ORDER, start=1):
                dataset.set_band_description(band, description)
            dataset.update_tags(**tags)
        with rasterio.open(
            tmp_mask,
            "w",
            driver="GTiff",
            height=clean.shape[1],
            width=clean.shape[2],
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            nodata=0,
            compress="deflate",
        ) as dataset:
            dataset.write(valid_2d.astype(np.uint8), 1)
            dataset.set_band_description(1, "valid")
            dataset.update_tags(**tags)
        os.replace(tmp_spectral, final_spectral)
        os.replace(tmp_mask, final_mask)
    finally:
        tmp_spectral.unlink(missing_ok=True)
        tmp_mask.unlink(missing_ok=True)


def align_to_levir_grid(
    source_channels: np.ndarray,
    source_valid: np.ndarray,
    source_crs: Any,
    source_transform: Any,
    bbox: Sequence[float],
    output_shape: Tuple[int, int] = (1024, 1024),
) -> Tuple[np.ndarray, np.ndarray, Any]:
    rasterio, Resampling, from_bounds, reproject = _rasterio()
    del rasterio
    height, width = (int(output_shape[0]), int(output_shape[1]))
    west, south, east, north = (float(value) for value in bbox)
    destination_transform = from_bounds(west, south, east, north, width, height)
    source_channels = np.asarray(source_channels, dtype=np.float32)
    if source_channels.ndim != 3 or source_channels.shape[0] < 1:
        raise ValueError(f"Source raster must have shape [C,H,W], got {source_channels.shape}")
    output = np.zeros((source_channels.shape[0], height, width), dtype=np.float32)
    source_valid = np.asarray(source_valid, dtype=bool)
    if source_valid.ndim == 3:
        source_valid = source_valid.all(axis=0)
    valid_output = np.zeros((height, width), dtype=np.uint8)
    # Mask-aware bilinear reprojection. Reprojecting raw samples would let
    # arbitrary nodata values bleed into neighbouring valid pixels. Instead,
    # interpolate value*valid and valid weights separately, then divide. A
    # nearest-neighbour mask still defines the explicit output validity.
    source_weights = source_valid.astype(np.float32)
    for band in range(source_channels.shape[0]):
        numerator = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        weighted_source = np.where(
            source_valid, source_channels[band], np.float32(0.0)
        ).astype(np.float32, copy=False)
        reproject(
            source=weighted_source,
            destination=numerator,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=destination_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=None,
        )
        reproject(
            source=source_weights,
            destination=weights,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=destination_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=None,
        )
        np.divide(
            numerator,
            weights,
            out=output[band],
            where=weights > np.float32(1e-6),
        )
        output[band, np.abs(output[band]) < np.float32(1e-7)] = 0.0
    reproject(
        source=source_valid.astype(np.uint8),
        destination=valid_output,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=destination_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    valid_bool = valid_output.astype(bool)
    valid_bool &= np.isfinite(output).all(axis=0)
    output[:, ~valid_bool] = 0.0
    return output, valid_bool, destination_transform


def align_binary_mask_to_levir_grid(
    source_mask: np.ndarray,
    source_crs: Any,
    source_transform: Any,
    bbox: Sequence[float],
    output_shape: Tuple[int, int] = (1024, 1024),
) -> Tuple[np.ndarray, Any]:
    """Nearest-neighbour alignment for categorical QA diagnostics."""

    _, Resampling, from_bounds, reproject = _rasterio()
    height, width = (int(output_shape[0]), int(output_shape[1]))
    west, south, east, north = (float(value) for value in bbox)
    destination_transform = from_bounds(west, south, east, north, width, height)
    output = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=np.asarray(source_mask, dtype=np.uint8),
        destination=output,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=destination_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    return output.astype(bool), destination_transform


def validate_cached_rasters(
    spectral_path: Path,
    valid_path: Path,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    rasterio, _, _, _ = _rasterio()
    if not spectral_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(f"Missing cached raster(s): {spectral_path}, {valid_path}")
    with rasterio.open(spectral_path) as dataset:
        if dataset.count != len(SPECTRAL_CHANNEL_ORDER):
            raise ValueError(
                f"Expected five physical/index bands in {spectral_path}, found {dataset.count}"
            )
        if not dataset.crs:
            raise ValueError(f"Spectral raster has no CRS: {spectral_path}")
        if not all(np.issubdtype(np.dtype(dtype), np.floating) for dtype in dataset.dtypes):
            raise ValueError(f"Spectral raster must use floating-point bands: {spectral_path}")
        channels = dataset.read().astype(np.float32)
        descriptions = tuple(dataset.descriptions)
        index_crs = dataset.crs
        index_transform = dataset.transform
        index_bounds = dataset.bounds
        index_dtypes = tuple(dataset.dtypes)
        index_tags = dataset.tags()
    if descriptions != SPECTRAL_CHANNEL_ORDER:
        raise ValueError(
            "Spectral band order/descriptions must be green, red, nir, ndvi, "
            f"ndwi_mcfeeters; got {descriptions}"
        )
    with rasterio.open(valid_path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"Expected one validity band in {valid_path}")
        if not dataset.crs:
            raise ValueError(f"Validity raster has no CRS: {valid_path}")
        raw_valid = dataset.read(1)
        if not set(np.unique(raw_valid).tolist()).issubset({0, 1}):
            raise ValueError(f"Validity raster is not binary: {valid_path}")
        valid = raw_valid > 0
        if dataset.crs != index_crs or dataset.transform != index_transform:
            raise ValueError("Index and validity rasters do not share CRS/transform")
        if (dataset.height, dataset.width) != tuple(channels.shape[1:]):
            raise ValueError("Spectral and validity raster shapes differ")
    if expected_shape and tuple(channels.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"Raster shape {channels.shape[1:]} != expected {expected_shape}")
    if not all(math.isfinite(float(value)) for value in index_bounds):
        raise ValueError(f"Spectral raster bounds are not finite: {spectral_path}")
    epsilon = float(index_tags.get("epsilon", 1e-6))
    report = validate_spectral_arrays(channels, valid, epsilon=epsilon)
    report["band_descriptions"] = list(descriptions)
    report["dtype"] = list(index_dtypes)
    report["crs"] = str(index_crs)
    report["transform"] = list(index_transform)[:6]
    report["bounds"] = list(index_bounds)
    report["metadata_tags"] = index_tags
    report["spectral_sha256"] = sha256_file(spectral_path)
    report["valid_mask_sha256"] = sha256_file(valid_path)
    return report


@dataclass
class ProviderResult:
    status: str
    record: Dict[str, Any]


class SpectralProvider(ABC):
    name: str

    @abstractmethod
    def preflight(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        sample: Mapping[str, Any],
        temporal_key: str,
        output_root: Path,
        options: Mapping[str, Any],
    ) -> ProviderResult:
        raise NotImplementedError


class PrecomputedProvider(SpectralProvider):
    name = "precomputed"

    def __init__(self, source_manifest: Path):
        self.source_manifest_path = source_manifest.resolve()
        if not self.source_manifest_path.is_file():
            raise ProviderUnavailable(f"Precomputed source manifest not found: {source_manifest}")
        document = json.loads(self.source_manifest_path.read_text(encoding="utf-8"))
        raw = document.get("samples", document)
        if not isinstance(raw, Mapping):
            raise ProviderUnavailable("Precomputed source manifest samples must be keyed by LEVIR source")
        self.samples = {str(key).replace(".png", ""): value for key, value in raw.items() if isinstance(value, Mapping)}

    def preflight(self) -> Dict[str, Any]:
        _rasterio()
        return {
            "provider": self.name,
            "source_manifest": str(self.source_manifest_path),
            "source_records": len(self.samples),
            "ready": True,
        }

    def _entry(self, sample: Mapping[str, Any], temporal_key: str) -> Mapping[str, Any]:
        source = self.samples.get(Path(sample["filename"]).stem)
        if not source or not isinstance(source.get(temporal_key), Mapping):
            raise ProviderUnavailable(f"No precomputed {temporal_key} source for {sample['filename']}")
        return source[temporal_key]

    def generate(self, sample: Mapping[str, Any], temporal_key: str, output_root: Path,
                 options: Mapping[str, Any]) -> ProviderResult:
        rasterio, _, _, _ = _rasterio()
        entry = self._entry(sample, temporal_key)
        required_provenance = ("provider", "collection", "sensor", "acquisition_dates")
        missing = [key for key in required_provenance if not entry.get(key)]
        if missing:
            raise ValueError(f"Precomputed entry lacks provenance fields {missing}")
        source_base = self.source_manifest_path.parent
        physical: np.ndarray
        valid: np.ndarray
        source_crs: Any
        source_transform: Any
        native_source = ""

        if entry.get("multispectral_path"):
            path = (source_base / str(entry["multispectral_path"])).resolve()
            band_mapping = entry.get("band_mapping", {})
            if not all(name in band_mapping for name in ("green", "red", "nir")):
                raise ValueError(
                    "Genuine indices require semantic green/red/nir mappings; RGB-only input is rejected"
                )
            with rasterio.open(path) as dataset:
                if not dataset.crs:
                    raise ValueError(f"Multispectral source is not georeferenced: {path}")
                band_numbers = [int(band_mapping[name]) for name in ("green", "red", "nir")]
                if min(band_numbers) < 1 or max(band_numbers) > dataset.count:
                    raise ValueError(f"Band mapping exceeds raster band count for {path}")
                arrays = dataset.read(band_numbers).astype(np.float32)
                masks = dataset.read_masks(band_numbers) > 0
                nodata = dataset.nodata
                source_crs, source_transform = dataset.crs, dataset.transform
            scale = float(entry.get("scale_factor", 1.0))
            offset = float(entry.get("offset", 0.0))
            arrays = apply_scale_offset(arrays, scale, offset)
            base_valid = masks.all(axis=0) & np.isfinite(arrays).all(axis=0)
            if nodata is not None:
                base_valid &= (arrays != (float(nodata) * scale + offset)).all(axis=0)
            _, per_band_valid = compute_ndvi_ndwi(
                arrays[0], arrays[1], arrays[2], base_valid, float(options["epsilon"])
            )
            valid = per_band_valid.all(axis=0)
            physical = arrays[:3]
            native_source = path_relative_to(path, source_base)
        elif entry.get("spectral_path"):
            if entry.get("source_kind") not in {
                "genuine_multispectral",
                "genuine_multispectral_reflectance_indices",
            }:
                raise ValueError(
                    "Precomputed spectral entries must explicitly declare a genuine "
                    "multispectral source_kind"
                )
            formulae = entry.get("formulae", {})
            if "nir" not in str(formulae.get("ndvi", "")).lower() or "nir" not in str(formulae.get("ndwi", "")).lower():
                raise ValueError("Precomputed formula metadata does not prove NIR-based NDVI/NDWI")
            path = (source_base / str(entry["spectral_path"])).resolve()
            with rasterio.open(path) as dataset:
                if dataset.count != len(SPECTRAL_CHANNEL_ORDER) or not dataset.crs:
                    raise ValueError(
                        f"Spectral source must be a georeferenced five-band raster: {path}"
                    )
                descriptions = tuple(dataset.descriptions)
                if descriptions != SPECTRAL_CHANNEL_ORDER:
                    raise ValueError(
                        f"Precomputed spectral band descriptions are {descriptions}, "
                        f"expected {SPECTRAL_CHANNEL_ORDER}"
                    )
                source_channels = dataset.read().astype(np.float32)
                valid = (dataset.read_masks() > 0).all(axis=0)
                source_crs, source_transform = dataset.crs, dataset.transform
            if entry.get("valid_mask_path"):
                mask_path = (source_base / str(entry["valid_mask_path"])).resolve()
                with rasterio.open(mask_path) as dataset:
                    valid &= dataset.read(1) > 0
            validate_spectral_arrays(
                source_channels,
                valid,
                epsilon=float(options["epsilon"]),
            )
            physical = source_channels[:3]
            native_source = path_relative_to(path, source_base)
        else:
            raise ValueError(
                "Precomputed entry needs multispectral_path or a five-band spectral_path; "
                "an indices-only source cannot provide genuine Green/Red/NIR measurements"
            )

        height = int(options["height"])
        width = int(options["width"])
        aligned_physical, aligned_valid, aligned_transform = align_to_levir_grid(
            physical, valid, source_crs, source_transform, sample["bbox"], (height, width)
        )
        aligned_channels, aligned_valid = compose_spectral_channels(
            aligned_physical[0],
            aligned_physical[1],
            aligned_physical[2],
            aligned_valid,
            float(options["epsilon"]),
        )
        validation = validate_spectral_arrays(
            aligned_channels, aligned_valid, epsilon=float(options["epsilon"])
        )
        out_dir = output_root / "aligned" / sample["split"] / Path(sample["filename"]).stem
        spectral_path = out_dir / f"{temporal_key}_spectral.tif"
        valid_path = out_dir / f"{temporal_key}_valid.tif"
        tags = {
            "provider": self.name,
            "spectral_channel_order": SPECTRAL_CHANNEL_ORDER,
            "surface_reflectance_units": "scaled unitless surface reflectance",
            "surface_reflectance_valid_range": SURFACE_REFLECTANCE_VALID_RANGE,
            "ndwi_variant": NDWI_VARIANT,
            "formulae": FORMULAE,
            "denominator_policy": DENOMINATOR_POLICY,
            "epsilon": options["epsilon"],
            "alignment_assumption": "north_up_bbox_linear",
        }
        write_spectral_rasters(
            spectral_path,
            valid_path,
            aligned_channels,
            aligned_valid,
            "EPSG:4326",
            aligned_transform,
            tags,
        )
        fraction = validation["valid_fraction"]
        threshold = float(options["min_valid_fraction"])
        status = "ok" if fraction >= 1.0 - 1e-9 else ("ok_partial" if fraction >= threshold else "insufficient_valid_pixels")
        record = {
            "status": status,
            "provider": self.name,
            "source_provider": entry["provider"],
            "collection": entry["collection"],
            "sensor": entry["sensor"],
            "sensors": [entry["sensor"]],
            "scene_ids": list(entry.get("scene_ids", [])),
            "scene_dates": list(entry.get("acquisition_dates", [])),
            "semantic_bands": {"green": "green", "red": "red", "nir": "nir"},
            "spectral_channel_order": list(SPECTRAL_CHANNEL_ORDER),
            "surface_reflectance_order": list(SURFACE_REFLECTANCE_ORDER),
            "surface_reflectance_units": "scaled unitless surface reflectance",
            "surface_reflectance_valid_range": list(SURFACE_REFLECTANCE_VALID_RANGE),
            "physical_bands": entry.get("band_mapping", entry.get("physical_bands", {})),
            "scale_factor": entry.get("scale_factor", 1.0),
            "offset": entry.get("offset", 0.0),
            "formulae": FORMULAE,
            "denominator_policy": DENOMINATOR_POLICY,
            "epsilon": float(options["epsilon"]),
            "ndwi_variant": NDWI_VARIANT,
            "cloud_mask": entry.get("cloud_mask", "provided source validity/nodata mask"),
            "composite_method": entry.get("composite_method", "provided"),
            "native_source": native_source,
            "native_source_sha256": sha256_file(path),
            "native_resolution_m": entry.get("native_resolution_m"),
            "output_resolution": {"shape": [height, width], "bbox": list(sample["bbox"]), "crs": "EPSG:4326"},
            "aligned_shape": [height, width],
            "resampling": "bilinear",
            "mask_resampling": "nearest",
            "valid_fraction": fraction,
            "invalid_fraction": validation["invalid_fraction"],
            "per_band_valid_fraction": validation["per_band_valid_fraction"],
            "spectral_path": path_relative_to(spectral_path, output_root),
            "valid_mask_path": path_relative_to(valid_path, output_root),
            "spectral_sha256": sha256_file(spectral_path),
            "valid_mask_sha256": sha256_file(valid_path),
            "alignment_assumption": "north_up_bbox_linear",
            "attribution": entry.get("attribution", "User-supplied precomputed multispectral data"),
        }
        return ProviderResult(status, record)


class EarthEngineLandsatProvider(SpectralProvider):
    name = "earth_engine_landsat"

    def __init__(
        self,
        project: str = "",
        high_volume: bool = False,
        timeout: float = 180.0,
        sensor_policy: str = "landsat7_consistent",
    ):
        self.project = str(project).strip()
        self.high_volume = bool(high_volume)
        self.timeout = float(timeout)
        self.sensor_policy = str(sensor_policy)
        sensors_for_policy(self.sensor_policy)
        try:
            import ee
        except ImportError as exc:
            raise ProviderUnavailable(
                "earthengine-api is not installed; install requirements-auxiliary.txt"
            ) from exc
        self.ee = ee
        try:
            kwargs: Dict[str, Any] = {}
            if self.project:
                kwargs["project"] = self.project
            if self.high_volume:
                kwargs["opt_url"] = "https://earthengine-highvolume.googleapis.com"
            ee.Initialize(**kwargs)
        except Exception as exc:
            raise AuthenticationRequired(
                earth_engine_access_guidance(self.project, exc)
            ) from exc

    def preflight(self) -> Dict[str, Any]:
        _rasterio()
        try:
            value = self.ee.Number(1).getInfo()
        except Exception as exc:
            raise AuthenticationRequired(
                earth_engine_access_guidance(self.project, exc)
            ) from exc
        return {
            "provider": self.name,
            "ready": value == 1,
            "project": self.project,
            "sensor_policy": self.sensor_policy,
            "sensor_policy_description": LANDSAT_SENSOR_POLICY_DESCRIPTIONS[
                self.sensor_policy
            ],
            "collections": [
                LANDSAT_SENSORS[sensor]["collection"]
                for sensor in sensors_for_policy(self.sensor_policy)
            ],
            "sensors": list(sensors_for_policy(self.sensor_policy)),
            "scale_factor": LANDSAT_SCALE,
            "offset": LANDSAT_OFFSET,
            "native_resolution_m": LANDSAT_NATIVE_RESOLUTION_M,
            "formulae": FORMULAE,
            "denominator_policy": DENOMINATOR_POLICY,
            "landsat7_slc_failure_date": LANDSAT_7_SLC_FAILURE_DATE.isoformat(),
            "landsat7_slc_measurement": (
                "QA_RADSAT bit 9 is measured per scene; window-level dropped-any, "
                "composite-recovered, and output-loss fractions are reported"
            ),
        }

    def _project_provenance(self) -> Dict[str, str]:
        """Return the explicit CLI project for records and aligned TIFF tags."""

        return {"earth_engine_project": self.project}

    def _processed_collection(
        self,
        start: date,
        end: date,
        geometry: Any,
        sensor_policy: Optional[str] = None,
    ) -> Any:
        ee = self.ee
        collections = []
        active_policy = sensor_policy or self.sensor_policy
        for spacecraft in sensors_for_policy(active_policy):
            config = LANDSAT_SENSORS[spacecraft]
            semantic = config["bands"]
            pixel_bits = config["qa_pixel_mask_bits"]
            saturation_bits = config["qa_radsat_bits"]
            sensor_name = config["sensor"]

            def prepare(image: Any, spacecraft=spacecraft, semantic=semantic,
                        pixel_bits=pixel_bits, saturation_bits=saturation_bits,
                        sensor_name=sensor_name) -> Any:
                qa = image.select("QA_PIXEL")
                valid_without_slc = ee.Image.constant(1)
                for bit in pixel_bits:
                    valid_without_slc = valid_without_slc.And(
                        qa.bitwiseAnd(1 << bit).eq(0)
                    )
                radsat = image.select("QA_RADSAT")
                for bit in saturation_bits:
                    if bit != 9:
                        valid_without_slc = valid_without_slc.And(
                            radsat.bitwiseAnd(1 << bit).eq(0)
                        )
                if spacecraft == "LANDSAT_7":
                    slc_dropped = radsat.bitwiseAnd(1 << 9).neq(0)
                else:
                    slc_dropped = ee.Image.constant(0)
                green = image.select(semantic["green"]).multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET).rename("green")
                red = image.select(semantic["red"]).multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET).rename("red")
                nir = image.select(semantic["nir"]).multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET).rename("nir")
                ndvi_den = nir.add(red)
                ndwi_den = green.add(nir)
                denominator_valid = ndvi_den.abs().gt(1e-6).And(ndwi_den.abs().gt(1e-6))
                # Use an innocuous denominator only at invalid pixels, which
                # are explicitly masked below.  Do not add epsilon to valid
                # denominators and do not use normalizedDifference (it masks
                # valid negative reflectance inputs).
                safe_ndvi_den = ndvi_den.where(ndvi_den.abs().lte(1e-6), 1)
                safe_ndwi_den = ndwi_den.where(ndwi_den.abs().lte(1e-6), 1)
                ndvi_raw = nir.subtract(red).divide(safe_ndvi_den)
                ndwi_raw = green.subtract(nir).divide(safe_ndwi_den)
                range_valid = ndvi_raw.abs().lte(1.05).And(ndwi_raw.abs().lte(1.05))
                science_valid_without_slc = (
                    valid_without_slc.And(denominator_valid).And(range_valid)
                )
                final_valid = science_valid_without_slc.And(slc_dropped.Not())
                # Cache the physical measurements, not just products derived
                # from them.  Indices are also carried through scene inspection
                # for QA, then recomputed locally from the aligned composite so
                # the final five-band cache is formula-consistent.
                spectral = (
                    green.addBands(red)
                    .addBands(nir)
                    .addBands(ndvi_raw.clamp(-1, 1).rename("ndvi"))
                    .addBands(ndwi_raw.clamp(-1, 1).rename("ndwi_mcfeeters"))
                    .updateMask(final_valid)
                )
                diagnostics = (
                    slc_dropped.rename("slc_dropped").toUint8()
                    .addBands(
                        science_valid_without_slc.rename("valid_without_slc").toUint8()
                    )
                )
                return (
                    spectral.addBands(diagnostics)
                    .copyProperties(image, image.propertyNames())
                    .set({"ieft_spacecraft": spacecraft, "ieft_sensor": sensor_name})
                )

            collection = (
                ee.ImageCollection(config["collection"])
                .filterBounds(geometry)
                .filterDate(start.isoformat(), (end.fromordinal(end.toordinal() + 1)).isoformat())
                .map(prepare)
            )
            collections.append(collection)
        merged = collections[0]
        for collection in collections[1:]:
            merged = merged.merge(collection)
        return merged.sort("system:time_start").sort("system:index")

    def inspect_window(
        self,
        sample: Mapping[str, Any],
        temporal_key: str,
        expansion_days: int,
        sensor_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        ee = self.ee
        west, south, east, north = sample["bbox"]
        geometry = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
        month = sample[temporal_key]["image_month"]
        other_key = "t2" if temporal_key == "t1" else "t1"
        t1_window, t2_window, actual_expansion = non_overlapping_windows(
            sample["t1"]["image_month"], sample["t2"]["image_month"], expansion_days
        )
        start, end = t1_window if temporal_key == "t1" else t2_window
        collection = self._processed_collection(
            start, end, geometry, sensor_policy or self.sensor_policy
        )
        count = int(collection.size().getInfo())
        if count == 0:
            return {
                "collection": collection,
                "geometry": geometry,
                "scene_count": 0,
                "valid_fraction": 0.0,
                "start": start,
                "end": end,
                "expansion_days": actual_expansion,
                "scene_metadata": [],
                "slc_dropped_any_fraction": 0.0,
                "slc_dropped_output_loss_fraction": 0.0,
                "slc_dropped_recovered_fraction": 0.0,
            }
        composite = collection.median().select(list(SPECTRAL_CHANNEL_ORDER))
        validity = (
            composite.mask()
            .reduce(ee.Reducer.min())
            .unmask(0, sameFootprint=False)
            .rename("valid")
            .clip(geometry)
        )
        observation_count = collection.select("ndvi").count().rename("observation_count")
        valid_without_slc = (
            collection.select("valid_without_slc")
            .max()
            .unmask(0, sameFootprint=False)
            .rename("valid_without_slc")
            .clip(geometry)
        )
        slc_dropped_any = (
            collection.select("slc_dropped")
            .max()
            .unmask(0, sameFootprint=False)
            .rename("slc_dropped_any")
            .clip(geometry)
        )
        slc_dropped_output_loss = (
            valid_without_slc.And(validity.Not())
            .rename("slc_dropped_output_loss")
        )
        slc_dropped_recovered = (
            slc_dropped_any.And(validity)
            .rename("slc_dropped_recovered")
        )
        window_metrics = (
            validity.addBands(valid_without_slc)
            .addBands(slc_dropped_any)
            .addBands(slc_dropped_output_loss)
            .addBands(slc_dropped_recovered)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=LANDSAT_NATIVE_RESOLUTION_M,
                bestEffort=True,
                maxPixels=1_000_000,
            )
            .getInfo()
        )

        def scene_feature(image: Any) -> Any:
            diagnostics = (
                image.select(["slc_dropped", "valid_without_slc"])
                .unmask(0, sameFootprint=False)
                .addBands(
                    image.select("green")
                    .mask()
                    .unmask(0, sameFootprint=False)
                    .rename("final_valid")
                )
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=LANDSAT_NATIVE_RESOLUTION_M,
                    bestEffort=True,
                    maxPixels=1_000_000,
                )
            )
            return ee.Feature(
                None,
                {
                    "scene_id": image.get("LANDSAT_PRODUCT_ID"),
                    "system_index": image.get("system:index"),
                    "acquisition_time": image.get("system:time_start"),
                    "spacecraft": image.get("ieft_spacecraft"),
                    "sensor": image.get("ieft_sensor"),
                    "cloud_cover": image.get("CLOUD_COVER"),
                    "slc_dropped_pixel_fraction": diagnostics.get("slc_dropped"),
                    "valid_without_slc_fraction": diagnostics.get("valid_without_slc"),
                    "final_valid_fraction": diagnostics.get("final_valid"),
                },
            )

        # ImageCollection.map must return Images. Convert the bounded scene list
        # to a FeatureCollection for metadata/reduction records instead.
        raw_metadata = self._scene_metadata_feature_collection(
            collection, count, scene_feature
        ).getInfo()
        scene_metadata = []
        for feature in raw_metadata.get("features", []):
            props = feature.get("properties", {})
            timestamp = props.get("acquisition_time")
            acquisition_date = datetime.fromtimestamp(timestamp / 1000, timezone.utc).date().isoformat() if timestamp else None
            scene_metadata.append(
                {
                    **props,
                    "acquisition_date": acquisition_date,
                    "slc_mode": landsat7_slc_mode(acquisition_date)
                    if props.get("spacecraft") == "LANDSAT_7"
                    else "not_applicable",
                }
            )
        return {
            "collection": collection,
            "geometry": geometry,
            "composite": composite,
            "validity": validity,
            "observation_count": observation_count,
            "slc_dropped_any": slc_dropped_any,
            "slc_dropped_output_loss": slc_dropped_output_loss,
            "slc_dropped_recovered": slc_dropped_recovered,
            "scene_count": count,
            "valid_fraction": float(window_metrics.get("valid") or 0.0),
            "valid_without_slc_fraction": float(
                window_metrics.get("valid_without_slc") or 0.0
            ),
            "slc_dropped_any_fraction": float(
                window_metrics.get("slc_dropped_any") or 0.0
            ),
            "slc_dropped_output_loss_fraction": float(
                window_metrics.get("slc_dropped_output_loss") or 0.0
            ),
            "slc_dropped_recovered_fraction": float(
                window_metrics.get("slc_dropped_recovered") or 0.0
            ),
            "start": start,
            "end": end,
            "expansion_days": actual_expansion,
            "scene_metadata": scene_metadata,
        }

    def _scene_metadata_feature_collection(
        self, collection: Any, count: int, scene_feature: Any
    ) -> Any:
        """Map Images to Features through a List, not ImageCollection.map."""

        ee = self.ee
        return ee.FeatureCollection(
            collection.toList(int(count)).map(
                lambda raw_image: scene_feature(ee.Image(raw_image))
            )
        )

    def _download_native(self, image: Any, geometry: Any, destination: Path) -> None:
        image = image.unmask(-9999, sameFootprint=False)
        url = image.getDownloadURL(
            {
                "region": geometry,
                "scale": LANDSAT_NATIVE_RESOLUTION_M,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
                "filePerBand": False,
                "name": destination.stem,
            }
        )
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(response.content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, destination)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise

    def generate(self, sample: Mapping[str, Any], temporal_key: str, output_root: Path,
                 options: Mapping[str, Any]) -> ProviderResult:
        rasterio, _, _, _ = _rasterio()
        selected = None
        attempts = []
        for expansion in expansion_steps(int(options["expand_step_days"]), int(options["max_window_days"])):
            inspected = self.inspect_window(
                sample, temporal_key, expansion, str(options["sensor_policy"])
            )
            attempts.append(
                {
                    "window": [inspected["start"].isoformat(), inspected["end"].isoformat()],
                    "expansion_days": inspected["expansion_days"],
                    "scene_count": inspected["scene_count"],
                    "valid_fraction": inspected["valid_fraction"],
                    "valid_without_slc_fraction": inspected.get(
                        "valid_without_slc_fraction", 0.0
                    ),
                    "slc_dropped_any_fraction": inspected.get(
                        "slc_dropped_any_fraction", 0.0
                    ),
                    "slc_dropped_output_loss_fraction": inspected.get(
                        "slc_dropped_output_loss_fraction", 0.0
                    ),
                    "slc_dropped_recovered_fraction": inspected.get(
                        "slc_dropped_recovered_fraction", 0.0
                    ),
                }
            )
            selected = inspected
            if inspected["scene_count"] and inspected["valid_fraction"] >= float(options["min_valid_fraction"]):
                break
        assert selected is not None
        fraction = float(selected["valid_fraction"])
        if selected["scene_count"] == 0:
            return ProviderResult("source_unavailable", {"status": "source_unavailable", "attempted_windows": attempts})
        if fraction < float(options["min_valid_fraction"]):
            status = "insufficient_valid_pixels"
        else:
            status = "ok" if fraction >= 1.0 - 1e-9 else "ok_partial"

        native_dir = output_root / "native" / sample["split"] / Path(sample["filename"]).stem
        native_path = native_dir / f"{temporal_key}_composite.tif"
        download_image = (
            selected["composite"]
            .addBands(selected["validity"])
            .addBands(selected["observation_count"])
            .addBands(selected["slc_dropped_any"])
            .addBands(selected["slc_dropped_output_loss"])
            .addBands(selected["slc_dropped_recovered"])
        )
        self._download_native(download_image, selected["geometry"], native_path)
        with rasterio.open(native_path) as dataset:
            expected_native_bands = len(SPECTRAL_CHANNEL_ORDER) + 5
            if dataset.count < expected_native_bands or not dataset.crs:
                raise ValueError(f"Downloaded Earth Engine raster is malformed: {native_path}")
            arrays = dataset.read(list(range(1, expected_native_bands + 1))).astype(np.float32)
            source_crs, source_transform = dataset.crs, dataset.transform
            native_resolution = tuple(abs(value) for value in dataset.res)
        physical = arrays[:3]
        diagnostic_offset = len(SPECTRAL_CHANNEL_ORDER)
        valid = (
            (arrays[diagnostic_offset] > 0.5)
            & np.isfinite(physical).all(axis=0)
            & (physical != -9999).all(axis=0)
        )
        native_slc_dropped_any = arrays[diagnostic_offset + 2] > 0.5
        native_slc_output_loss = arrays[diagnostic_offset + 3] > 0.5
        native_slc_recovered = arrays[diagnostic_offset + 4] > 0.5
        physical[:, ~valid] = 0.0
        height, width = int(options["height"]), int(options["width"])
        aligned_physical, aligned_valid, transform = align_to_levir_grid(
            physical, valid, source_crs, source_transform, sample["bbox"], (height, width)
        )
        aligned_channels, aligned_valid = compose_spectral_channels(
            aligned_physical[0],
            aligned_physical[1],
            aligned_physical[2],
            aligned_valid,
            float(options["epsilon"]),
        )
        aligned_slc_dropped_any, _ = align_binary_mask_to_levir_grid(
            native_slc_dropped_any,
            source_crs,
            source_transform,
            sample["bbox"],
            (height, width),
        )
        aligned_slc_output_loss, _ = align_binary_mask_to_levir_grid(
            native_slc_output_loss,
            source_crs,
            source_transform,
            sample["bbox"],
            (height, width),
        )
        aligned_slc_recovered, _ = align_binary_mask_to_levir_grid(
            native_slc_recovered,
            source_crs,
            source_transform,
            sample["bbox"],
            (height, width),
        )
        validation = validate_spectral_arrays(
            aligned_channels, aligned_valid, epsilon=float(options["epsilon"])
        )
        aligned_fraction = validation["valid_fraction"]
        if aligned_fraction < float(options["min_valid_fraction"]):
            status = "insufficient_valid_pixels"
        elif aligned_fraction < 1.0 - 1e-9:
            status = "ok_partial"
        else:
            status = "ok"
        out_dir = output_root / "aligned" / sample["split"] / Path(sample["filename"]).stem
        spectral_path = out_dir / f"{temporal_key}_spectral.tif"
        valid_path = out_dir / f"{temporal_key}_valid.tif"
        tags = {
            "provider": self.name,
            **self._project_provenance(),
            "spectral_channel_order": SPECTRAL_CHANNEL_ORDER,
            "surface_reflectance_units": "scaled unitless surface reflectance",
            "surface_reflectance_valid_range": SURFACE_REFLECTANCE_VALID_RANGE,
            "collections": [
                LANDSAT_SENSORS[sensor]["collection"]
                for sensor in sensors_for_policy(str(options["sensor_policy"]))
            ],
            "formulae": FORMULAE,
            "denominator_policy": DENOMINATOR_POLICY,
            "epsilon": float(options["epsilon"]),
            "ndwi_variant": NDWI_VARIANT,
            "scale_factor": LANDSAT_SCALE,
            "offset": LANDSAT_OFFSET,
            "qa_mask": "QA_PIXEL fill,dilated cloud,cirrus(L8),cloud,shadow,snow; QA_RADSAT semantic bands/dropped pixels",
            "composite": "deterministic median",
            "alignment_assumption": "north_up_bbox_linear",
            "landsat7_slc_failure_date": LANDSAT_7_SLC_FAILURE_DATE.isoformat(),
            "landsat7_slc_dropped_pixel_qa_radsat_bit": 9,
        }
        write_spectral_rasters(
            spectral_path,
            valid_path,
            aligned_channels,
            aligned_valid,
            "EPSG:4326",
            transform,
            tags,
        )
        scenes = selected["scene_metadata"]
        sensors = sorted({str(item.get("spacecraft")) for item in scenes if item.get("spacecraft")})
        scene_cloud_cover = [item.get("cloud_cover") for item in scenes]
        numeric_scene_cloud_cover = [
            float(value) for value in scene_cloud_cover if value is not None and math.isfinite(float(value))
        ]
        record = {
            "status": status,
            "provider": self.name,
            **self._project_provenance(),
            "collection": "Landsat Collection 2 Tier 1 Level 2 surface reflectance",
            "collections": [LANDSAT_SENSORS[sensor]["collection"] for sensor in sensors if sensor in LANDSAT_SENSORS],
            "sensor": ",".join(sensors),
            "sensors": sensors,
            "sensor_policy": options["sensor_policy"],
            "sensor_policy_description": LANDSAT_SENSOR_POLICY_DESCRIPTIONS[
                str(options["sensor_policy"])
            ],
            "harmonization": "not_applicable_single_sensor",
            "semantic_bands": {"green": "green", "red": "red", "nir": "nir"},
            "spectral_channel_order": list(SPECTRAL_CHANNEL_ORDER),
            "surface_reflectance_order": list(SURFACE_REFLECTANCE_ORDER),
            "surface_reflectance_units": "scaled unitless surface reflectance",
            "surface_reflectance_valid_range": list(SURFACE_REFLECTANCE_VALID_RANGE),
            "physical_bands": {sensor: LANDSAT_SENSORS[sensor]["bands"] for sensor in sensors if sensor in LANDSAT_SENSORS},
            "scale_factor": LANDSAT_SCALE,
            "offset": LANDSAT_OFFSET,
            "formulae": FORMULAE,
            "denominator_policy": DENOMINATOR_POLICY,
            "epsilon": float(options["epsilon"]),
            "ndwi_variant": NDWI_VARIANT,
            "cloud_mask": {
                sensor: {
                    "qa_pixel_invalid_bits": LANDSAT_SENSORS[sensor]["qa_pixel_mask_bits"],
                    "qa_radsat_invalid_bits": LANDSAT_SENSORS[sensor]["qa_radsat_bits"],
                }
                for sensor in sensors if sensor in LANDSAT_SENSORS
            },
            "requested_window": [calendar_month_window(sample[temporal_key]["image_month"])[0].isoformat(), calendar_month_window(sample[temporal_key]["image_month"])[1].isoformat()],
            "actual_window": [selected["start"].isoformat(), selected["end"].isoformat()],
            "window_expanded": selected["expansion_days"] > 0,
            "expansion_days": selected["expansion_days"],
            "attempted_windows": attempts,
            "scene_ids": [item.get("scene_id") or item.get("system_index") for item in scenes],
            "scene_dates": [item.get("acquisition_date") for item in scenes],
            "scene_metadata": scenes,
            "scene_cloud_cover": scene_cloud_cover,
            "mean_scene_cloud_cover_fraction": (
                float(np.mean(numeric_scene_cloud_cover) / 100.0)
                if numeric_scene_cloud_cover else None
            ),
            "mean_scene_cloud_cover_definition": (
                "mean Landsat catalog CLOUD_COVER property divided by 100; scene-level metadata, "
                "not a QA-derived cloud fraction over the LEVIR bbox"
            ),
            "observation_count": selected["scene_count"],
            "composite_method": "median",
            "native_resolution_m": LANDSAT_NATIVE_RESOLUTION_M,
            "downloaded_native_pixel_size": list(native_resolution),
            "native_path": path_relative_to(native_path, output_root),
            "native_sha256": sha256_file(native_path),
            "output_resolution": {"shape": [height, width], "bbox": list(sample["bbox"]), "crs": "EPSG:4326"},
            "aligned_shape": [height, width],
            "resampling": "bilinear",
            "mask_resampling": "nearest",
            "valid_fraction": aligned_fraction,
            "invalid_fraction": validation["invalid_fraction"],
            "native_valid_fraction": float(valid.mean()),
            "native_invalid_fraction": float(1.0 - valid.mean()),
            "earth_engine_window_valid_fraction": fraction,
            "per_band_valid_fraction": validation["per_band_valid_fraction"],
            "landsat7_slc_failure_date": LANDSAT_7_SLC_FAILURE_DATE.isoformat(),
            "landsat7_slc_dropped_pixel_qa_radsat_bit": 9,
            "slc_on_scene_count": sum(item.get("slc_mode") == "on" for item in scenes),
            "slc_off_scene_count": sum(item.get("slc_mode") == "off" for item in scenes),
            "slc_dropped_any_fraction": float(native_slc_dropped_any.mean()),
            "slc_dropped_output_loss_fraction": float(native_slc_output_loss.mean()),
            "slc_dropped_recovered_fraction": float(native_slc_recovered.mean()),
            "aligned_slc_dropped_any_fraction": float(aligned_slc_dropped_any.mean()),
            "aligned_slc_dropped_output_loss_fraction": float(
                aligned_slc_output_loss.mean()
            ),
            "aligned_slc_dropped_recovered_fraction": float(
                aligned_slc_recovered.mean()
            ),
            "earth_engine_slc_dropped_any_fraction": selected.get(
                "slc_dropped_any_fraction", 0.0
            ),
            "earth_engine_slc_dropped_output_loss_fraction": selected.get(
                "slc_dropped_output_loss_fraction", 0.0
            ),
            "earth_engine_slc_dropped_recovered_fraction": selected.get(
                "slc_dropped_recovered_fraction", 0.0
            ),
            "slc_effect_definition": (
                "dropped-any is QA_RADSAT bit 9; output-loss is a pixel with an "
                "otherwise valid observation but no final valid composite observation; "
                "recovered is a dropped-any pixel filled by another valid observation"
            ),
            "spectral_path": path_relative_to(spectral_path, output_root),
            "valid_mask_path": path_relative_to(valid_path, output_root),
            "spectral_sha256": sha256_file(spectral_path),
            "valid_mask_sha256": sha256_file(valid_path),
            "alignment_assumption": "north_up_bbox_linear",
            "attribution": "Landsat Collection 2 Level-2 imagery courtesy of the U.S. Geological Survey; processed with Google Earth Engine",
        }
        return ProviderResult(status, record)


__all__ = [
    "ACCEPTABLE_STATUSES",
    "AuthenticationRequired",
    "DENOMINATOR_POLICY",
    "EARTH_ENGINE_REGISTRATION_URL",
    "EarthEngineLandsatProvider",
    "FAILED_STATUSES",
    "FORMULAE",
    "INDEX_ORDER",
    "LANDSAT_OFFSET",
    "LANDSAT_SCALE",
    "LANDSAT_7_SLC_FAILURE_DATE",
    "LANDSAT_SENSORS",
    "NDWI_VARIANT",
    "PrecomputedProvider",
    "ProviderResult",
    "ProviderUnavailable",
    "SPECTRAL_SCHEMA_VERSION",
    "SPECTRAL_CHANNEL_ORDER",
    "SURFACE_REFLECTANCE_ORDER",
    "SURFACE_REFLECTANCE_VALID_RANGE",
    "SpectralProvider",
    "align_to_levir_grid",
    "align_binary_mask_to_levir_grid",
    "apply_scale_offset",
    "compute_ndvi_ndwi",
    "compose_spectral_channels",
    "deterministic_nanmedian",
    "earth_engine_access_guidance",
    "expansion_steps",
    "landsat_qa_valid_mask",
    "landsat7_slc_mode",
    "LANDSAT_SENSOR_POLICIES",
    "LANDSAT_SENSOR_POLICY_DESCRIPTIONS",
    "sensors_for_policy",
    "validate_cached_rasters",
    "validate_index_arrays",
    "validate_spectral_arrays",
    "write_spectral_rasters",
]
