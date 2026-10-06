"""Offline-only auxiliary-data helpers for the LEVIR-CD runtime dataset.

This module deliberately contains no HTTP clients.  It reads generated manifests
and local raster caches only, so importing or using a DataLoader worker can never
contact OSM, ohsome, Earth Engine, or another external service.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from IEFT.levir_metadata import (
        load_manifest as _load_common_manifest,
        natural_sample_sort_key,
        normalize_sample_key,
    )
except (ImportError, AttributeError):  # pragma: no cover - compatibility with old installs.
    _load_common_manifest = None

    def normalize_sample_key(value: Any, keep_extension: bool = False) -> str:
        raw = Path(str(value).replace("\\", "/")).name.strip()
        raw = re.sub(r"(?:_y\d+_x\d+|_r\d+_c\d+)$", "", raw, flags=re.IGNORECASE)
        raw = Path(raw).stem
        match = re.fullmatch(r"(train|val|test)_(\d+)", raw, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Unrecognized LEVIR sample key: {value!r}")
        key = f"{match.group(1).lower()}_{int(match.group(2))}"
        return key + ".png" if keep_extension else key

    def natural_sample_sort_key(value: str) -> Tuple[int, int, str]:
        key = normalize_sample_key(value)
        split, index = key.rsplit("_", 1)
        return ({"train": 0, "val": 1, "test": 2}[split], int(index), key)


OSM_AVAILABLE_STATUSES = frozenset({"ok", "ok_empty", "success", "available"})
SPECTRAL_AVAILABLE_STATUSES = frozenset(
    {"ok", "ok_partial", "success", "available", "complete", "partial"}
)
SURFACE_REFLECTANCE_ORDER = ("green", "red", "nir")
SPECTRAL_INDEX_ORDER = ("ndvi", "ndwi_mcfeeters")
SPECTRAL_CHANNEL_ORDER = SURFACE_REFLECTANCE_ORDER + SPECTRAL_INDEX_ORDER
SURFACE_REFLECTANCE_VALID_RANGE = (-0.2, 1.6)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _load_json_manifest_fallback(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"Manifest root must be an object: {path}")
    raw_samples = document.get("samples", document.get("records", document))
    if isinstance(raw_samples, list):
        samples = raw_samples
    elif isinstance(raw_samples, Mapping):
        samples = []
        for key, raw in raw_samples.items():
            if key in {"schema_version", "metadata", "generation_report"}:
                continue
            if isinstance(raw, Mapping):
                item = dict(raw)
                item.setdefault("filename", key)
                samples.append(item)
    else:
        raise ValueError(f"Manifest must contain a samples list or keyed records: {path}")

    records: Dict[str, Dict[str, Any]] = {}
    for raw in samples:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Manifest sample entries must be objects: {path}")
        item = dict(raw)
        stem = normalize_sample_key(item.get("filename", item.get("source_pair_id", "")))
        if stem in records:
            raise ValueError(f"Duplicate manifest sample {stem}: {path}")
        item["filename"] = stem + ".png"
        item.setdefault("source_pair_id", stem)
        item.setdefault("split", stem.rsplit("_", 1)[0])
        records[stem] = item
    return dict(document), records


class ManifestIndex:
    """Immutable in-memory index over a generated LEVIR auxiliary manifest."""

    def __init__(self, path: str = "", kind: str = "auxiliary"):
        self.kind = str(kind)
        self.path = Path(path).expanduser() if _safe_text(path) else None
        self.base_dir = self.path.resolve().parent if self.path is not None else None
        self.document: Dict[str, Any] = {}
        self.records: Dict[str, Dict[str, Any]] = {}
        self.schema_version = ""
        self.sha256 = ""
        if self.path is None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.kind} manifest not found: {self.path}")
        if _load_common_manifest is not None:
            document, records = _load_common_manifest(self.path)
        else:  # pragma: no cover - used only with an older package checkout.
            document, records = _load_json_manifest_fallback(self.path)
        self.document = document
        self.records = records
        self.schema_version = _safe_text(document.get("schema_version", ""))
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def get(self, source_key: str) -> Optional[Dict[str, Any]]:
        return self.records.get(normalize_sample_key(source_key))

    def resolve_path(self, value: Any) -> Optional[Path]:
        text = _safe_text(value)
        if not text:
            return None
        path = Path(text.replace("\\", "/"))
        # Path accepts forward slashes on Windows.  Do not resolve an absent
        # cache relative to the process CWD: manifests are self-contained.
        if not path.is_absolute():
            if self.base_dir is None:
                return path
            # Canonical manifests store paths relative to the manifest folder.
            # The retained schema-less ``data_spectral_v2`` manifest predates
            # that rule and stores repository-relative values beginning with
            # the manifest directory name.  Resolve that one documented layout
            # without falling back to arbitrary process-CWD paths.
            parts = path.parts
            if parts and parts[0].lower() == self.base_dir.name.lower():
                path = self.base_dir.parent / path
            else:
                path = self.base_dir / path
        return path.resolve()


def _status(record: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(record, Mapping):
        return "missing"
    return _safe_text(record.get("status", "missing")).lower() or "missing"


def _valid_fraction(record: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(record, Mapping):
        return None
    value = record.get(
        "valid_fraction",
        record.get("valid_pixel_fraction", record.get("valid_pixel_ratio")),
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_path_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = record.get(name)
        if _safe_text(value):
            return value
    return ""


def _legacy_index_cache_path(
    sample: Optional[Mapping[str, Any]], manifest: ManifestIndex
) -> Optional[Path]:
    """Return the retained schema-less paired-index NPZ, if declared."""

    if not isinstance(sample, Mapping) or manifest.schema_version:
        return None
    value = _first_path_value(sample, ("raster_path", "spectral_raster_path"))
    return manifest.resolve_path(value)


def spectral_paths(record: Optional[Mapping[str, Any]], manifest: ManifestIndex) -> Dict[str, Optional[Path]]:
    if not isinstance(record, Mapping):
        return {
            "spectral": None,
            "reflectance": None,
            "indices": None,
            "ndvi": None,
            "ndwi": None,
            "mask": None,
        }
    return {
        "spectral": manifest.resolve_path(
            _first_path_value(
                record,
                (
                    "spectral_path",
                    "spectral_raster_path",
                    "reflectance_indices_path",
                ),
            )
        ),
        "reflectance": manifest.resolve_path(
            _first_path_value(
                record,
                ("surface_reflectance_path", "reflectance_path", "bands_path"),
            )
        ),
        "indices": manifest.resolve_path(
            _first_path_value(record, ("indices_path", "spectral_indices_path"))
        ),
        "ndvi": manifest.resolve_path(_first_path_value(record, ("ndvi_path", "NDVI_path"))),
        "ndwi": manifest.resolve_path(_first_path_value(record, ("ndwi_path", "NDWI_path"))),
        "mask": manifest.resolve_path(
            _first_path_value(record, ("valid_mask_path", "mask_path", "validity_path"))
        ),
    }


def assess_temporal_source(
    source_key: str,
    manifest: ManifestIndex,
    kind: str,
    min_valid_fraction: float = 0.0,
    accept_partial: bool = True,
    require_nonempty: bool = False,
    require_osm_cache: bool = False,
    temporal_keys: Sequence[str] = ("t1", "t2"),
    allow_legacy_spectral_indices: bool = False,
) -> Dict[str, Any]:
    """Return per-date availability and machine-readable exclusion reasons."""

    sample = manifest.get(source_key) if manifest.enabled else None
    result: Dict[str, Any] = {
        "sample": sample,
        "available": {"t1": False, "t2": False},
        "status": {"t1": "manifest_disabled", "t2": "manifest_disabled"},
        "valid_fraction": {"t1": None, "t2": None},
        "paths": {"t1": {}, "t2": {}},
        "reasons": [],
        "reasons_by_date": {"t1": [], "t2": []},
    }
    if not manifest.enabled:
        return result
    if sample is None:
        result["status"] = {"t1": "missing_sample", "t2": "missing_sample"}
        for temporal_key in ("t1", "t2"):
            result["reasons_by_date"][temporal_key].append(
                f"{kind}_{temporal_key}:missing_sample"
            )
        result["reasons"] = [
            reason
            for temporal_key in ("t1", "t2")
            for reason in result["reasons_by_date"][temporal_key]
        ]
        return result

    requested_keys = tuple(str(value).strip().lower() for value in temporal_keys)
    if not requested_keys or any(value not in {"t1", "t2"} for value in requested_keys):
        raise ValueError(f"temporal_keys must contain t1 and/or t2, got {temporal_keys!r}")
    allowed = OSM_AVAILABLE_STATUSES if kind == "osm" else SPECTRAL_AVAILABLE_STATUSES
    if kind == "spectral" and not accept_partial:
        allowed = frozenset(allowed - {"ok_partial", "partial"})
    for temporal_key in ("t1", "t2"):
        date_reasons = result["reasons_by_date"][temporal_key]
        if temporal_key not in requested_keys:
            result["status"][temporal_key] = "not_requested_runtime"
            continue
        date_record = sample.get(temporal_key)
        status = _status(date_record if isinstance(date_record, Mapping) else None)
        result["status"][temporal_key] = status
        if not isinstance(date_record, Mapping):
            date_reasons.append(f"{kind}_{temporal_key}:missing_record")
            continue
        if status not in allowed:
            if kind == "spectral" and status in {"ok_partial", "partial"}:
                date_reasons.append(f"spectral_{temporal_key}:partial_disallowed")
            else:
                date_reasons.append(f"{kind}_{temporal_key}:{status}")
            continue
        if kind == "osm" and require_nonempty and status == "ok_empty":
            date_reasons.append(f"osm_{temporal_key}:empty_disallowed")
            # ok_empty remains genuinely available; this reason controls only
            # strict experiment eligibility, not the runtime availability mask.
        if kind == "osm" and require_osm_cache:
            raw_path = manifest.resolve_path(
                _first_path_value(date_record, ("raw_geojson", "raw_geojson_path", "raw_path"))
            )
            result["paths"][temporal_key] = {"raw_geojson": raw_path}
            if raw_path is None or not raw_path.is_file():
                date_reasons.append(f"osm_{temporal_key}:missing_cached_file")
                result.setdefault("missing_paths", []).append(
                    "" if raw_path is None else str(raw_path)
                )
                continue
            try:
                document = json.loads(raw_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(document, Mapping)
                    or document.get("type") != "FeatureCollection"
                    or not isinstance(document.get("features"), list)
                ):
                    raise ValueError("expected a GeoJSON FeatureCollection")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                date_reasons.append(f"osm_{temporal_key}:missing_cached_file")
                result.setdefault("cache_errors", {})[temporal_key] = str(exc)
                continue
        if kind == "spectral":
            fraction = _valid_fraction(date_record)
            result["valid_fraction"][temporal_key] = fraction
            if fraction is None and float(min_valid_fraction) > 0:
                date_reasons.append(f"spectral_{temporal_key}:missing_valid_fraction")
                continue
            if fraction is not None and fraction < float(min_valid_fraction):
                date_reasons.append(f"spectral_{temporal_key}:below_min_valid_fraction")
                continue
            legacy_path = (
                _legacy_index_cache_path(sample, manifest)
                if allow_legacy_spectral_indices
                else None
            )
            if legacy_path is not None:
                result["paths"][temporal_key] = {
                    "legacy_indices": legacy_path,
                }
                if not legacy_path.is_file():
                    date_reasons.append(f"spectral_{temporal_key}:missing_cached_file")
                    result.setdefault("missing_paths", []).append(str(legacy_path))
                    continue
                # This cache contains genuine Landsat-derived NDVI/NDWI and an
                # explicit validity mask, but predates physical-band/checksum
                # provenance.  Admit it as a compatibility tier and expose that
                # limitation in every runtime sample; never claim formula-level
                # verification that its schema cannot support.
                try:
                    with np.load(legacy_path, allow_pickle=False) as archive:
                        required = {
                            f"ndvi_{temporal_key}",
                            f"ndwi_{temporal_key}",
                            f"valid_mask_{temporal_key}",
                        }
                        missing_keys = sorted(required.difference(archive.files))
                    if missing_keys:
                        raise KeyError(", ".join(missing_keys))
                except (OSError, ValueError, KeyError) as exc:
                    date_reasons.append(f"spectral_{temporal_key}:invalid_legacy_cache")
                    result.setdefault("cache_errors", {})[temporal_key] = str(exc)
                    continue
                result.setdefault("compatibility_tier", "legacy_indices_only")
                result["available"][temporal_key] = True
                continue

            paths = spectral_paths(date_record, manifest)
            result["paths"][temporal_key] = paths
            declared_order = date_record.get("spectral_channel_order", ())
            if tuple(str(value).lower() for value in declared_order) != SPECTRAL_CHANNEL_ORDER:
                date_reasons.append(f"spectral_{temporal_key}:invalid_channel_provenance")
                continue
            declared_range = date_record.get("surface_reflectance_valid_range", ())
            if tuple(declared_range) != SURFACE_REFLECTANCE_VALID_RANGE:
                date_reasons.append(f"spectral_{temporal_key}:invalid_reflectance_provenance")
                continue
            has_combined = paths["spectral"] is not None
            has_indices = paths["indices"] is not None or (
                paths["ndvi"] is not None and paths["ndwi"] is not None
            )
            has_separate = paths["reflectance"] is not None and has_indices
            if not has_combined and not has_separate:
                date_reasons.append(f"spectral_{temporal_key}:missing_physical_spectral_path")
                continue
            if has_combined:
                required_paths = [paths["spectral"]]
            elif paths["indices"] is not None:
                required_paths = [paths["reflectance"], paths["indices"]]
            else:
                required_paths = [paths["reflectance"], paths["ndvi"], paths["ndwi"]]
            if paths["mask"] is None:
                date_reasons.append(f"spectral_{temporal_key}:missing_valid_mask_path")
                continue
            required_hashes = ["valid_mask_sha256"]
            if has_combined:
                required_hashes.append("spectral_sha256")
            elif paths["indices"] is not None:
                required_hashes.extend(["surface_reflectance_sha256", "indices_sha256"])
            else:
                required_hashes.extend(
                    ["surface_reflectance_sha256", "ndvi_sha256", "ndwi_sha256"]
                )
            invalid_hashes = [
                name
                for name in required_hashes
                if not re.fullmatch(r"[0-9a-fA-F]{64}", _safe_text(date_record.get(name)))
            ]
            if invalid_hashes:
                date_reasons.append(f"spectral_{temporal_key}:missing_cache_checksums")
                result.setdefault("invalid_checksums", {})[temporal_key] = invalid_hashes
                continue
            missing_paths = [str(path) for path in required_paths if path is None or not path.is_file()]
            if not paths["mask"].is_file():
                missing_paths.append(str(paths["mask"]))
            if missing_paths:
                date_reasons.append(f"spectral_{temporal_key}:missing_cached_file")
                result.setdefault("missing_paths", []).extend(missing_paths)
                continue
        result["available"][temporal_key] = True
    result["reasons"] = [
        reason
        for temporal_key in ("t1", "t2")
        for reason in result["reasons_by_date"][temporal_key]
    ]
    return result


def _norm_count(value: Any, cap: float) -> float:
    try:
        return float(min(max(float(value), 0.0), cap) / cap)
    except (TypeError, ValueError):
        return 0.0


def _summary_entry(record: Mapping[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    summary = record.get("summary")
    if isinstance(summary, Mapping):
        entry.update(summary)
    for key in (
        "text_v21", "summary", "source_text", "tags", "phrases", "osm_text",
        "osm_struct", "osm_struct_hint", "feature_summary",
    ):
        if key in record and not (key == "summary" and isinstance(record[key], Mapping)):
            entry[key] = record[key]
    feature_summary = entry.get("feature_summary")
    if isinstance(feature_summary, Mapping):
        for key, value in feature_summary.items():
            entry.setdefault(key, value)
    return entry


def osm_struct_from_entry(raw_entry: Optional[Mapping[str, Any]], dim: int = 16) -> torch.Tensor:
    """Normalize canonical or legacy OSM summaries to the legacy 16-D contract."""

    if not isinstance(raw_entry, Mapping):
        return torch.zeros(dim, dtype=torch.float32)
    entry = _summary_entry(raw_entry)
    explicit = entry.get("osm_struct")
    if isinstance(explicit, torch.Tensor):
        vector = explicit.detach().float().flatten()
        if vector.numel() != dim:
            raise ValueError(f"OSM structure must have {dim} values, got {vector.numel()}")
        return vector
    if isinstance(explicit, np.ndarray):
        explicit = explicit.reshape(-1).tolist()
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes, Mapping)):
        if len(explicit) != dim:
            raise ValueError(f"OSM structure must have {dim} values, got {len(explicit)}")
        return torch.tensor([float(value) for value in explicit], dtype=torch.float32)

    hint = explicit if isinstance(explicit, Mapping) else entry.get("osm_struct_hint", {})
    if not isinstance(hint, Mapping):
        hint = {}
    # Canonical summaries may put counts directly on the summary object.
    if not hint:
        hint = entry
    tags = entry.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    text_values: List[str] = []
    for name in ("text_v21", "summary", "source_text", "osm_text"):
        value = entry.get(name, "")
        if isinstance(value, str):
            text_values.append(value)
    phrases = entry.get("phrases", [])
    if isinstance(phrases, list):
        text_values.extend(_safe_text(value) for value in phrases)
    text_values.extend(_safe_text(value) for value in tags)
    text_blob = " ".join(text_values).lower()

    def has_any(words: Iterable[str]) -> float:
        return 1.0 if any(word in text_blob for word in words) else 0.0

    def count(*names: str) -> float:
        for name in names:
            if name in hint:
                try:
                    return float(hint[name])
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    building_count = count("building_count", "buildings")
    road_count = count("road_count", "roads")
    railway_count = count("railway_count", "railways")
    water_count = count("water_count", "water")
    vegetation_count = count("vegetation_count", "vegetation", "green")
    construction_count = count("construction_count", "construction")
    industrial_count = count("industrial_count", "industrial")
    residential_count = count("residential_count", "residential")
    amenity_count = count("amenity_count", "amenities")

    vector = np.zeros(dim, dtype=np.float32)
    vector[0] = max(has_any(("building", "built-up", "built up", "residential", "industrial")), float(building_count > 0))
    vector[1] = max(has_any(("road", "transport", "highway", "street")), float(road_count > 0))
    vector[2] = max(has_any(("railway", "rail")), float(railway_count > 0))
    vector[3] = max(has_any(("water", "river", "wetland")), float(water_count > 0))
    vector[4] = max(has_any(("vegetation", "forest", "wood", "park", "green")), float(vegetation_count > 0))
    vector[5] = max(has_any(("residential", "neighborhood", "neighbourhood")), float(residential_count > 0))
    vector[6] = max(has_any(("industrial", "commercial")), float(industrial_count > 0))
    vector[7] = max(has_any(("construction",)), float(construction_count > 0))
    vector[8] = max(has_any(("amenity", "parking")), float(amenity_count > 0))
    vector[9] = _norm_count(building_count, 120.0)
    vector[10] = _norm_count(road_count + railway_count, 180.0)
    vector[11] = _norm_count(water_count, 30.0)
    vector[12] = _norm_count(vegetation_count, 60.0)
    vector[13] = min(1.0, 0.55 * vector[0] + 0.25 * vector[5] + 0.20 * vector[6] + 0.35 * vector[9])
    vector[14] = min(1.0, 0.55 * vector[1] + 0.20 * vector[2] + 0.25 * vector[10])
    vector[15] = min(1.0, 0.50 * vector[13] + 0.30 * vector[14] - 0.20 * vector[3] - 0.10 * vector[4] + 0.25 * vector[7])
    return torch.from_numpy(vector)


def osm_text_from_entry(raw_entry: Optional[Mapping[str, Any]], fallback: str = "") -> Tuple[str, List[str]]:
    if not isinstance(raw_entry, Mapping):
        return fallback, []
    entry = _summary_entry(raw_entry)
    raw_text = entry.get("osm_text", entry.get("text_v21", ""))
    if isinstance(raw_text, list):
        phrases = [_safe_text(value) for value in raw_text if _safe_text(value)]
        text = " ; ".join(phrases)
    else:
        text = _safe_text(raw_text)
        phrases = []
    raw_phrases = entry.get("phrases", [])
    if isinstance(raw_phrases, list):
        phrases.extend(_safe_text(value) for value in raw_phrases if _safe_text(value))
    if not text:
        summary = entry.get("summary", "")
        text = _safe_text(summary) if isinstance(summary, str) else ""
    if not text and phrases:
        text = " ; ".join(phrases)
    seen: List[str] = []
    for phrase in phrases:
        if phrase not in seen:
            seen.append(phrase)
    return text or fallback, seen


def _coerce_indices(array: np.ndarray, source: Path) -> np.ndarray:
    value = np.asarray(array)
    value = np.squeeze(value)
    if value.ndim != 3:
        raise ValueError(f"Spectral indices in {source} must be 3-D with two bands; got {value.shape}")
    if value.shape[0] == 2:
        value = value.transpose(1, 2, 0)
    elif value.shape[-1] != 2:
        raise ValueError(f"Spectral indices in {source} must contain exactly NDVI/NDWI; got {value.shape}")
    return value.astype(np.float32, copy=False)


def _coerce_channels(
    array: np.ndarray,
    source: Path,
    count: int,
    label: str,
) -> np.ndarray:
    """Return a channel-last array with an exact, unambiguous band count."""

    value = np.asarray(array)
    value = np.squeeze(value)
    if value.ndim != 3:
        raise ValueError(f"{label} in {source} must be 3-D; got {value.shape}")
    if value.shape[0] == int(count):
        value = value.transpose(1, 2, 0)
    elif value.shape[-1] != int(count):
        raise ValueError(
            f"{label} in {source} must contain exactly {count} channels; got {value.shape}"
        )
    return value.astype(np.float32, copy=False)


def _coerce_spectral(array: np.ndarray, source: Path) -> np.ndarray:
    return _coerce_channels(
        array,
        source,
        len(SPECTRAL_CHANNEL_ORDER),
        "Canonical Green/Red/NIR/NDVI/NDWI data",
    )


def _coerce_reflectance(array: np.ndarray, source: Path) -> np.ndarray:
    return _coerce_channels(
        array,
        source,
        len(SURFACE_REFLECTANCE_ORDER),
        "Green/Red/NIR surface reflectance",
    )


def _coerce_mask(array: np.ndarray, source: Path) -> np.ndarray:
    value = np.asarray(array)
    value = np.squeeze(value)
    if value.ndim == 3:
        if value.shape[0] in (1, 2):
            value = np.all(value > 0, axis=0)
        elif value.shape[-1] in (1, 2):
            value = np.all(value > 0, axis=-1)
    if value.ndim != 2:
        raise ValueError(f"Spectral validity mask in {source} must be 2-D; got {value.shape}")
    return (value > 0).astype(np.float32)


def _load_npz_indices(path: Path, record: Mapping[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        key = _safe_text(record.get("indices_key", ""))
        if key and key in data:
            indices = _coerce_indices(data[key], path)
        elif "indices" in data:
            indices = _coerce_indices(data["indices"], path)
        elif "spectral_indices" in data:
            indices = _coerce_indices(data["spectral_indices"], path)
        elif "ndvi" in data and "ndwi" in data:
            indices = np.stack([np.asarray(data["ndvi"]), np.asarray(data["ndwi"])], axis=-1).astype(np.float32)
        elif "NDVI" in data and "NDWI" in data:
            indices = np.stack([np.asarray(data["NDVI"]), np.asarray(data["NDWI"])], axis=-1).astype(np.float32)
        elif "arr_0" in data:
            indices = _coerce_indices(data["arr_0"], path)
        else:
            raise KeyError(f"{path} has no indices, spectral_indices, or NDVI/NDWI arrays")
        embedded_mask = None
        for mask_key in ("valid_mask", "spectral_valid", "mask", "validity"):
            if mask_key in data:
                embedded_mask = _coerce_mask(data[mask_key], path)
                break
    return indices, embedded_mask


def _load_npz_spectral(path: Path, record: Mapping[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        key = _safe_text(record.get("spectral_key", ""))
        if key and key in data:
            channels = _coerce_spectral(data[key], path)
        elif "spectral_channels" in data:
            channels = _coerce_spectral(data["spectral_channels"], path)
        elif "spectral" in data:
            channels = _coerce_spectral(data["spectral"], path)
        elif all(name in data for name in SPECTRAL_CHANNEL_ORDER):
            channels = np.stack(
                [np.asarray(data[name]) for name in SPECTRAL_CHANNEL_ORDER], axis=-1
            ).astype(np.float32)
        elif "arr_0" in data:
            channels = _coerce_spectral(data["arr_0"], path)
        else:
            raise KeyError(
                f"{path} has no canonical five-channel Green/Red/NIR/NDVI/NDWI array"
            )
        embedded_mask = None
        for mask_key in ("valid_mask", "spectral_valid", "mask", "validity"):
            if mask_key in data:
                embedded_mask = _coerce_mask(data[mask_key], path)
                break
    return channels, embedded_mask


def _load_raster(
    path: Path,
    count: Optional[int] = None,
    expected_descriptions: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            f"rasterio is required to read georeferenced spectral raster {path}; "
            "install the project's spectral optional dependencies"
        ) from exc
    with rasterio.open(path) as dataset:
        if count is not None and dataset.count < count:
            raise ValueError(f"{path} has {dataset.count} band(s), expected at least {count}")
        if expected_descriptions is not None:
            descriptions = tuple(dataset.descriptions[: len(expected_descriptions)])
            if descriptions != tuple(expected_descriptions):
                raise ValueError(
                    f"{path} band descriptions are {descriptions}, expected "
                    f"{tuple(expected_descriptions)}"
                )
        indexes = list(range(1, (count or dataset.count) + 1))
        array = dataset.read(indexes)
        masks = dataset.read_masks(indexes)
        valid = np.all(masks > 0, axis=0).astype(np.float32)
    return array, valid


def _load_single_band(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.squeeze(np.load(path, allow_pickle=False)).astype(np.float32), None
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = "arr_0" if "arr_0" in data else data.files[0]
            return np.squeeze(data[key]).astype(np.float32), None
    if suffix in {".tif", ".tiff"}:
        array, mask = _load_raster(path, count=1)
        return array[0].astype(np.float32), mask
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float32), None


def _verify_file_checksum(path: Optional[Path], expected: Any, label: str) -> None:
    """Verify a declared SHA-256 before a cache is admitted to runtime."""

    expected_text = _safe_text(expected).lower()
    if not expected_text:
        return
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Cannot verify {label}; cached file is missing: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_text):
        raise ValueError(f"Invalid declared SHA-256 for {label}: {expected!r}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_text:
        raise ValueError(
            f"{label} checksum mismatch for {path}: expected {expected_text}, got {actual}"
        )


def _validity_aware_resize_indices(
    indices: np.ndarray,
    valid: np.ndarray,
    expected_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize coarse indices without blending cloud/nodata values into data.

    Continuous index numerators and validity weights are bilinearly resized
    together, while the final categorical validity support is nearest-neighbor.
    This is deliberately different from resizing an already-zero-filled raster.
    """

    if indices.ndim != 3 or indices.shape[-1] != len(SPECTRAL_INDEX_ORDER):
        raise ValueError(f"Expected [H,W,2] indices, got {indices.shape}")
    if valid.shape != indices.shape[:2]:
        raise ValueError(f"Index/mask shape mismatch: {indices.shape} versus {valid.shape}")
    target_h, target_w = (int(value) for value in expected_shape)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid spectral target shape: {expected_shape!r}")

    values = torch.from_numpy(indices.transpose(2, 0, 1)).unsqueeze(0).float()
    mask = torch.from_numpy(valid).view(1, 1, *valid.shape).float().clamp(0.0, 1.0)
    weighted = F.interpolate(
        values * mask,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    support = F.interpolate(
        mask,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    nearest = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
    resized = weighted / support.clamp_min(1.0e-6)
    output_valid = ((nearest > 0.5) & (support > 1.0e-6)).float()
    resized = torch.where(output_valid > 0, resized, torch.zeros_like(resized))
    return (
        resized[0].permute(1, 2, 0).numpy().astype(np.float32, copy=False),
        output_valid[0, 0].numpy().astype(np.float32, copy=False),
    )


def _load_legacy_paired_indices(
    sample_record: Mapping[str, Any],
    temporal_key: str,
    manifest: ManifestIndex,
    expected_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the retained schema-less Landsat NDVI/NDWI compatibility cache."""

    temporal_key = str(temporal_key).strip().lower()
    if temporal_key not in {"t1", "t2"}:
        raise ValueError(f"Invalid spectral temporal key: {temporal_key!r}")
    path = _legacy_index_cache_path(sample_record, manifest)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Legacy spectral cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        ndvi = np.asarray(archive[f"ndvi_{temporal_key}"], dtype=np.float32)
        ndwi = np.asarray(archive[f"ndwi_{temporal_key}"], dtype=np.float32)
        valid = np.asarray(archive[f"valid_mask_{temporal_key}"], dtype=np.float32)
    ndvi = np.squeeze(ndvi)
    ndwi = np.squeeze(ndwi)
    valid = np.squeeze(valid)
    if ndvi.ndim != 2 or ndwi.shape != ndvi.shape or valid.shape != ndvi.shape:
        raise ValueError(
            f"Invalid legacy spectral shapes in {path}: "
            f"NDVI={ndvi.shape}, NDWI={ndwi.shape}, mask={valid.shape}"
        )
    finite = np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(valid)
    valid = ((valid > 0.5) & finite).astype(np.float32)
    indices = np.stack(
        [np.nan_to_num(ndvi, nan=0.0), np.nan_to_num(ndwi, nan=0.0)], axis=-1
    ).astype(np.float32)
    observed = indices[valid > 0]
    if observed.size and (float(observed.min()) < -1.0001 or float(observed.max()) > 1.0001):
        raise ValueError(
            f"Valid legacy NDVI/NDWI values in {path} lie outside [-1,1]: "
            f"[{float(observed.min()):.6g}, {float(observed.max()):.6g}]"
        )
    np.clip(indices, -1.0, 1.0, out=indices)
    indices[valid <= 0] = 0.0
    if indices.shape[:2] != tuple(expected_shape):
        indices, valid = _validity_aware_resize_indices(indices, valid, expected_shape)

    # Preserve the canonical five-channel transport tensor without fabricating
    # Green/Red/NIR values.  The first three channels are explicitly unavailable
    # compatibility padding; model code consumes the two index views.
    channels = np.zeros((*expected_shape, len(SPECTRAL_CHANNEL_ORDER)), dtype=np.float32)
    channels[..., 3:5] = indices
    channels[valid <= 0] = 0.0
    return channels, valid


def load_spectral_date(
    record: Mapping[str, Any],
    manifest: ManifestIndex,
    expected_shape: Tuple[int, int],
    *,
    sample_record: Optional[Mapping[str, Any]] = None,
    temporal_key: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load aligned ``[H,W,5]`` physical/index data and explicit validity."""

    if sample_record is not None and _legacy_index_cache_path(sample_record, manifest) is not None:
        return _load_legacy_paired_indices(
            sample_record,
            temporal_key,
            manifest,
            expected_shape,
        )

    paths = spectral_paths(record, manifest)
    _verify_file_checksum(paths["spectral"], record.get("spectral_sha256"), "spectral cache")
    _verify_file_checksum(
        paths["reflectance"],
        record.get("surface_reflectance_sha256", record.get("reflectance_sha256")),
        "surface-reflectance cache",
    )
    _verify_file_checksum(paths["indices"], record.get("indices_sha256"), "index cache")
    _verify_file_checksum(paths["ndvi"], record.get("ndvi_sha256"), "NDVI cache")
    _verify_file_checksum(paths["ndwi"], record.get("ndwi_sha256"), "NDWI cache")
    _verify_file_checksum(paths["mask"], record.get("valid_mask_sha256"), "validity cache")
    embedded_mask: Optional[np.ndarray] = None
    source_path: Optional[Path] = None
    if paths["spectral"] is not None:
        path = paths["spectral"]
        source_path = path
        suffix = path.suffix.lower()
        if suffix == ".npz":
            channels, embedded_mask = _load_npz_spectral(path, record)
        elif suffix == ".npy":
            channels = _coerce_spectral(np.load(path, allow_pickle=False), path)
        elif suffix in {".tif", ".tiff"}:
            raster, embedded_mask = _load_raster(
                path,
                count=len(SPECTRAL_CHANNEL_ORDER),
                expected_descriptions=SPECTRAL_CHANNEL_ORDER,
            )
            channels = _coerce_spectral(raster, path)
        else:
            raise ValueError(f"Unsupported five-channel spectral format: {path}")
    elif paths["reflectance"] is not None:
        reflectance_path = paths["reflectance"]
        suffix = reflectance_path.suffix.lower()
        if suffix == ".npy":
            reflectance = _coerce_reflectance(
                np.load(reflectance_path, allow_pickle=False), reflectance_path
            )
        elif suffix == ".npz":
            with np.load(reflectance_path, allow_pickle=False) as data:
                key = _safe_text(record.get("reflectance_key", ""))
                if key and key in data:
                    raw_reflectance = data[key]
                elif "surface_reflectance" in data:
                    raw_reflectance = data["surface_reflectance"]
                elif "reflectance" in data:
                    raw_reflectance = data["reflectance"]
                elif all(name in data for name in SURFACE_REFLECTANCE_ORDER):
                    raw_reflectance = np.stack(
                        [np.asarray(data[name]) for name in SURFACE_REFLECTANCE_ORDER],
                        axis=-1,
                    )
                else:
                    raise KeyError(f"{reflectance_path} has no Green/Red/NIR array")
                reflectance = _coerce_reflectance(raw_reflectance, reflectance_path)
        elif suffix in {".tif", ".tiff"}:
            raster, reflectance_mask = _load_raster(
                reflectance_path,
                count=len(SURFACE_REFLECTANCE_ORDER),
                expected_descriptions=SURFACE_REFLECTANCE_ORDER,
            )
            reflectance = _coerce_reflectance(raster, reflectance_path)
            embedded_mask = reflectance_mask
        else:
            raise ValueError(f"Unsupported surface-reflectance format: {reflectance_path}")

        if paths["indices"] is not None:
            index_path = paths["indices"]
            if index_path.suffix.lower() == ".npz":
                indices, indices_mask = _load_npz_indices(index_path, record)
            elif index_path.suffix.lower() == ".npy":
                indices = _coerce_indices(
                    np.load(index_path, allow_pickle=False), index_path
                )
                indices_mask = None
            elif index_path.suffix.lower() in {".tif", ".tiff"}:
                raster, indices_mask = _load_raster(
                    index_path,
                    count=2,
                    expected_descriptions=SPECTRAL_INDEX_ORDER,
                )
                indices = _coerce_indices(raster, index_path)
            else:
                raise ValueError(f"Unsupported spectral-index format: {index_path}")
        elif paths["ndvi"] is not None and paths["ndwi"] is not None:
            ndvi, mask_ndvi = _load_single_band(paths["ndvi"])
            ndwi, mask_ndwi = _load_single_band(paths["ndwi"])
            if ndvi.shape != ndwi.shape:
                raise ValueError(f"NDVI/NDWI shape mismatch: {ndvi.shape} versus {ndwi.shape}")
            indices = np.stack([ndvi, ndwi], axis=-1).astype(np.float32)
            mask_ndvi = np.ones(ndvi.shape, dtype=np.float32) if mask_ndvi is None else mask_ndvi
            mask_ndwi = np.ones(ndwi.shape, dtype=np.float32) if mask_ndwi is None else mask_ndwi
            indices_mask = mask_ndvi * mask_ndwi
        else:
            raise ValueError("Surface reflectance is present but NDVI/NDWI are missing")
        if reflectance.shape[:2] != indices.shape[:2]:
            raise ValueError(
                f"Reflectance/index shape mismatch: {reflectance.shape} versus {indices.shape}"
            )
        channels = np.concatenate([reflectance, indices], axis=-1).astype(np.float32)
        if indices_mask is not None:
            embedded_mask = (
                _coerce_mask(indices_mask, paths["indices"] or paths["ndvi"])
                if embedded_mask is None
                else embedded_mask
                * _coerce_mask(indices_mask, paths["indices"] or paths["ndvi"])
            )
    else:
        raise ValueError(
            "Spectral record has no physical Green/Red/NIR cache; indices-only data are rejected"
        )

    explicit_mask: Optional[np.ndarray] = None
    if paths["mask"] is not None:
        explicit_mask, _ = _load_single_band(paths["mask"])
        explicit_mask = _coerce_mask(explicit_mask, paths["mask"])
    finite_mask = np.all(np.isfinite(channels), axis=-1).astype(np.float32)
    valid = finite_mask
    if embedded_mask is not None:
        valid *= _coerce_mask(embedded_mask, source_path or paths["reflectance"])
    if explicit_mask is not None:
        valid *= explicit_mask
    if channels.shape[:2] != tuple(expected_shape) or valid.shape != tuple(expected_shape):
        raise ValueError(
            f"Spectral cache is not aligned to LEVIR source: channels={channels.shape}, "
            f"mask={valid.shape}, expected={tuple(expected_shape)}"
        )
    declared_order = record.get("spectral_channel_order", SPECTRAL_CHANNEL_ORDER)
    if tuple(str(value).lower() for value in declared_order) != SPECTRAL_CHANNEL_ORDER:
        raise ValueError(
            f"Spectral record channel order {declared_order!r} does not match "
            f"{SPECTRAL_CHANNEL_ORDER}"
        )
    index_values = channels[..., 3:5][valid > 0]
    if index_values.size and (
        float(index_values.min()) < -1.0001 or float(index_values.max()) > 1.0001
    ):
        raise ValueError(
            f"Valid NDVI/NDWI values must lie in [-1, 1], got "
            f"[{float(index_values.min()):.6g}, {float(index_values.max()):.6g}]"
        )
    reflectance_values = channels[..., :3][valid > 0]
    if reflectance_values.size:
        allowed_min, allowed_max = SURFACE_REFLECTANCE_VALID_RANGE
        observed_min = float(reflectance_values.min())
        observed_max = float(reflectance_values.max())
        if observed_min < allowed_min - 1e-3 or observed_max > allowed_max + 1e-3:
            raise ValueError(
                "Valid surface reflectance lies outside the scaled physical range "
                f"{SURFACE_REFLECTANCE_VALID_RANGE}: [{observed_min:.6g}, {observed_max:.6g}]"
            )
    epsilon = float(record.get("epsilon", 1e-6))
    green, red, nir = (channels[..., index] for index in range(3))
    ndvi_denominator = nir + red
    ndwi_denominator = green + nir
    formula_valid = (
        (np.abs(ndvi_denominator) > epsilon)
        & (np.abs(ndwi_denominator) > epsilon)
        & (valid > 0)
    )
    if np.any((valid > 0) & ~formula_valid):
        raise ValueError("Valid spectral cache violates the recorded denominator policy")
    expected_indices = np.zeros_like(channels[..., 3:5], dtype=np.float32)
    np.divide(
        nir - red,
        ndvi_denominator,
        out=expected_indices[..., 0],
        where=formula_valid,
    )
    np.divide(
        green - nir,
        ndwi_denominator,
        out=expected_indices[..., 1],
        where=formula_valid,
    )
    np.clip(expected_indices, -1.0, 1.0, out=expected_indices)
    if formula_valid.any():
        formula_error = float(
            np.max(
                np.abs(
                    channels[..., 3:5][formula_valid]
                    - expected_indices[formula_valid]
                )
            )
        )
        if formula_error > 1e-4:
            raise ValueError(
                "Cached indices are not derived from the cached physical bands: "
                f"maximum absolute error {formula_error:.6g}"
            )
    channels[..., 3:5] = np.clip(channels[..., 3:5], -1.0, 1.0)
    channels[valid <= 0] = 0.0
    return channels.astype(np.float32, copy=False), valid.astype(np.float32, copy=False)


class SpectralCache:
    """Tiny per-process LRU cache; safe to copy into DataLoader workers."""

    def __init__(self, max_sources: int = 2):
        self.max_sources = max(0, int(max_sources))
        self._values: "OrderedDict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def get(self, key: Tuple[str, str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: Tuple[str, str], value: Tuple[np.ndarray, np.ndarray]) -> None:
        if self.max_sources <= 0:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_sources * 2:
            self._values.popitem(last=False)

    def __getstate__(self) -> Dict[str, Any]:
        # Worker processes start empty rather than duplicating cached source arrays.
        return {"max_sources": self.max_sources, "_values": OrderedDict()}


class OSMGeoJSONCache:
    """Per-worker immutable cache for source/date historical GeoJSON snapshots."""

    def __init__(self, max_sources: int = 1):
        # Each retained source has at most two snapshots (T1/T2). Keeping the
        # setting in source-pair units makes its memory meaning predictable.
        self.max_sources = max(0, int(max_sources))
        self._values: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()

    def get_or_load(self, key: Tuple[str, str], path: Path) -> Dict[str, Any]:
        cached = self._values.get(key)
        if cached is not None:
            self._values.move_to_end(key)
            return cached
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read cached historical OSM GeoJSON {path}: {exc}") from exc
        if (
            not isinstance(document, Mapping)
            or document.get("type") != "FeatureCollection"
            or not isinstance(document.get("features"), list)
        ):
            raise ValueError(f"Cached historical OSM file is not a FeatureCollection: {path}")
        value = dict(document)
        if self.max_sources > 0:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.max_sources * 2:
                self._values.popitem(last=False)
        return value

    def __getstate__(self) -> Dict[str, Any]:
        # Spawned DataLoader workers own their cache; never duplicate a cache
        # populated during a main-process smoke test.
        return {"max_sources": self.max_sources, "_values": OrderedDict()}


def load_index_normalization(
    mode: str,
    stats_path: str = "",
    spectral_manifest_sha256: str = "",
) -> Dict[str, Any]:
    """Load immutable TRAIN-only spectral normalization statistics.

    Modes
    -----
    natural
        Leave all channels unchanged.
    train_stats
        One global five-channel (or legacy NDVI/NDWI-only) mean/std vector.
    sensor_train_stats
        Per-Landsat-family NDVI/NDWI statistics.  This reproduces the retained
        controlled multimodal experiments without allowing VAL/TEST pixels to
        influence normalization.
    """

    normalized_mode = _safe_text(mode).lower() or "natural"
    if normalized_mode in {"natural", "none", "raw"}:
        return {
            "mode": "natural",
            "path": "",
            "mean": np.zeros(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32),
            "std": np.ones(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32),
            "channel_order": SPECTRAL_CHANNEL_ORDER,
            "sensor_stats": {},
            "document": {},
        }

    global_aliases = {"train_stats", "train", "standardize"}
    sensor_aliases = {
        "sensor_train_stats",
        "per_sensor_train_stats",
        "sensor_standardize",
        "per_sensor",
    }
    if normalized_mode not in global_aliases | sensor_aliases:
        raise ValueError(
            "Unsupported LEVIR index normalization "
            f"{mode!r}; expected natural, train_stats, or sensor_train_stats"
        )
    if not _safe_text(stats_path):
        raise ValueError(
            f"index_normalization={normalized_mode!r} requires a normalization stats JSON"
        )
    path = Path(stats_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Spectral normalization statistics not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"Normalization statistics root must be an object: {path}")
    if _safe_text(document.get("split_used", "")).lower() != "train":
        raise ValueError("Spectral normalization statistics must declare split_used='train'")
    if document.get("validation_or_test_used") is not False:
        raise ValueError(
            "Spectral normalization statistics must declare validation_or_test_used=false"
        )
    stats_manifest_hash = _safe_text(document.get("source_manifest_sha256", ""))
    if spectral_manifest_sha256 and stats_manifest_hash != spectral_manifest_sha256:
        raise ValueError(
            "Normalization statistics source_manifest_sha256 does not match the configured "
            "spectral manifest"
        )

    if normalized_mode in sensor_aliases:
        raw_sensors = document.get("sensors")
        if not isinstance(raw_sensors, Mapping):
            raise ValueError("sensor_train_stats requires a 'sensors' object")
        expected = {
            "landsat-5": 1,
            "landsat-7": 2,
            "landsat-8": 3,
        }
        sensor_stats: Dict[int, Dict[str, np.ndarray]] = {}
        for sensor_name, sensor_id in expected.items():
            record = raw_sensors.get(sensor_name)
            if not isinstance(record, Mapping):
                raise ValueError(f"Missing TRAIN statistics for {sensor_name}")
            try:
                ndvi_mean = float(record["ndvi_mean"])
                ndvi_std = float(record["ndvi_std"])
                ndwi_mean = float(record["ndwi_mean"])
                ndwi_std = float(record["ndwi_std"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid TRAIN statistics for {sensor_name}") from exc
            values = np.asarray(
                [ndvi_mean, ndvi_std, ndwi_mean, ndwi_std], dtype=np.float32
            )
            if not np.all(np.isfinite(values)) or ndvi_std <= 0.0 or ndwi_std <= 0.0:
                raise ValueError(f"Non-finite/non-positive TRAIN statistics for {sensor_name}")
            mean = np.zeros(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32)
            std = np.ones(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32)
            mean[3] = ndvi_mean
            mean[4] = ndwi_mean
            std[3] = ndvi_std
            std[4] = ndwi_std
            sensor_stats[int(sensor_id)] = {"mean": mean, "std": std}
        return {
            "mode": "sensor_train_stats",
            "path": str(path),
            "mean": np.zeros(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32),
            "std": np.ones(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32),
            "channel_order": SPECTRAL_CHANNEL_ORDER,
            "sensor_stats": sensor_stats,
            "document": dict(document),
        }

    order = document.get("channel_order")
    legacy_index_only = order is None
    if legacy_index_only:
        index_order = document.get("index_order", ["NDVI", "NDWI"])
        if [str(value).upper() for value in index_order] != ["NDVI", "NDWI"]:
            raise ValueError(
                f"Normalization index_order must be ['NDVI', 'NDWI'], got {index_order!r}"
            )
    elif tuple(str(value).lower() for value in order) != SPECTRAL_CHANNEL_ORDER:
        raise ValueError(
            f"Normalization channel_order must be {SPECTRAL_CHANNEL_ORDER}, got {order!r}"
        )
    try:
        mean = np.asarray(document["mean"], dtype=np.float32)
        std = np.asarray(document["std"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid mean/std in normalization statistics: {path}") from exc
    expected_count = 2 if legacy_index_only else len(SPECTRAL_CHANNEL_ORDER)
    if mean.shape != (expected_count,) or std.shape != (expected_count,):
        raise ValueError(
            f"Normalization mean/std must contain {expected_count} values, "
            f"got {mean.shape}/{std.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Normalization mean/std must be finite and each std must be positive")
    if legacy_index_only:
        expanded_mean = np.zeros(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32)
        expanded_std = np.ones(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float32)
        expanded_mean[3:5] = mean
        expanded_std[3:5] = std
        mean, std = expanded_mean, expanded_std
    return {
        "mode": "train_stats",
        "path": str(path),
        "mean": mean,
        "std": std,
        "channel_order": SPECTRAL_CHANNEL_ORDER,
        "legacy_index_only": legacy_index_only,
        "sensor_stats": {},
        "document": dict(document),
    }


class DeterministicTextTokenizer:
    """Small offline tokenizer matching the minimal HuggingFace return contract."""

    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = max(1000, int(vocab_size))
        self.pad_token_id = 0
        self.unk_token_id = 100
        self.cls_token_id = 101
        self.sep_token_id = 102

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return 103 + (int.from_bytes(digest, "big") % (self.vocab_size - 103))

    def __call__(
        self,
        text: Any,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 40,
        return_tensors: str = "pt",
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        texts = [text] if isinstance(text, str) else list(text)
        all_ids: List[List[int]] = []
        all_masks: List[List[int]] = []
        for value in texts:
            words = re.findall(r"[A-Za-z0-9_]+|[^\w\s]", _safe_text(value).lower())
            ids = [self.cls_token_id]
            ids.extend(self._token_id(word) for word in words[: max(0, int(max_length) - 2)])
            if len(ids) < int(max_length):
                ids.append(self.sep_token_id)
            ids = ids[: int(max_length)]
            mask = [1] * len(ids)
            pad = int(max_length) - len(ids)
            ids.extend([self.pad_token_id] * pad)
            mask.extend([0] * pad)
            all_ids.append(ids)
            all_masks.append(mask)
        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }


def build_levir_tokenizer(
    tokenizer: Any = None,
    mode: str = "simple",
    name: str = "bert-base-uncased",
    local_files_only: bool = True,
    fallback_to_simple: bool = False,
    vocab_size: int = 30522,
) -> Any:
    if tokenizer is not None:
        return tokenizer
    normalized_mode = _safe_text(mode).lower() or "simple"
    if normalized_mode in {"simple", "fixed", "offline", "deterministic", "none"}:
        return DeterministicTextTokenizer(vocab_size=vocab_size)
    if normalized_mode not in {"hf", "huggingface", "legacy", "auto"}:
        raise ValueError(f"Unsupported LEVIR tokenizer mode: {mode!r}")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, local_files_only=bool(local_files_only))
    except Exception:
        if fallback_to_simple:
            return DeterministicTextTokenizer(vocab_size=vocab_size)
        raise


__all__ = [
    "DeterministicTextTokenizer",
    "ManifestIndex",
    "OSMGeoJSONCache",
    "OSM_AVAILABLE_STATUSES",
    "SPECTRAL_AVAILABLE_STATUSES",
    "SPECTRAL_CHANNEL_ORDER",
    "SPECTRAL_INDEX_ORDER",
    "SURFACE_REFLECTANCE_ORDER",
    "SURFACE_REFLECTANCE_VALID_RANGE",
    "SpectralCache",
    "assess_temporal_source",
    "build_levir_tokenizer",
    "load_spectral_date",
    "load_index_normalization",
    "natural_sample_sort_key",
    "normalize_sample_key",
    "osm_struct_from_entry",
    "osm_text_from_entry",
    "spectral_paths",
]
