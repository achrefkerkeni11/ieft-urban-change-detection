#!/usr/bin/env python
r"""Build per-Landsat NDVI/NDWI normalization statistics from TRAIN only.

This utility is specifically for the retained ``data_spectral_v2`` compatibility
cache, where each source NPZ contains NDVI/NDWI plus validity masks but not the
underlying Green/Red/NIR reflectance arrays.  It never uses VAL/TEST pixels and
never modifies the spectral cache.

Example:

    python prepare_legacy_spectral_train_stats.py \
      --manifest data_spectral_v2/manifest.json \
      --output data_spectral_v2/train_sensor_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


SENSOR_NAMES = ("landsat-5", "landsat-7", "landsat-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def clean_sensor(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = str(value or "").lower().replace("_", "-")
    if "landsat-5" in text or "lt05" in text or text in {"tm", "landsat tm"}:
        return "landsat-5"
    if "landsat-7" in text or "le07" in text or "etm" in text:
        return "landsat-7"
    if "landsat-8" in text or "lc08" in text or "oli" in text:
        return "landsat-8"
    return ""


def manifest_records(document: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    """Yield ``(source_id, record)`` from both retained and newer schemas."""

    if isinstance(document, Mapping) and isinstance(document.get("samples"), list):
        for value in document["samples"]:
            if not isinstance(value, Mapping):
                continue
            source_id = str(
                value.get("source_pair_id", value.get("source_id", Path(str(value.get("filename", ""))).stem))
            )
            if source_id:
                yield source_id, value
        return
    if isinstance(document, Mapping):
        for key, value in document.items():
            if isinstance(value, Mapping):
                source_id = str(value.get("source_pair_id", value.get("source_id", key)))
                yield source_id, value
        return
    raise ValueError("Unsupported spectral manifest schema")


def resolve_raster_path(
    manifest_path: Path, project_root: Path, record: Mapping[str, Any]
) -> Optional[Path]:
    raw = str(record.get("raster_path", "")).strip()
    if not raw:
        return None
    candidate = Path(raw)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend((project_root / candidate, manifest_path.parent / candidate))
    for value in candidates:
        if value.is_file():
            return value.resolve()
    return candidates[0].resolve() if candidates else None


def new_accumulator() -> Dict[str, float]:
    return {"count": 0, "sum": 0.0, "sumsq": 0.0}


def update(acc: MutableMapping[str, float], values: np.ndarray) -> None:
    if values.size == 0:
        return
    values64 = values.astype(np.float64, copy=False)
    acc["count"] += int(values64.size)
    acc["sum"] += float(values64.sum(dtype=np.float64))
    acc["sumsq"] += float(np.square(values64).sum(dtype=np.float64))


def valid_values(array: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(array)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        finite &= np.isfinite(valid_mask)
        finite &= valid_mask > 0.5
    return array[finite]


def finalize(acc: Mapping[str, float], label: str) -> Tuple[float, float, int]:
    count = int(acc["count"])
    if count <= 1:
        raise RuntimeError(f"Not enough valid TRAIN values for {label}: n={count}")
    mean = float(acc["sum"]) / count
    variance = max(float(acc["sumsq"]) / count - mean * mean, 1.0e-12)
    return mean, math.sqrt(variance), count


def build_stats(manifest_path: Path, project_root: Path) -> Dict[str, Any]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    accumulators = {
        sensor: {"ndvi": new_accumulator(), "ndwi": new_accumulator()}
        for sensor in SENSOR_NAMES
    }
    readable_train_sources = 0
    skipped_nontrain = 0
    missing_raster = 0
    unknown_sensor_dates = []
    load_errors = []

    for source_id, record in manifest_records(document):
        split = str(record.get("split", "")).strip().lower()
        is_train = split == "train" or source_id.startswith("train_")
        if not is_train:
            skipped_nontrain += 1
            continue
        raster_path = resolve_raster_path(manifest_path, project_root, record)
        if raster_path is None or not raster_path.is_file():
            missing_raster += 1
            continue
        try:
            with np.load(raster_path, allow_pickle=False) as archive:
                for temporal_key in ("t1", "t2"):
                    date_record = record.get(temporal_key, {})
                    if not isinstance(date_record, Mapping):
                        continue
                    status = str(date_record.get("status", "")).lower()
                    if not status.startswith("ok"):
                        continue
                    sensor = clean_sensor(
                        date_record.get("sensor", date_record.get("sensors", ""))
                    )
                    if not sensor:
                        unknown_sensor_dates.append(f"{source_id}:{temporal_key}")
                        continue
                    ndvi_key = f"ndvi_{temporal_key}"
                    ndwi_key = f"ndwi_{temporal_key}"
                    valid_key = f"valid_mask_{temporal_key}"
                    if ndvi_key not in archive or ndwi_key not in archive:
                        raise KeyError(f"missing {ndvi_key}/{ndwi_key}")
                    ndvi = np.asarray(archive[ndvi_key], dtype=np.float32)
                    ndwi = np.asarray(archive[ndwi_key], dtype=np.float32)
                    valid = (
                        np.asarray(archive[valid_key], dtype=np.float32)
                        if valid_key in archive
                        else None
                    )
                    update(accumulators[sensor]["ndvi"], valid_values(ndvi, valid))
                    update(accumulators[sensor]["ndwi"], valid_values(ndwi, valid))
            readable_train_sources += 1
        except Exception as exc:  # report every real data problem; never hide it
            load_errors.append(f"{source_id}: {type(exc).__name__}: {exc}")

    if load_errors:
        raise RuntimeError(
            f"Failed to read {len(load_errors)} TRAIN spectral raster(s). First errors: "
            + "; ".join(load_errors[:10])
        )
    if unknown_sensor_dates:
        raise RuntimeError(
            f"Found {len(unknown_sensor_dates)} valid TRAIN date(s) with unknown sensor. "
            "First: " + ", ".join(unknown_sensor_dates[:10])
        )

    sensors: Dict[str, Any] = {}
    for sensor in SENSOR_NAMES:
        ndvi_mean, ndvi_std, ndvi_count = finalize(
            accumulators[sensor]["ndvi"], f"{sensor} NDVI"
        )
        ndwi_mean, ndwi_std, ndwi_count = finalize(
            accumulators[sensor]["ndwi"], f"{sensor} NDWI"
        )
        sensors[sensor] = {
            "ndvi_mean": ndvi_mean,
            "ndvi_std": ndvi_std,
            "ndvi_count": ndvi_count,
            "ndwi_mean": ndwi_mean,
            "ndwi_std": ndwi_std,
            "ndwi_count": ndwi_count,
        }

    return {
        "schema": "ieft.legacy_spectral_sensor_train_stats.v1",
        "split_used": "train",
        "validation_or_test_used": False,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "index_order": ["NDVI", "NDWI"],
        "normalization": "per_sensor_zscore",
        "invalid_pixel_policy": "masked_and_zero_after_normalization",
        "readable_train_sources": readable_train_sources,
        "missing_train_rasters": missing_raster,
        "skipped_nontrain_records": skipped_nontrain,
        "sensors": sensors,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root used to resolve retained raster_path entries (default: current directory)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    stats = build_stats(manifest, project_root)
    atomic_write_json(output, stats)
    print(f"[PASS] TRAIN-only sensor statistics written to: {output}")
    print(f"[INFO] source manifest SHA-256: {stats['source_manifest_sha256']}")
    print(f"[INFO] readable TRAIN sources: {stats['readable_train_sources']}")
    for sensor in SENSOR_NAMES:
        record = stats["sensors"][sensor]
        print(
            f"[INFO] {sensor}: "
            f"NDVI mean/std={record['ndvi_mean']:.6f}/{record['ndvi_std']:.6f} "
            f"(n={record['ndvi_count']:,}) | "
            f"NDWI mean/std={record['ndwi_mean']:.6f}/{record['ndwi_std']:.6f} "
            f"(n={record['ndwi_count']:,})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
