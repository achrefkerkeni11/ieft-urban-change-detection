#!/usr/bin/env python
"""Prepare genuine bitemporal multispectral rasters for LEVIR-CD.

External providers are used only by this offline command.  Runtime dataset and
model code read cached Green/Red/NIR surface reflectance plus derived
NDVI/McFeeters-NDWI without network I/O.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from IEFT.levir_metadata import (
    COORDINATE_SOURCE_URL,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    LEVIRValidationError,
    atomic_write_json,
    build_source_records,
    load_coordinate_json,
    manifest_samples,
    natural_sample_sort_key,
    normalize_sample_key,
    path_relative_to,
    sha256_file,
    validate_local_levir,
)
from IEFT.spectral_indices import (
    ACCEPTABLE_STATUSES,
    AuthenticationRequired,
    DENOMINATOR_POLICY,
    EARTH_ENGINE_REGISTRATION_URL,
    EarthEngineLandsatProvider,
    FAILED_STATUSES,
    FORMULAE,
    INDEX_ORDER,
    LANDSAT_OFFSET,
    LANDSAT_SCALE,
    LANDSAT_7_SLC_FAILURE_DATE,
    LANDSAT_SENSORS,
    NDWI_VARIANT,
    PrecomputedProvider,
    ProviderUnavailable,
    SPECTRAL_CHANNEL_ORDER,
    SPECTRAL_SCHEMA_VERSION,
    SURFACE_REFLECTANCE_ORDER,
    SURFACE_REFLECTANCE_VALID_RANGE,
    SpectralProvider,
    validate_cached_rasters,
)


# These statuses describe an interrupted or unavailable execution context and
# are safe to retry without revisiting scientifically terminal exclusions.
TRANSIENT_FAILURE_STATUSES = {"authentication_required", "failed_retryable"}
MANIFEST_FILENAME = "manifest.json"


def retry_selected(status: Any, *, retry_failed: bool, retry_transient: bool) -> bool:
    """Return whether a previously failed date record should be regenerated."""

    if status not in FAILED_STATUSES:
        return True
    return bool(retry_failed) or (
        bool(retry_transient) and status in TRANSIENT_FAILURE_STATUSES
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_provider(args: argparse.Namespace) -> SpectralProvider:
    if args.provider == "precomputed":
        if not args.precomputed_source_manifest:
            raise ProviderUnavailable("--precomputed-source-manifest is required for provider=precomputed")
        return PrecomputedProvider(Path(args.precomputed_source_manifest))
    if args.provider == "earth_engine_landsat":
        return EarthEngineLandsatProvider(
            project=args.ee_project,
            high_volume=args.ee_high_volume,
            timeout=args.timeout,
            sensor_policy=args.sensor_policy,
        )
    raise ProviderUnavailable(f"Unsupported spectral provider: {args.provider}")


def get_coordinate_path(args: argparse.Namespace) -> Path:
    if not args.coords_json:
        raise FileNotFoundError(
            "--coords-json is required. Download the official file from " + COORDINATE_SOURCE_URL
        )
    path = Path(args.coords_json).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Coordinate JSON not found: {path}")
    return path


def common_metadata(args: argparse.Namespace, coordinate_path: Path) -> Dict[str, Any]:
    earth_engine_project = (
        str(args.ee_project).strip()
        if args.provider == "earth_engine_landsat"
        else None
    )
    return {
        "coordinate_source_url": COORDINATE_SOURCE_URL,
        "coordinate_json": coordinate_path.name,
        "coordinate_sha256": sha256_file(coordinate_path),
        "provider": args.provider,
        "earth_engine_project": earth_engine_project,
        "date_policy": "published_month",
        "timestamp_policy": args.timestamp_policy,
        "initial_window": "complete published calendar month",
        "expand_step_days": args.expand_step_days,
        "max_window_days": args.max_window_days,
        "composite": args.composite,
        "minimum_valid_fraction": args.min_valid_fraction,
        "sensor_policy": (
            args.sensor_policy
            if args.provider == "earth_engine_landsat"
            else "precomputed_declared_source"
        ),
        "spectral_channel_order": list(SPECTRAL_CHANNEL_ORDER),
        "surface_reflectance_order": list(SURFACE_REFLECTANCE_ORDER),
        "surface_reflectance_units": "scaled unitless surface reflectance",
        "surface_reflectance_valid_range": list(SURFACE_REFLECTANCE_VALID_RANGE),
        "index_band_order": list(INDEX_ORDER),
        "ndwi_variant": NDWI_VARIANT,
        "formulae": FORMULAE,
        "denominator_policy": DENOMINATOR_POLICY,
        "epsilon": args.epsilon,
        "target_shape": [args.height, args.width],
        "target_crs": "EPSG:4326",
        "alignment_assumption": "north_up_bbox_linear",
        "continuous_resampling": "bilinear",
        "mask_resampling": "nearest",
        "generated_at": utc_now(),
        "landsat7_slc_failure_date": LANDSAT_7_SLC_FAILURE_DATE.isoformat(),
        "landsat7_slc_measurement": (
            "QA_RADSAT bit 9 per scene plus dropped-any, composite-recovered, "
            "and output-loss fractions"
        ),
    }


def skeleton_manifest(
    records: Sequence[Mapping[str, Any]], args: argparse.Namespace, coordinate_path: Path
) -> Dict[str, Any]:
    samples = []
    for source in records:
        item = copy.deepcopy(dict(source))
        for temporal_key in ("t1", "t2"):
            item[temporal_key].update(
                {
                    "status": "pending",
                    "provider": args.provider,
                    "earth_engine_project": (
                        str(args.ee_project).strip()
                        if args.provider == "earth_engine_landsat"
                        else None
                    ),
                    "valid_fraction": None,
                    "invalid_fraction": None,
                    "spectral_channel_order": list(SPECTRAL_CHANNEL_ORDER),
                    "surface_reflectance_order": list(SURFACE_REFLECTANCE_ORDER),
                    "surface_reflectance_units": "scaled unitless surface reflectance",
                    "surface_reflectance_valid_range": list(
                        SURFACE_REFLECTANCE_VALID_RANGE
                    ),
                    "spectral_path": "",
                    "valid_mask_path": "",
                    "spectral_sha256": "",
                    "valid_mask_sha256": "",
                    "error": None,
                    "generated_at": None,
                }
            )
        samples.append(item)
    return {
        "schema_version": SPECTRAL_SCHEMA_VERSION,
        "metadata": common_metadata(args, coordinate_path),
        "samples": samples,
    }


def merge_resume(current: Dict[str, Any], previous: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return current
    if previous.get("schema_version") != SPECTRAL_SCHEMA_VERSION:
        raise LEVIRValidationError(
            f"Cannot resume spectral schema {previous.get('schema_version')!r}; "
            f"expected {SPECTRAL_SCHEMA_VERSION}"
        )
    previous_meta = previous.get("metadata", {})
    current_meta = current["metadata"]
    invariant_keys = (
        "coordinate_sha256", "provider", "date_policy", "timestamp_policy",
        "expand_step_days", "max_window_days", "composite", "sensor_policy",
        "spectral_channel_order", "surface_reflectance_order",
        "surface_reflectance_units", "surface_reflectance_valid_range",
        "ndwi_variant", "formulae",
        "denominator_policy", "epsilon",
        "minimum_valid_fraction", "earth_engine_project", "target_shape",
        "target_crs",
    )
    mismatches = []
    for key in invariant_keys:
        prior_value = previous_meta.get(key)
        current_value = current_meta.get(key)
        # Releases before timestamp-policy was exposed always used month_end.
        if key == "timestamp_policy" and prior_value is None:
            prior_value = "month_end"
        if key == "timestamp_policy" and current_value is None:
            current_value = "month_end"
        if prior_value != current_value:
            mismatches.append(key)

    # Releases before Earth Engine project provenance was persisted can be
    # bound once to a project supplied explicitly by the resume command. This
    # migration is intentionally manifest-only: legacy cached date records and
    # their GeoTIFFs are preserved byte-for-byte and remain visibly unlabelled.
    provenance_migrations = copy.deepcopy(
        previous_meta.get("provenance_migration_history", [])
    )
    if not isinstance(provenance_migrations, list):
        raise LEVIRValidationError(
            "Spectral provenance_migration_history must be a list"
        )
    if "earth_engine_project" in mismatches:
        prior_project = previous_meta.get("earth_engine_project")
        current_project = current_meta.get("earth_engine_project")
        if (
            previous_meta.get("provider") == "earth_engine_landsat"
            and current_meta.get("provider") == "earth_engine_landsat"
            and prior_project in (None, "")
            and isinstance(current_project, str)
            and bool(current_project.strip())
        ):
            migrated_at = utc_now()
            legacy_acceptable_record_ids = sorted(
                f"{item['filename']}:{date_key}"
                for item in manifest_samples(previous)
                for date_key in ("t1", "t2")
                if item.get(date_key, {}).get("status") in ACCEPTABLE_STATUSES
            )
            mismatches.remove("earth_engine_project")
            provenance_migrations.append(
                {
                    "migration": "earth_engine_project_manifest_binding_v1",
                    "field": "earth_engine_project",
                    "from": None if prior_project is None else prior_project,
                    "to": current_project,
                    "reason": (
                        "legacy Earth Engine manifest omitted the project and "
                        "the resume command supplied a nonempty project"
                    ),
                    "legacy_acceptable_date_records": len(
                        legacy_acceptable_record_ids
                    ),
                    "legacy_unlabelled_date_record_ids": (
                        legacy_acceptable_record_ids
                    ),
                    "cached_date_records_modified": False,
                    "cached_rasters_rewritten": False,
                    "migrated_at": migrated_at,
                }
            )
    if provenance_migrations:
        current["metadata"]["provenance_migration_history"] = (
            provenance_migrations
        )
    # Migrate science metadata only when no acceptable raster exists. Never
    # relabel output that was generated under a prior policy/formula revision.
    previous_has_output = any(
        item.get(date_key, {}).get("status") in ACCEPTABLE_STATUSES
        for item in manifest_samples(previous)
        for date_key in ("t1", "t2")
    )
    safely_migrated = []
    if not previous_has_output:
        if (
            "sensor_policy" in mismatches
            and previous_meta.get("sensor_policy") == "consistent_or_harmonized"
            and current_meta.get("sensor_policy")
            in {"landsat7_consistent", "precomputed_declared_source"}
        ):
            mismatches.remove("sensor_policy")
            safely_migrated.append("sensor_policy")
            current["metadata"]["migrated_from_sensor_policy"] = (
                "consistent_or_harmonized (no acceptable rasters existed)"
            )
        formula_revision_keys = {"formulae", "denominator_policy"}
        if set(mismatches).issubset(formula_revision_keys):
            safely_migrated.extend(mismatches)
            mismatches = []
        if safely_migrated:
            current["metadata"]["science_metadata_migration"] = {
                "fields": sorted(set(safely_migrated)),
                "reason": "no acceptable rasters existed",
                "migrated_at": utc_now(),
            }
    if mismatches:
        raise LEVIRValidationError(
            "Resume configuration differs for: " + ", ".join(mismatches)
        )
    old = {item["filename"]: item for item in manifest_samples(previous)}
    for sample in current["samples"]:
        prior = old.get(sample["filename"])
        if not prior:
            continue
        for key in ("split", "region_id", "bbox"):
            if prior.get(key) != sample.get(key):
                raise LEVIRValidationError(f"Resume mapping mismatch: {sample['filename']} {key}")
        for temporal_key in ("t1", "t2"):
            prior_date = prior.get(temporal_key, {})
            if prior_date.get("image_month") != sample[temporal_key]["image_month"]:
                raise LEVIRValidationError(f"Resume date mismatch: {sample['filename']} {temporal_key}")
            if prior_date.get("status") in ACCEPTABLE_STATUSES | FAILED_STATUSES:
                sample[temporal_key] = copy.deepcopy(prior_date)
    current["metadata"]["resumed_at"] = utc_now()
    return current


def selected(sample: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if args.split and sample["split"] not in args.split:
        return False
    if args.region and int(sample["region_id"]) not in args.region:
        return False
    if args.sample:
        names = {normalize_sample_key(value, keep_extension=True) for value in args.sample}
        if sample["filename"] not in names:
            return False
    return True


def provider_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "height": args.height,
        "width": args.width,
        "epsilon": args.epsilon,
        "min_valid_fraction": args.min_valid_fraction,
        "expand_step_days": args.expand_step_days,
        "max_window_days": args.max_window_days,
        "composite": args.composite,
        "sensor_policy": args.sensor_policy,
    }


def cached_record_error(output_root: Path, record: Mapping[str, Any], shape: Tuple[int, int]) -> Optional[str]:
    if record.get("status") not in ACCEPTABLE_STATUSES:
        return None
    try:
        spectral = output_root / str(record.get("spectral_path", ""))
        valid = output_root / str(record.get("valid_mask_path", ""))
        report = validate_cached_rasters(spectral, valid, shape)
        if abs(float(record.get("valid_fraction", -1.0)) - report["valid_fraction"]) > 1e-6:
            return "cached valid fraction differs from manifest"
        for key in ("spectral_sha256", "valid_mask_sha256"):
            expected = record.get(key)
            if expected and expected != report.get(key):
                return f"cached {key} differs from manifest"
    except Exception as exc:
        return str(exc)
    return None


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, AuthenticationRequired):
        return "authentication_required"
    if isinstance(exc, (ProviderUnavailable, FileNotFoundError)):
        return "source_unavailable"
    if isinstance(exc, (ValueError, LEVIRValidationError)):
        return "failed_permanent"
    return "failed_retryable"


def make_blocked_record(base: Mapping[str, Any], status: str, error: str) -> Dict[str, Any]:
    record = copy.deepcopy(dict(base))
    record.update(
        {
            "status": status,
            "error": error,
            "valid_fraction": None,
            "invalid_fraction": None,
            "spectral_path": "",
            "valid_mask_path": "",
            "spectral_sha256": "",
            "valid_mask_sha256": "",
            "generated_at": utc_now(),
        }
    )
    return record


def coverage_report(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    samples = manifest_samples(manifest)
    by_split: Dict[str, Counter[str]] = defaultdict(Counter)
    by_region: Dict[str, Counter[str]] = defaultdict(Counter)
    valid_fractions: Dict[str, List[float]] = defaultdict(list)
    valid_fractions_by_split_date: Dict[str, List[float]] = defaultdict(list)
    slc_effect_by_split_date: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    expansion = Counter()
    sensors = Counter()
    exclusion_reasons = Counter()
    slc_scene_modes = Counter()
    paired = Counter()
    for sample in samples:
        for temporal_key in ("t1", "t2"):
            record = sample[temporal_key]
            status = str(record.get("status", "missing"))
            by_split[sample["split"]][f"{temporal_key}:{status}"] += 1
            by_region[str(sample["region_id"])][f"{temporal_key}:{status}"] += 1
            if record.get("valid_fraction") is not None:
                valid_fractions[sample["split"]].append(float(record["valid_fraction"]))
                valid_fractions_by_split_date[
                    f"{sample['split']}:{temporal_key}"
                ].append(float(record["valid_fraction"]))
            expansion[str(record.get("expansion_days", 0))] += int(status in ACCEPTABLE_STATUSES)
            for sensor in record.get("sensors", []):
                sensors[str(sensor)] += 1
            if status not in ACCEPTABLE_STATUSES:
                exclusion_reasons[
                    f"{sample['split']}:{temporal_key}:{status}"
                ] += 1
            slc_key = f"{sample['split']}:{temporal_key}"
            for metric in (
                "aligned_slc_dropped_any_fraction",
                "aligned_slc_dropped_output_loss_fraction",
                "aligned_slc_dropped_recovered_fraction",
            ):
                if record.get(metric) is not None:
                    slc_effect_by_split_date[slc_key][metric].append(
                        float(record[metric])
                    )
            for scene in record.get("scene_metadata", []):
                if scene.get("slc_mode"):
                    slc_scene_modes[str(scene["slc_mode"])] += 1
        paired[sample["split"]] += int(
            sample["t1"].get("status") in ACCEPTABLE_STATUSES
            and sample["t2"].get("status") in ACCEPTABLE_STATUSES
        )
    return {
        "schema_version": SPECTRAL_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "total_samples": len(samples),
        "by_split": {key: dict(sorted(value.items())) for key, value in sorted(by_split.items())},
        "by_region": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_region.items(), key=lambda item: int(item[0]))
        },
        "paired_acceptable_by_split": dict(sorted(paired.items())),
        "valid_fraction_by_split": {
            key: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "minimum": float(np.min(values)) if values else None,
                "maximum": float(np.max(values)) if values else None,
            }
            for key, values in sorted(valid_fractions.items())
        },
        "valid_fraction_by_split_date": {
            key: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "minimum": float(np.min(values)) if values else None,
                "maximum": float(np.max(values)) if values else None,
            }
            for key, values in sorted(valid_fractions_by_split_date.items())
        },
        "landsat7_slc_effect_by_split_date": {
            key: {
                metric: {
                    "count": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "maximum": float(np.max(values)) if values else None,
                }
                for metric, values in sorted(metrics.items())
            }
            for key, metrics in sorted(slc_effect_by_split_date.items())
        },
        "landsat7_scene_modes": dict(sorted(slc_scene_modes.items())),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "accepted_window_expansion_days": dict(sorted(expansion.items(), key=lambda item: int(item[0]))),
        "sensor_date_records": dict(sorted(sensors.items())),
    }


def write_outputs(output_root: Path, manifest: Mapping[str, Any]) -> None:
    atomic_write_json(output_root / MANIFEST_FILENAME, manifest)
    atomic_write_json(output_root / "generation_report.json", coverage_report(manifest))


def prepare_common(args: argparse.Namespace) -> Tuple[Path, List[Dict[str, Any]], Path]:
    coordinate_path = get_coordinate_path(args)
    coordinates = load_coordinate_json(coordinate_path)
    if args.data_root:
        validate_local_levir(
            args.data_root, coordinates, inspect_images=not args.skip_image_validation
        )
    records = build_source_records(coordinates, args.timestamp_policy)
    return coordinate_path, records, Path(args.output_root).expanduser().resolve()


def earth_engine_user_actions(project: str) -> List[str]:
    project = str(project or "YOUR_GOOGLE_CLOUD_PROJECT").strip()
    return [
        f"gcloud services enable earthengine.googleapis.com --project={project}",
        "earthengine authenticate",
        f"earthengine set_project {project}",
        EARTH_ENGINE_REGISTRATION_URL.format(project=project),
    ]


def spectral_preflight_resume_command(args: argparse.Namespace) -> str:
    values = [
        "python prepare_levir_spectral_indices.py preflight",
        f'--coords-json "{args.coords_json}"',
        f'--data-root "{args.data_root}"' if args.data_root else "",
        f'--output-root "{args.output_root}"',
        "--provider earth_engine_landsat",
        f"--timestamp-policy {args.timestamp_policy}",
        f"--sensor-policy {args.sensor_policy}",
        f"--ee-project {args.ee_project or 'YOUR_GOOGLE_CLOUD_PROJECT'}",
    ]
    return " ".join(value for value in values if value)


def run_preflight(args: argparse.Namespace) -> int:
    coordinate_path, records, _ = prepare_common(args)
    result: Dict[str, Any] = {
        "mapping_records": len(records),
        "coordinate_json": str(coordinate_path),
        "local_data_validated": bool(args.data_root),
        "provider": args.provider,
        "formulae": FORMULAE,
        "ndwi_variant": NDWI_VARIANT,
        "rgb_only_rejected": True,
    }
    try:
        provider = get_provider(args)
        result["provider_preflight"] = provider.preflight()
        result["ready"] = True
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ProviderUnavailable, AuthenticationRequired) as exc:
        result["ready"] = False
        result["blocker"] = str(exc)
        if args.provider == "earth_engine_landsat":
            result["required_user_actions"] = earth_engine_user_actions(args.ee_project)
            result["resume_command"] = spectral_preflight_resume_command(args)
        else:
            result["resume_command"] = (
                "Install requirements-auxiliary.txt and supply "
                "--precomputed-source-manifest"
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2


def run_generate(args: argparse.Namespace) -> int:
    if not args.data_root:
        raise LEVIRValidationError("--data-root is required before spectral generation")
    coordinate_path, records, output_root = prepare_common(args)
    manifest_path = output_root / MANIFEST_FILENAME
    manifest = skeleton_manifest(records, args, coordinate_path)
    if manifest_path.is_file() and not args.no_resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = merge_resume(manifest, previous)
    chosen = [item for item in manifest["samples"] if selected(item, args)]
    if args.limit is not None:
        chosen = chosen[: max(0, args.limit)]

    provider: Optional[SpectralProvider] = None
    provider_error: Optional[BaseException] = None
    # A mapping-only dry run is intentionally offline.  Scene inspection and
    # real generation are the only generate paths that initialize a provider.
    if not args.dry_run or args.inspect_scenes:
        try:
            provider = get_provider(args)
            provider.preflight()
        except (ProviderUnavailable, AuthenticationRequired) as exc:
            provider_error = exc

    work = []
    for sample in chosen:
        for temporal_key in ("t1", "t2"):
            record = sample[temporal_key]
            status = record.get("status")
            if status in ACCEPTABLE_STATUSES:
                error = cached_record_error(output_root, record, (args.height, args.width))
                if error is None:
                    continue
                record["status"] = "failed_retryable"
                record["error"] = "Cached output invalid: " + error
                status = "failed_retryable"
            if status in FAILED_STATUSES:
                if not retry_selected(
                    status,
                    retry_failed=args.retry_failed,
                    retry_transient=args.retry_transient,
                ):
                    continue
            work.append((sample, temporal_key))

    dry = {
        "selected_sources": len(chosen),
        "planned_date_records": len(work),
        "provider": args.provider,
        "published_month_windows": [
            {
                "filename": sample["filename"],
                "t1_month": sample["t1"]["image_month"],
                "t2_month": sample["t2"]["image_month"],
                "bbox": sample["bbox"],
            }
            for sample in chosen[:20]
        ],
        "provider_blocker": str(provider_error) if provider_error else None,
    }
    if args.dry_run:
        if args.inspect_scenes and isinstance(provider, EarthEngineLandsatProvider):
            inspections = []
            for sample, temporal_key in work:
                info = provider.inspect_window(sample, temporal_key, 0)
                inspections.append(
                    {
                        "filename": sample["filename"],
                        "date": temporal_key,
                        "window": [info["start"].isoformat(), info["end"].isoformat()],
                        "scene_count": info["scene_count"],
                        "valid_fraction": info["valid_fraction"],
                        "valid_without_slc_fraction": info.get(
                            "valid_without_slc_fraction", 0.0
                        ),
                        "slc_dropped_any_fraction": info.get(
                            "slc_dropped_any_fraction", 0.0
                        ),
                        "slc_dropped_output_loss_fraction": info.get(
                            "slc_dropped_output_loss_fraction", 0.0
                        ),
                        "slc_dropped_recovered_fraction": info.get(
                            "slc_dropped_recovered_fraction", 0.0
                        ),
                        "scenes": info["scene_metadata"],
                    }
                )
            dry["scene_inspection"] = inspections
        print(json.dumps(dry, indent=2, ensure_ascii=False))
        return 2 if provider_error else 0

    output_root.mkdir(parents=True, exist_ok=True)
    if provider_error:
        status = classify_exception(provider_error)
        for sample, temporal_key in work:
            sample[temporal_key] = make_blocked_record(
                sample[temporal_key], status, str(provider_error)
            )
        write_outputs(output_root, manifest)
        print(json.dumps(coverage_report(manifest), indent=2, ensure_ascii=False))
        print(f"BLOCKED: {provider_error}")
        return 2

    assert provider is not None
    options = provider_options(args)
    atomic_write_json(manifest_path, manifest)
    for sample, temporal_key in work:
        try:
            result = provider.generate(sample, temporal_key, output_root, options)
            record = copy.deepcopy(sample[temporal_key])
            record.update(result.record)
            record["status"] = result.status
            record["error"] = None
            record["generated_at"] = utc_now()
        except BaseException as exc:
            record = make_blocked_record(sample[temporal_key], classify_exception(exc), str(exc))
        sample[temporal_key] = record
        manifest["metadata"]["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        print(
            f"{sample['filename']} {temporal_key}: {record['status']} "
            f"(valid_fraction={record.get('valid_fraction')})"
        )
    write_outputs(output_root, manifest)
    print(json.dumps(coverage_report(manifest), indent=2, ensure_ascii=False))
    return 0


def load_spectral_manifest(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SPECTRAL_SCHEMA_VERSION:
        raise LEVIRValidationError(
            f"Unexpected spectral schema {document.get('schema_version')!r}"
        )
    return document


def validate_manifest(path: Path, require_complete: bool = False) -> Dict[str, Any]:
    document = load_spectral_manifest(path)
    samples = manifest_samples(document)
    if len(samples) != EXPECTED_TOTAL:
        raise LEVIRValidationError(f"Spectral manifest has {len(samples)} samples, expected 637")
    counts = Counter(item["split"] for item in samples)
    if dict(counts) != EXPECTED_SPLIT_COUNTS:
        raise LEVIRValidationError(f"Spectral split counts {dict(counts)} != {EXPECTED_SPLIT_COUNTS}")
    metadata = document.get("metadata", {})
    has_acceptable_output = any(
        sample.get(key, {}).get("status") in ACCEPTABLE_STATUSES
        for sample in samples
        for key in ("t1", "t2")
    )
    if metadata.get("ndwi_variant") != NDWI_VARIANT:
        raise LEVIRValidationError("Manifest NDWI is not the required McFeeters variant")
    if tuple(metadata.get("spectral_channel_order", ())) != SPECTRAL_CHANNEL_ORDER:
        raise LEVIRValidationError(
            "Manifest spectral_channel_order is not Green/Red/NIR/NDVI/McFeeters-NDWI"
        )
    if tuple(metadata.get("surface_reflectance_order", ())) != SURFACE_REFLECTANCE_ORDER:
        raise LEVIRValidationError("Manifest surface-reflectance band order is invalid")
    if tuple(metadata.get("surface_reflectance_valid_range", ())) != tuple(
        SURFACE_REFLECTANCE_VALID_RANGE
    ):
        raise LEVIRValidationError("Manifest surface-reflectance valid range is invalid")
    if metadata.get("formulae") != FORMULAE or metadata.get(
        "denominator_policy"
    ) != DENOMINATOR_POLICY:
        legacy_blocked_formulae = {
            "ndvi": "(nir - red) / (nir + red + epsilon)",
            "ndwi": "(green - nir) / (green + nir + epsilon)",
        }
        if has_acceptable_output or metadata.get("formulae") != legacy_blocked_formulae:
            raise LEVIRValidationError(
                "Manifest formula/denominator metadata differs from the implemented policy"
            )
    expected_shape = tuple(metadata.get("target_shape", [1024, 1024]))
    errors = []
    minimum_valid_fraction: Optional[float] = None
    try:
        minimum_valid_fraction = float(metadata["minimum_valid_fraction"])
        if not 0.0 <= minimum_valid_fraction <= 1.0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        if has_acceptable_output:
            errors.append("manifest has an invalid minimum_valid_fraction")

    manifest_provider = metadata.get("provider")
    manifest_project = metadata.get("earth_engine_project")
    if (
        has_acceptable_output
        and manifest_provider == "earth_engine_landsat"
        and (
            not isinstance(manifest_project, str)
            or not manifest_project.strip()
        )
    ):
        errors.append("Earth Engine manifest is missing earth_engine_project")

    legacy_project_cutoff: Optional[str] = None
    legacy_unlabelled_record_ids = set()
    migration_history = metadata.get("provenance_migration_history", [])
    if not isinstance(migration_history, list):
        errors.append("manifest provenance_migration_history is not a list")
        migration_history = []
    for migration in migration_history:
        if not isinstance(migration, Mapping):
            errors.append("manifest contains an invalid provenance migration record")
            continue
        migrated_at = migration.get("migrated_at")
        if (
            migration.get("migration")
            == "earth_engine_project_manifest_binding_v1"
            and migration.get("to") == manifest_project
            and migration.get("cached_date_records_modified") is False
            and migration.get("cached_rasters_rewritten") is False
            and isinstance(migrated_at, str)
            and migrated_at
            and (
                legacy_project_cutoff is None
                or migrated_at > legacy_project_cutoff
            )
        ):
            legacy_project_cutoff = migrated_at
            record_ids = migration.get("legacy_unlabelled_date_record_ids")
            if (
                not isinstance(record_ids, list)
                or not all(isinstance(value, str) for value in record_ids)
                or migration.get("legacy_acceptable_date_records")
                != len(record_ids)
            ):
                errors.append(
                    "Earth Engine project migration has invalid legacy record IDs"
                )
                continue
            legacy_unlabelled_record_ids.update(record_ids)

    allowed = ACCEPTABLE_STATUSES | FAILED_STATUSES | {"pending"}
    for sample in samples:
        for temporal_key in ("t1", "t2"):
            record = sample.get(temporal_key, {})
            status = record.get("status")
            if status not in allowed:
                errors.append(f"{sample['filename']} {temporal_key}: invalid status {status!r}")
                continue
            if require_complete and status in {
                "pending",
                "authentication_required",
                "failed_retryable",
            }:
                errors.append(
                    f"{sample['filename']} {temporal_key}: incomplete status {status}"
                )
            if status in ACCEPTABLE_STATUSES:
                error = cached_record_error(path.parent, record, expected_shape)
                if error:
                    errors.append(f"{sample['filename']} {temporal_key}: {error}")
                fraction = record.get("valid_fraction")
                try:
                    fraction_value = float(fraction)
                    if not 0.0 <= fraction_value <= 1.0:
                        raise ValueError
                except (TypeError, ValueError):
                    fraction_value = None
                    errors.append(f"{sample['filename']} {temporal_key}: invalid valid_fraction")
                if (
                    fraction_value is not None
                    and minimum_valid_fraction is not None
                    and fraction_value + 1e-12 < minimum_valid_fraction
                ):
                    errors.append(
                        f"{sample['filename']} {temporal_key}: valid_fraction "
                        f"{fraction_value} is below minimum_valid_fraction "
                        f"{minimum_valid_fraction}"
                    )
                invalid_fraction = record.get("invalid_fraction")
                try:
                    invalid_fraction_value = float(invalid_fraction)
                    if not 0.0 <= invalid_fraction_value <= 1.0:
                        raise ValueError
                except (TypeError, ValueError):
                    invalid_fraction_value = None
                if invalid_fraction_value is None or (
                    fraction_value is not None
                    and abs(fraction_value + invalid_fraction_value - 1.0) > 1e-6
                ):
                    errors.append(f"{sample['filename']} {temporal_key}: invalid invalid_fraction")
                if record.get("formulae") != FORMULAE:
                    errors.append(f"{sample['filename']} {temporal_key}: formula provenance mismatch")
                if record.get("denominator_policy") != DENOMINATOR_POLICY:
                    errors.append(
                        f"{sample['filename']} {temporal_key}: denominator provenance mismatch"
                    )
                if record.get("ndwi_variant") != NDWI_VARIANT:
                    errors.append(f"{sample['filename']} {temporal_key}: NDWI provenance mismatch")
                if tuple(record.get("spectral_channel_order", ())) != SPECTRAL_CHANNEL_ORDER:
                    errors.append(
                        f"{sample['filename']} {temporal_key}: spectral channel provenance mismatch"
                    )
                if tuple(record.get("surface_reflectance_valid_range", ())) != tuple(
                    SURFACE_REFLECTANCE_VALID_RANGE
                ):
                    errors.append(
                        f"{sample['filename']} {temporal_key}: surface-reflectance range provenance mismatch"
                    )
                for hash_key in ("spectral_sha256", "valid_mask_sha256"):
                    value = record.get(hash_key)
                    if not isinstance(value, str) or len(value) != 64:
                        errors.append(
                            f"{sample['filename']} {temporal_key}: missing {hash_key}"
                        )
                if record.get("provider") == "earth_engine_landsat":
                    record_project = record.get("earth_engine_project")
                    generated_at = record.get("generated_at")
                    legacy_unlabelled_record = (
                        record_project in (None, "")
                        and legacy_project_cutoff is not None
                        and isinstance(generated_at, str)
                        and generated_at <= legacy_project_cutoff
                        and f"{sample['filename']}:{temporal_key}"
                        in legacy_unlabelled_record_ids
                    )
                    if (
                        record_project != manifest_project
                        and not legacy_unlabelled_record
                    ):
                        errors.append(
                            f"{sample['filename']} {temporal_key}: "
                            "Earth Engine project provenance mismatch"
                        )
                    for metric in (
                        "slc_dropped_any_fraction",
                        "slc_dropped_output_loss_fraction",
                        "slc_dropped_recovered_fraction",
                        "aligned_slc_dropped_any_fraction",
                        "aligned_slc_dropped_output_loss_fraction",
                        "aligned_slc_dropped_recovered_fraction",
                    ):
                        value = record.get(metric)
                        if value is None or not 0.0 <= float(value) <= 1.0:
                            errors.append(
                                f"{sample['filename']} {temporal_key}: invalid {metric}"
                            )
                    if record.get("landsat7_slc_dropped_pixel_qa_radsat_bit") != 9:
                        errors.append(
                            f"{sample['filename']} {temporal_key}: missing Landsat-7 SLC QA provenance"
                        )
    if errors:
        raise LEVIRValidationError(
            f"Spectral validation failed with {len(errors)} error(s): " + "; ".join(errors[:20])
        )
    return coverage_report(document)


def run_validate(args: argparse.Namespace) -> int:
    report = validate_manifest(Path(args.manifest).expanduser().resolve(), args.require_complete)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def run_stats(args: argparse.Namespace) -> int:
    """Compute optional normalization only from acceptable retained training records."""

    try:
        import rasterio
    except ImportError as exc:
        raise ProviderUnavailable("rasterio is required to compute training statistics") from exc
    path = Path(args.manifest).expanduser().resolve()
    document = load_spectral_manifest(path)
    sums = np.zeros(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float64)
    square_sums = np.zeros(len(SPECTRAL_CHANNEL_ORDER), dtype=np.float64)
    counts = np.zeros(len(SPECTRAL_CHANNEL_ORDER), dtype=np.int64)
    source_ids = []
    for sample in manifest_samples(document):
        if sample["split"] != "train":
            continue
        if not all(sample[key].get("status") in ACCEPTABLE_STATUSES for key in ("t1", "t2")):
            continue
        source_ids.append(Path(sample["filename"]).stem)
        for temporal_key in ("t1", "t2"):
            record = sample[temporal_key]
            with rasterio.open(path.parent / record["spectral_path"]) as dataset:
                values = dataset.read().astype(np.float64)
            with rasterio.open(path.parent / record["valid_mask_path"]) as dataset:
                valid = dataset.read(1) > 0
            for band in range(len(SPECTRAL_CHANNEL_ORDER)):
                selected_values = values[band][valid]
                sums[band] += selected_values.sum(dtype=np.float64)
                square_sums[band] += np.square(selected_values).sum(dtype=np.float64)
                counts[band] += selected_values.size
    if (counts == 0).any():
        raise LEVIRValidationError("No acceptable training pixels available for statistics")
    means = sums / counts
    variance = np.maximum(square_sums / counts - np.square(means), 0.0)
    output = {
        "schema_version": "levir-spectral-train-normalization-v2",
        "source_manifest": path.name,
        "source_manifest_sha256": sha256_file(path),
        "split_used": "train",
        "validation_or_test_used": False,
        "retained_training_source_ids": source_ids,
        "channel_order": list(SPECTRAL_CHANNEL_ORDER),
        "index_order": list(INDEX_ORDER),
        "mean": means.tolist(),
        "std": np.sqrt(variance).tolist(),
        "valid_pixel_counts": counts.tolist(),
        "generated_at": utc_now(),
    }
    destination = Path(args.output).expanduser().resolve() if args.output else path.parent / "train_normalization.json"
    atomic_write_json(destination, output)
    print(json.dumps(output, indent=2))
    return 0


def colorize_index(values: np.ndarray, valid: np.ndarray, kind: str) -> np.ndarray:
    normalized = np.clip((values + 1.0) * 0.5, 0.0, 1.0)
    if kind == "ndvi":
        rgb = np.stack([1.0 - normalized, normalized, 0.25 * (1.0 - normalized)], axis=-1)
    else:
        rgb = np.stack([0.15 * (1.0 - normalized), 0.45 + 0.35 * normalized, normalized], axis=-1)
    rgb[~valid] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def colorize_signed_delta(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fixed-scale blue/white/red rendering for a T2-minus-T1 index delta."""

    normalized = np.clip(np.asarray(values, dtype=np.float32) / 2.0, -1.0, 1.0)
    magnitude = np.abs(normalized)
    red = np.where(normalized >= 0.0, 1.0, 1.0 - magnitude)
    blue = np.where(normalized <= 0.0, 1.0, 1.0 - magnitude)
    green = 1.0 - magnitude
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[~np.asarray(valid, dtype=bool)] = 0.0
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def run_qa(args: argparse.Namespace) -> int:
    try:
        import rasterio
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ProviderUnavailable("rasterio and Pillow are required for QA previews") from exc
    manifest_path = Path(args.manifest).expanduser().resolve()
    document = load_spectral_manifest(manifest_path)
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    chosen_groups = set()
    created = []
    for sample in manifest_samples(document):
        group = (sample["split"], int(sample["region_id"]))
        if group in chosen_groups:
            continue
        if not all(sample[key].get("status") in ACCEPTABLE_STATUSES for key in ("t1", "t2")):
            continue
        chosen_groups.add(group)
        stem = Path(sample["filename"]).stem
        date_arrays: Dict[str, np.ndarray] = {}
        date_valid: Dict[str, np.ndarray] = {}
        rgb_arrays: Dict[str, np.ndarray] = {}
        for temporal_key, rgb_dir in (("t1", "A"), ("t2", "B")):
            record = sample[temporal_key]
            with rasterio.open(manifest_path.parent / record["spectral_path"]) as dataset:
                date_arrays[temporal_key] = dataset.read().astype(np.float32)
            with rasterio.open(manifest_path.parent / record["valid_mask_path"]) as dataset:
                date_valid[temporal_key] = dataset.read(1) > 0
            rgb_path = data_root / sample["split"] / rgb_dir / sample["filename"]
            rgb_arrays[temporal_key] = np.asarray(Image.open(rgb_path).convert("RGB"))

        if date_arrays["t1"].shape != date_arrays["t2"].shape:
            raise LEVIRValidationError(f"QA index shapes differ for {sample['filename']}")
        if date_valid["t1"].shape != date_valid["t2"].shape:
            raise LEVIRValidationError(f"QA validity shapes differ for {sample['filename']}")
        paired_valid = date_valid["t1"] & date_valid["t2"]
        ndvi_delta = date_arrays["t2"][3] - date_arrays["t1"][3]
        ndwi_delta = date_arrays["t2"][4] - date_arrays["t1"][4]
        mask_t1 = np.repeat(
            (date_valid["t1"].astype(np.uint8) * 255)[..., None], 3, axis=2
        )
        mask_t2 = np.repeat(
            (date_valid["t2"].astype(np.uint8) * 255)[..., None], 3, axis=2
        )
        panel_specs = [
            ("RGB T1", rgb_arrays["t1"], Image.Resampling.BILINEAR),
            ("RGB T2", rgb_arrays["t2"], Image.Resampling.BILINEAR),
            ("NDVI T1", colorize_index(date_arrays["t1"][3], date_valid["t1"], "ndvi"), Image.Resampling.BILINEAR),
            ("NDVI T2", colorize_index(date_arrays["t2"][3], date_valid["t2"], "ndvi"), Image.Resampling.BILINEAR),
            ("NDVI delta T2-T1", colorize_signed_delta(ndvi_delta, paired_valid), Image.Resampling.BILINEAR),
            ("NDWI T1", colorize_index(date_arrays["t1"][4], date_valid["t1"], "ndwi"), Image.Resampling.BILINEAR),
            ("NDWI T2", colorize_index(date_arrays["t2"][4], date_valid["t2"], "ndwi"), Image.Resampling.BILINEAR),
            ("NDWI delta T2-T1", colorize_signed_delta(ndwi_delta, paired_valid), Image.Resampling.BILINEAR),
            ("validity T1", mask_t1, Image.Resampling.NEAREST),
            ("validity T2", mask_t2, Image.Resampling.NEAREST),
        ]
        size = min(384, rgb_arrays["t1"].shape[0], rgb_arrays["t1"].shape[1])
        header_height = 52
        caption_height = 22
        canvas = Image.new(
            "RGB", (size * 5, header_height + (size + caption_height) * 2), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 6),
            f"{stem} region {sample['region_id']} | paired temporal spectral QA",
            fill="black",
        )
        draw.text(
            (8, 27),
            f"T1 valid={sample['t1'].get('valid_fraction')} | T2 valid={sample['t2'].get('valid_fraction')} | delta=T2-T1",
            fill="black",
        )
        for index, (title, array, resampling) in enumerate(panel_specs):
            row, column = divmod(index, 5)
            x = column * size
            y = header_height + row * (size + caption_height)
            draw.text((x + 4, y + 3), title, fill="black")
            panel = Image.fromarray(array).resize((size, size), resampling)
            canvas.paste(panel, (x, y + caption_height))
        destination = (
            output_root
            / sample["split"]
            / f"region_{sample['region_id']:02d}_{stem}_paired.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        metadata_path = destination.with_suffix(".json")
        atomic_write_json(
            metadata_path,
            {
                "sample": sample,
                "panel_order": [title for title, _, _ in panel_specs],
                "signed_delta_definition": "T2 - T1",
                "signed_delta_validity": "T1 validity AND T2 validity",
                "delta_display_range": [-2.0, 2.0],
                "paired_valid_fraction": float(paired_valid.mean()),
                "ndvi_delta_valid_min": float(ndvi_delta[paired_valid].min())
                if paired_valid.any()
                else None,
                "ndvi_delta_valid_max": float(ndvi_delta[paired_valid].max())
                if paired_valid.any()
                else None,
                "ndwi_delta_valid_min": float(ndwi_delta[paired_valid].min())
                if paired_valid.any()
                else None,
                "ndwi_delta_valid_max": float(ndwi_delta[paired_valid].max())
                if paired_valid.any()
                else None,
            },
        )
        created.append(destination.as_posix())
    print(json.dumps({"created": len(created), "previews": created}, indent=2))
    return 0


def add_mapping_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coords-json", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-root", default="data_spectral")
    parser.add_argument("--provider", choices=("precomputed", "earth_engine_landsat"), default="earth_engine_landsat")
    parser.add_argument("--precomputed-source-manifest", default="")
    parser.add_argument("--ee-project", default="")
    parser.add_argument("--ee-high-volume", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--min-valid-fraction", type=float, default=0.80)
    parser.add_argument("--expand-step-days", type=int, default=16)
    parser.add_argument("--max-window-days", type=int, default=64)
    parser.add_argument("--composite", choices=("median",), default="median")
    parser.add_argument(
        "--timestamp-policy",
        choices=("month_start", "month_end"),
        default="month_end",
        help="Shared LEVIR year/month timestamp policy recorded with OSM and spectral data",
    )
    parser.add_argument(
        "--sensor-policy",
        choices=("landsat7_consistent",),
        default="landsat7_consistent",
        help="Use Landsat 7 ETM+ for both dates; no silent cross-sensor mixing",
    )
    parser.add_argument("--skip-image-validation", action="store_true")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate data and provider/authentication")
    add_mapping_options(preflight)
    preflight.set_defaults(func=run_preflight)

    generate = subparsers.add_parser("generate", help="Generate or resume spectral rasters")
    add_mapping_options(generate)
    generate.add_argument("--split", action="append", choices=("train", "val", "test"))
    generate.add_argument("--region", action="append", type=int, choices=range(1, 21))
    generate.add_argument("--sample", action="append")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--inspect-scenes", action="store_true")
    retry_group = generate.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry every failed status, including scientifically terminal exclusions",
    )
    retry_group.add_argument(
        "--retry-transient",
        action="store_true",
        help=(
            "Retry only authentication_required and failed_retryable records; "
            "preserve terminal insufficient/source/permanent exclusions"
        ),
    )
    generate.add_argument("--no-resume", action="store_true")
    generate.set_defaults(func=run_generate)

    validate = subparsers.add_parser("validate", help="Validate manifest, ranges, masks, and cached rasters")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--require-complete", action="store_true")
    validate.set_defaults(func=run_validate)

    stats = subparsers.add_parser("stats", help="Compute optional normalization from training data only")
    stats.add_argument("--manifest", required=True)
    stats.add_argument("--output", default="")
    stats.set_defaults(func=run_stats)

    qa = subparsers.add_parser("qa", help="Create deterministic alignment QA previews")
    qa.add_argument("--manifest", required=True)
    qa.add_argument("--data-root", required=True)
    qa.add_argument("--output-root", default="data_spectral/qa_previews")
    qa.set_defaults(func=run_qa)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except (LEVIRValidationError, ProviderUnavailable, AuthenticationRequired, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
