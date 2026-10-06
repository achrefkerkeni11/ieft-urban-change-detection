#!/usr/bin/env python
"""Prepare reproducible historical OSM context for LEVIR-CD.

This command is the only code path that contacts the ohsome API.  Training,
validation, testing, and inference consume only its local manifest and cached
GeoJSON files.  Generation can request both dates or one explicit date; dates
that were deliberately omitted are recorded as ``not_requested``.
"""

from __future__ import annotations

import argparse
import copy
import email.utils
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from IEFT.levir_metadata import (
    COORDINATE_SOURCE_URL,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL,
    LEVIRValidationError,
    atomic_write_json,
    build_source_records,
    is_before_history,
    load_coordinate_json,
    manifest_samples,
    natural_sample_sort_key,
    normalize_sample_key,
    path_relative_to,
    sha256_file,
    summarize_statuses,
    validate_local_levir,
)
from IEFT.osm_geometry import (
    OSM_SPATIAL_CHANNEL_ORDER,
    bboxes_intersect,
    feature_tags,
    geometry_bbox,
    is_axis_aligned_bbox_polygon,
    summarize_feature_collection,
)


MANIFEST_SCHEMA_VERSION = "levir-temporal-osm-v3"
PREVIOUS_MANIFEST_SCHEMA_VERSION = "levir-temporal-osm-v2"
LEGACY_MANIFEST_SCHEMA_VERSION = "levir-temporal-osm-v1"
GENERATOR_VERSION = "3.0.0"
OHSOME_BASE_URL = "https://api.ohsome.org/v1"
OHSOME_METADATA_URL = f"{OHSOME_BASE_URL}/metadata"
OHSOME_GEOMETRY_URL = f"{OHSOME_BASE_URL}/elements/geometry"
OHSOME_BBOX_URL = f"{OHSOME_BASE_URL}/elements/bbox"
OHSOME_FILTER = (
    "(building=* or highway=* or railway=* or landuse=* or natural=* or "
    "waterway=* or amenity=* or leisure=*)"
)
OSM_ATTRIBUTION = "© OpenStreetMap contributors"
SUCCESS_STATUSES = {"ok", "ok_empty"}
NOT_REQUESTED_STATUS = "not_requested"
TEMPORAL_KEYS = ("t1", "t2")
TERMINAL_STATUSES = SUCCESS_STATUSES | {
    "unavailable_before_history",
    NOT_REQUESTED_STATUS,
}
FAILED_STATUSES = {"failed_retryable", "failed_permanent"}
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
ENDPOINT_UNAVAILABLE_HTTP = {403, 404, 405}
EXTRACTION_GEOMETRIES = {"geometry", "bbox"}
GEOMETRY_SEMANTICS = {
    "geometry": "exact_osm_feature_geometries",
    "bbox": "feature_bounding_boxes_not_exact_geometries",
}
# Exact official documentation example used by the retained live capability
# evidence.  The narrow amenity/type filter keeps this probe small even though
# the documented Heidelberg bbox is larger than a LEVIR source footprint.
PROBE_BBOX = [8.625, 49.3711, 8.7334, 49.4397]
PROBE_TIMESTAMP = "2019-09-01T00:00:00Z"
PROBE_FILTER = "amenity=bicycle_rental and type:node"
EXPECTED_TAG_KEYS = {
    "building",
    "highway",
    "railway",
    "landuse",
    "natural",
    "waterway",
    "amenity",
    "leisure",
}
OSM_ID_PATTERN = re.compile(r"^(node|way|relation)/([1-9][0-9]*)$")


def selected_temporal_keys(value: Any) -> Tuple[str, ...]:
    """Normalize the CLI/manifest temporal selection to a stable tuple."""

    if isinstance(value, str):
        raw = value.strip().lower()
        if raw == "both":
            return TEMPORAL_KEYS
        values = (raw,)
    elif isinstance(value, Sequence):
        values = tuple(str(item).strip().lower() for item in value)
    else:
        raise ValueError(f"Invalid temporal-key selection {value!r}")
    if not values or any(item not in TEMPORAL_KEYS for item in values):
        raise ValueError(
            "temporal-key selection must contain only 't1' and/or 't2'"
        )
    return tuple(item for item in TEMPORAL_KEYS if item in set(values))


def manifest_temporal_keys(document: Mapping[str, Any]) -> Tuple[str, ...]:
    """Read v3 selection provenance; v1/v2 manifests always requested both."""

    metadata = document.get("metadata", {})
    if isinstance(metadata, Mapping) and "requested_temporal_keys" in metadata:
        return selected_temporal_keys(metadata["requested_temporal_keys"])
    return TEMPORAL_KEYS


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_retry_after(value: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            return max(0.0, (parsed.astimezone(timezone.utc) - current).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def extraction_endpoint(base_url: str, extraction_geometry: str) -> str:
    mode = str(extraction_geometry).lower()
    if mode not in EXTRACTION_GEOMETRIES:
        raise ValueError(f"Unsupported extraction geometry {extraction_geometry!r}")
    return f"{base_url.rstrip('/')}/elements/{mode}"


def validate_extraction_document(
    document: Mapping[str, Any],
    extraction_geometry: str,
    *,
    expected_timestamp: str = "",
    query_bbox: Optional[Sequence[float]] = None,
    require_nonempty: bool = False,
) -> Dict[str, Any]:
    """Validate an element-level ohsome extraction without changing it.

    Both extraction modes must preserve one feature per OSM element, its OSM
    identity, requested metadata, and at least one matching tag.  In bbox mode
    we additionally prove that each geometry is the documented axis-aligned
    feature bounding box rather than treating it as exact OSM geometry.
    """

    mode = str(extraction_geometry).lower()
    if mode not in EXTRACTION_GEOMETRIES:
        raise ValueError(f"Unsupported extraction geometry {extraction_geometry!r}")
    if not isinstance(document, Mapping) or document.get("type") != "FeatureCollection":
        raise ValueError("ohsome response is not a GeoJSON FeatureCollection")
    features = document.get("features")
    if not isinstance(features, list):
        raise ValueError("ohsome FeatureCollection lacks a features list")
    if require_nonempty and not features:
        raise ValueError("capability probe returned no feature; properties cannot be validated")
    attribution = document.get("attribution")
    attribution_text = attribution.get("text") if isinstance(attribution, Mapping) else ""
    if "OpenStreetMap contributors" not in str(attribution_text):
        raise ValueError("ohsome extraction lacks OpenStreetMap contributor attribution")
    if not str(document.get("apiVersion", "")).strip():
        raise ValueError("ohsome extraction lacks apiVersion provenance")

    osm_ids = set()
    geometry_types: Counter[str] = Counter()
    for index, feature in enumerate(features):
        prefix = f"feature {index}"
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise ValueError(f"{prefix} is not a GeoJSON Feature")
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(f"{prefix} lacks properties")
        osm_id = str(properties.get("@osmId", ""))
        match = OSM_ID_PATTERN.fullmatch(osm_id)
        if match is None:
            raise ValueError(f"{prefix} lacks a valid @osmId")
        if osm_id in osm_ids:
            raise ValueError(f"duplicate OSM element identity {osm_id!r}")
        osm_ids.add(osm_id)
        if str(properties.get("@osmType", "")) != match.group(1):
            raise ValueError(f"{prefix} @osmType does not match @osmId")
        for name in ("@version", "@changesetId", "@lastEdit", "@snapshotTimestamp"):
            if name not in properties or properties[name] in (None, ""):
                raise ValueError(f"{prefix} lacks requested metadata property {name}")
        if expected_timestamp and str(properties["@snapshotTimestamp"]) != expected_timestamp:
            raise ValueError(
                f"{prefix} snapshot timestamp {properties['@snapshotTimestamp']!r} "
                f"does not match query {expected_timestamp!r}"
            )
        tags = feature_tags(feature)
        if not tags or not EXPECTED_TAG_KEYS.intersection(tags):
            raise ValueError(f"{prefix} lacks a tag selected by the extraction filter")

        geometry = feature.get("geometry")
        extent = geometry_bbox(geometry if isinstance(geometry, Mapping) else None)
        if extent is None:
            raise ValueError(f"{prefix} lacks a finite GeoJSON geometry")
        if not all(float("-inf") < float(value) < float("inf") for value in extent):
            raise ValueError(f"{prefix} has non-finite geometry coordinates")
        geometry_type = str(geometry.get("type", "")) if isinstance(geometry, Mapping) else ""
        geometry_types[geometry_type] += 1
        if mode == "bbox" and not is_axis_aligned_bbox_polygon(geometry):
            raise ValueError(f"{prefix} is not an axis-aligned bbox Polygon")
        if query_bbox is not None and not bboxes_intersect(extent, query_bbox):
            raise ValueError(f"{prefix} extent does not intersect the requested bbox")

    return {
        "feature_count": len(features),
        "unique_osm_id_count": len(osm_ids),
        "geometry_types": dict(sorted(geometry_types.items())),
        "attribution": str(attribution_text),
        "api_version": str(document.get("apiVersion")),
    }


class PoliteRateLimiter:
    def __init__(self, minimum_interval: float, clock: Callable[[], float] = time.monotonic,
                 sleeper: Callable[[float], None] = time.sleep):
        self.minimum_interval = max(0.0, float(minimum_interval))
        self.clock = clock
        self.sleeper = sleeper
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        with self.lock:
            now = self.clock()
            delay = max(0.0, self.next_allowed - now)
            if delay:
                self.sleeper(delay)
                now = self.clock()
            self.next_allowed = now + self.minimum_interval


class OhsomeClient:
    def __init__(
        self,
        base_url: str = OHSOME_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 5,
        backoff_initial: float = 2.0,
        backoff_max: float = 90.0,
        rate_limit_seconds: float = 1.0,
        user_agent: str = "IEFT-LEVIR-temporal-context/1.0 (research; cached offline workflow)",
        session: Optional[requests.Session] = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.backoff_initial = max(0.0, float(backoff_initial))
        self.backoff_max = max(self.backoff_initial, float(backoff_max))
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.sleeper = sleeper
        self.random_fn = random_fn
        self.rate_limiter = PoliteRateLimiter(rate_limit_seconds, sleeper=sleeper)

    def _request(
        self,
        method: str,
        url: str,
        *,
        status_collector: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> requests.Response:
        last_exception: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if status_collector is not None:
                    status_collector.append(int(response.status_code))
                if response.status_code not in RETRYABLE_HTTP:
                    return response
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                if attempt >= self.max_retries:
                    return response
                delay = retry_after if retry_after is not None else min(
                    self.backoff_max,
                    self.backoff_initial * (2 ** attempt) * (0.75 + 0.5 * self.random_fn()),
                )
                self.sleeper(delay)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exception = exc
                if attempt >= self.max_retries:
                    raise
                delay = min(
                    self.backoff_max,
                    self.backoff_initial * (2 ** attempt) * (0.75 + 0.5 * self.random_fn()),
                )
                self.sleeper(delay)
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("unreachable retry state")

    def metadata(self) -> Dict[str, Any]:
        response = self._request("GET", f"{self.base_url}/metadata")
        response.raise_for_status()
        document = response.json()
        temporal = document.get("extractRegion", {}).get("temporalExtent", {})
        if not temporal.get("fromTimestamp"):
            raise ValueError("ohsome metadata lacks extractRegion.temporalExtent.fromTimestamp")
        return document

    def extract(
        self,
        bbox: Sequence[float],
        timestamp: str,
        extraction_geometry: str,
        *,
        osm_filter: str = OHSOME_FILTER,
        status_collector: Optional[List[int]] = None,
        require_nonempty: bool = False,
        http_method: str = "POST",
    ) -> Tuple[Dict[str, Any], int]:
        mode = str(extraction_geometry).lower()
        method = str(http_method).strip().upper()
        if method not in {"GET", "POST"}:
            raise ValueError(f"Unsupported ohsome extraction HTTP method {http_method!r}")
        payload = {
            "bboxes": ",".join(f"{float(value):.12g}" for value in bbox),
            "time": timestamp,
            "filter": osm_filter,
            "properties": "tags,metadata",
            "clipGeometry": "true",
        }
        request_kwargs = {"params": payload} if method == "GET" else {"data": payload}
        response = self._request(
            method,
            extraction_endpoint(self.base_url, mode),
            status_collector=status_collector,
            **request_kwargs,
        )
        status_code = int(response.status_code)
        if status_code >= 400:
            error = requests.HTTPError(
                f"ohsome {method} HTTP {status_code}: {response.text[:500]}",
                response=response,
            )
            setattr(error, "ohsome_http_method", method)
            raise error
        document = response.json()
        validate_extraction_document(
            document,
            mode,
            expected_timestamp=timestamp,
            query_bbox=bbox,
            require_nonempty=require_nonempty,
        )
        return document, status_code

    def geometry(self, bbox: Sequence[float], timestamp: str) -> Tuple[Dict[str, Any], int]:
        """Backward-compatible exact-geometry extraction wrapper."""

        return self.extract(bbox, timestamp, "geometry")


def detect_extraction_capability(
    client: OhsomeClient, requested_extraction_geometry: str
) -> Dict[str, Any]:
    """Select extraction geometry and HTTP method using a tiny valid probe.

    For exact geometry we deliberately try both documented transport forms:
    POST first (the historical implementation) and GET second.  ``auto`` falls
    back to bbox only when *both* exact-geometry methods are endpoint-unavailable
    (403/404/405).  Validation/schema errors are never hidden by bbox fallback.
    """

    requested = str(requested_extraction_geometry).lower()
    if requested not in {"auto", *EXTRACTION_GEOMETRIES}:
        raise ValueError(f"Unsupported requested extraction geometry {requested!r}")
    statuses: Dict[str, Dict[str, List[int]]] = {}
    validations: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, Dict[str, str]] = {}

    def probe(mode: str, methods: Sequence[str]) -> str:
        statuses.setdefault(mode, {})
        errors.setdefault(mode, {})
        last_endpoint_error: Optional[requests.HTTPError] = None
        for method in methods:
            method = str(method).upper()
            method_statuses: List[int] = []
            statuses[mode][method] = method_statuses
            try:
                document, _ = client.extract(
                    PROBE_BBOX,
                    PROBE_TIMESTAMP,
                    mode,
                    osm_filter=PROBE_FILTER,
                    status_collector=method_statuses,
                    require_nonempty=True,
                    http_method=method,
                )
                validations[mode] = validate_extraction_document(
                    document,
                    mode,
                    expected_timestamp=PROBE_TIMESTAMP,
                    query_bbox=PROBE_BBOX,
                    require_nonempty=True,
                )
                validations[mode]["http_method"] = method
                return method
            except requests.HTTPError as exc:
                code = int(exc.response.status_code) if exc.response is not None else 0
                errors[mode][method] = str(exc)
                if code not in ENDPOINT_UNAVAILABLE_HTTP:
                    raise
                last_endpoint_error = exc
        if last_endpoint_error is not None:
            raise last_endpoint_error
        raise RuntimeError(f"No ohsome HTTP method was attempted for {mode}")

    fallback_used = False
    fallback_reason = ""
    if requested == "bbox":
        effective_method = probe("bbox", ("POST", "GET"))
        effective = "bbox"
    else:
        try:
            effective_method = probe("geometry", ("POST", "GET"))
            effective = "geometry"
        except requests.HTTPError as exc:
            code = int(exc.response.status_code) if exc.response is not None else 0
            if requested != "auto" or code not in ENDPOINT_UNAVAILABLE_HTTP:
                raise
            fallback_used = True
            geometry_codes = [
                code_value
                for method_codes in statuses.get("geometry", {}).values()
                for code_value in method_codes
            ]
            fallback_reason = (
                "geometry_endpoint_unavailable_after_post_and_get_"
                + "_".join(str(value) for value in geometry_codes)
            )
            effective_method = probe("bbox", ("POST", "GET"))
            effective = "bbox"

    return {
        "requested_extraction_geometry": requested,
        "effective_extraction_geometry": effective,
        "effective_endpoint": extraction_endpoint(client.base_url, effective),
        "effective_http_method": effective_method,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "capability_probe_time": utc_now(),
        "capability_probe_http_statuses": statuses,
        "capability_probe_errors": errors,
        "capability_probe_validation": validations,
        "capability_probe_bbox": list(PROBE_BBOX),
        "capability_probe_timestamp": PROBE_TIMESTAMP,
        "capability_probe_filter": PROBE_FILTER,
        "geometry_semantics": GEOMETRY_SEMANTICS[effective],
    }


def _manifest_effective_geometry(document: Mapping[str, Any]) -> str:
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return ""
    explicit = str(metadata.get("effective_extraction_geometry", "")).lower()
    if explicit in EXTRACTION_GEOMETRIES:
        return explicit
    endpoint = str(metadata.get("effective_endpoint", metadata.get("ohsome_endpoint", "")))
    for mode in EXTRACTION_GEOMETRIES:
        if endpoint.rstrip("/").endswith(f"/elements/{mode}"):
            return mode
    return ""


def _manifest_has_known_geometry_403(document: Mapping[str, Any]) -> bool:
    if _manifest_effective_geometry(document) != "geometry":
        return False
    for sample in manifest_samples(document):
        for temporal_key in TEMPORAL_KEYS:
            if _is_known_geometry_endpoint_403(sample.get(temporal_key, {}), "geometry"):
                return True
    return False


def capability_for_generate(
    args: argparse.Namespace,
    client: OhsomeClient,
    previous: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Reuse a run's selected mode unless safe 403 recovery needs probing."""

    requested = str(args.extraction_geometry).lower()
    if not previous:
        return detect_extraction_capability(client, requested)
    previous_mode = _manifest_effective_geometry(previous)
    if not previous_mode:
        # A legacy manifest with no recognizable extraction endpoint cannot be
        # migrated safely because its cached geometry semantics are unknown.
        raise LEVIRValidationError(
            "Existing OSM manifest does not identify its extraction endpoint; "
            "use a new --output-root instead of guessing cache semantics"
        )
    previous_metadata = previous.get("metadata", {})
    previous_endpoint = str(
        previous_metadata.get("effective_endpoint", previous_metadata.get("ohsome_endpoint", ""))
    )
    expected_previous_endpoint = extraction_endpoint(args.ohsome_base_url, previous_mode)
    if previous_endpoint.rstrip("/") != expected_previous_endpoint.rstrip("/"):
        raise LEVIRValidationError(
            f"Existing manifest endpoint {previous_endpoint!r} does not match configured "
            f"ohsome service {expected_previous_endpoint!r}"
        )
    previous_schema = str(previous.get("schema_version", ""))
    needs_schema_migration = previous_schema == LEGACY_MANIFEST_SCHEMA_VERSION
    recovery_probe = previous_mode == "geometry" and _manifest_has_known_geometry_403(previous)

    if not needs_schema_migration and not recovery_probe:
        if requested not in {"auto", previous_mode}:
            raise LEVIRValidationError(
                f"Existing resumable run selected {previous_mode!r}, but "
                f"--extraction-geometry requested {requested!r}; use a new output root"
            )
        required = {
            "effective_endpoint",
            "fallback_used",
            "fallback_reason",
            "capability_probe_time",
            "capability_probe_http_statuses",
            "geometry_semantics",
        }
        missing = sorted(name for name in required if name not in previous_metadata)
        if missing:
            raise LEVIRValidationError(
                "Existing v2 manifest lacks capability provenance: " + ", ".join(missing)
            )
        result = {name: copy.deepcopy(previous_metadata.get(name)) for name in required}
        result.update(
            {
                "requested_extraction_geometry": str(
                    previous_metadata.get("requested_extraction_geometry", requested)
                ),
                "effective_extraction_geometry": previous_mode,
                "effective_http_method": str(
                    previous_metadata.get("effective_http_method", "POST")
                ).upper(),
                "capability_probe_validation": copy.deepcopy(
                    previous_metadata.get("capability_probe_validation", {})
                ),
                "capability_probe_bbox": copy.deepcopy(
                    previous_metadata.get("capability_probe_bbox", PROBE_BBOX)
                ),
                "capability_probe_timestamp": str(
                    previous_metadata.get("capability_probe_timestamp", PROBE_TIMESTAMP)
                ),
                "capability_probe_filter": str(
                    previous_metadata.get("capability_probe_filter", PROBE_FILTER)
                ),
                "capability_selection_source": "resumed_manifest",
            }
        )
        return result

    capability = detect_extraction_capability(client, requested)
    capability["capability_selection_source"] = (
        "known_geometry_403_recovery_probe" if recovery_probe else "legacy_manifest_migration_probe"
    )
    return capability


def _metadata_fields(document: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    extent = document.get("extractRegion", {}).get("temporalExtent", {})
    start = str(extent.get("fromTimestamp", ""))
    end = str(extent.get("toTimestamp", ""))
    if not start:
        raise LEVIRValidationError("ohsome metadata lacks temporal history start")
    attribution = document.get("attribution", {})
    attribution_text = str(attribution.get("text", OSM_ATTRIBUTION)) if isinstance(attribution, Mapping) else OSM_ATTRIBUTION
    return start, end, str(document.get("apiVersion", "unknown")), attribution_text


def skeleton_manifest(
    source_records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], timestamp_policy: str,
    coordinate_path: Path,
    capability: Optional[Mapping[str, Any]] = None,
    temporal_keys: Sequence[str] = TEMPORAL_KEYS,
) -> Dict[str, Any]:
    history_start, history_end, api_version, attribution = _metadata_fields(metadata)
    requested_keys = selected_temporal_keys(temporal_keys)
    selected = dict(
        capability
        or {
            "requested_extraction_geometry": "geometry",
            "effective_extraction_geometry": "geometry",
            "effective_endpoint": OHSOME_GEOMETRY_URL,
            "effective_http_method": "POST",
            "fallback_used": False,
            "fallback_reason": "",
            "capability_probe_time": "",
            "capability_probe_http_statuses": {},
            "capability_probe_validation": {},
            "capability_probe_bbox": list(PROBE_BBOX),
            "capability_probe_timestamp": PROBE_TIMESTAMP,
            "capability_probe_filter": PROBE_FILTER,
            "geometry_semantics": GEOMETRY_SEMANTICS["geometry"],
            "capability_selection_source": "test_or_legacy_default",
        }
    )
    effective_geometry = str(selected["effective_extraction_geometry"])
    samples: List[Dict[str, Any]] = []
    for source in source_records:
        item = copy.deepcopy(dict(source))
        for temporal_key in TEMPORAL_KEYS:
            date_record = item[temporal_key]
            requested = temporal_key in requested_keys
            before = requested and is_before_history(
                date_record["query_timestamp"], history_start
            )
            date_record.update(
                {
                    "status": (
                        NOT_REQUESTED_STATUS
                        if not requested
                        else "unavailable_before_history" if before else "pending"
                    ),
                    "feature_count": None,
                    "raw_geojson": "",
                    "error": None,
                    "generated_at": None,
                    "osm_struct": [0.0] * 16,
                    "osm_text": "",
                    "extraction_geometry": effective_geometry,
                    "effective_endpoint": str(selected["effective_endpoint"]),
                    "geometry_semantics": str(selected["geometry_semantics"]),
                }
            )
        samples.append(item)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metadata": {
            "coordinate_source_url": COORDINATE_SOURCE_URL,
            "coordinate_json": coordinate_path.name,
            "coordinate_sha256": sha256_file(coordinate_path),
            "ohsome_metadata_url": OHSOME_METADATA_URL,
            "ohsome_endpoint": str(selected["effective_endpoint"]),
            "effective_http_method": str(selected.get("effective_http_method", "POST")).upper(),
            "ohsome_api_version": api_version,
            "ohsome_history_start": history_start,
            "ohsome_history_end_at_generation": history_end,
            "filter": OHSOME_FILTER,
            "properties": "tags,metadata",
            "clip_geometry": True,
            "timestamp_policy": timestamp_policy,
            "requested_temporal_keys": list(requested_keys),
            "attribution": attribution or OSM_ATTRIBUTION,
            "generator_version": GENERATOR_VERSION,
            "migration_history": [],
            "generated_at": utc_now(),
            **selected,
        },
        "samples": samples,
    }


def _is_known_geometry_endpoint_403(
    date_record: Mapping[str, Any], previous_mode: str
) -> bool:
    if previous_mode != "geometry" or date_record.get("status") != "failed_permanent":
        return False
    try:
        stored_status = int(date_record.get("http_status", 0) or 0)
    except (TypeError, ValueError):
        stored_status = 0
    error = str(date_record.get("error", ""))
    if stored_status not in {0, 403}:
        return False
    if stored_status == 0 and not re.search(r"(?:HTTP|fixture)\s+403\b", error, re.IGNORECASE):
        return False
    endpoint = str(date_record.get("effective_endpoint", ""))
    if endpoint and not endpoint.rstrip("/").endswith("/elements/geometry"):
        return False
    lowered = error.lower()
    return "403" in lowered and (
        "forbidden" in lowered
        or "permission to access this resource" in lowered
        or "apache" in lowered
    )


def _merge_resume(
    skeleton: Dict[str, Any],
    previous: Optional[Mapping[str, Any]],
    *,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if not previous:
        return skeleton
    previous_schema = str(previous.get("schema_version", ""))
    supported_schemas = {
        MANIFEST_SCHEMA_VERSION,
        PREVIOUS_MANIFEST_SCHEMA_VERSION,
        LEGACY_MANIFEST_SCHEMA_VERSION,
    }
    if previous_schema not in supported_schemas:
        raise LEVIRValidationError(
            f"Cannot resume schema {previous_schema!r}; expected "
            f"one of {sorted(supported_schemas)!r}"
        )
    previous_mode = _manifest_effective_geometry(previous)
    effective_mode = _manifest_effective_geometry(skeleton)
    if not previous_mode or not effective_mode:
        raise LEVIRValidationError("Cannot establish extraction geometry while resuming manifest")
    mode_changed = previous_mode != effective_mode
    if mode_changed and (previous_mode, effective_mode) != ("geometry", "bbox"):
        raise LEVIRValidationError(
            f"Unsafe extraction-mode migration {previous_mode!r} -> {effective_mode!r}; "
            "use a new output root"
        )

    old = {item["filename"]: item for item in manifest_samples(previous)}
    reset_records: List[Dict[str, str]] = []
    upgraded_success_records: List[Dict[str, str]] = []
    invalid_success_records: List[Dict[str, str]] = []
    preserved_permanent = 0
    for current in skeleton["samples"]:
        prior = old.get(current["filename"])
        if not prior:
            continue
        for key in ("split", "region_id", "bbox"):
            if prior.get(key) != current.get(key):
                raise LEVIRValidationError(f"Resume metadata mismatch for {current['filename']} field {key}")
        for temporal_key in TEMPORAL_KEYS:
            old_date = prior.get(temporal_key, {})
            if old_date.get("image_month") != current[temporal_key]["image_month"]:
                raise LEVIRValidationError(f"Resume month mismatch for {current['filename']} {temporal_key}")
            if old_date.get("query_timestamp") != current[temporal_key]["query_timestamp"]:
                raise LEVIRValidationError(f"Resume timestamp mismatch for {current['filename']} {temporal_key}")
            old_status = old_date.get("status")
            current_status = current[temporal_key].get("status")
            # The current CLI selection is authoritative.  Never resurrect a
            # cached date that this run deliberately omitted, and allow a
            # later wider run to turn an old not_requested record into pending.
            if current_status == NOT_REQUESTED_STATUS:
                continue
            if old_status == NOT_REQUESTED_STATUS:
                continue
            if old_status in SUCCESS_STATUSES and mode_changed:
                if output_root is None:
                    raise LEVIRValidationError(
                        "Cannot change extraction mode while prior successful records exist "
                        "without validating their caches"
                    )
                cache_error = validate_cached_record(
                    output_root,
                    old_date,
                    extraction_geometry=previous_mode,
                    sample_bbox=prior.get("bbox"),
                )
                detail = "validated" if cache_error is None else f"invalid: {cache_error}"
                raise LEVIRValidationError(
                    f"Cannot mix {previous_mode} and {effective_mode} caches in one run: "
                    f"{current['filename']} {temporal_key} is marked successful ({detail}). "
                    "Use a new output root."
                )
            if (
                old_status in SUCCESS_STATUSES
                and previous_schema == LEGACY_MANIFEST_SCHEMA_VERSION
                and not mode_changed
            ):
                if output_root is None:
                    raise LEVIRValidationError(
                        "A legacy successful OSM record must be cache-validated during migration"
                    )
                cache_error = validate_cached_record(
                    output_root,
                    old_date,
                    extraction_geometry=previous_mode,
                    sample_bbox=prior.get("bbox"),
                )
                marker = {"filename": str(current["filename"]), "temporal_key": temporal_key}
                if cache_error is not None:
                    current[temporal_key].update(
                        {
                            "status": "failed_retryable",
                            "error": f"legacy successful cache failed validation: {cache_error}",
                            "migration_reason": "invalid_legacy_success_cache",
                        }
                    )
                    invalid_success_records.append(marker)
                    continue
                raw_path = output_root / Path(str(old_date["raw_geojson"]))
                cached_document = json.loads(raw_path.read_text(encoding="utf-8"))
                validation = validate_extraction_document(
                    cached_document,
                    previous_mode,
                    expected_timestamp=str(old_date.get("query_timestamp", "")),
                    query_bbox=prior.get("bbox"),
                )
                upgraded = copy.deepcopy(old_date)
                upgraded.update(
                    {
                        "raw_sha256": sha256_file(raw_path),
                        "extraction_geometry": effective_mode,
                        "effective_endpoint": skeleton["metadata"]["effective_endpoint"],
                        "geometry_semantics": skeleton["metadata"]["geometry_semantics"],
                        "response_api_version": validation["api_version"],
                        "response_attribution": validation["attribution"],
                        "unique_osm_id_count": validation["unique_osm_id_count"],
                        "geometry_types": validation["geometry_types"],
                        "migration_reason": "validated_legacy_success_cache",
                    }
                )
                current[temporal_key] = upgraded
                upgraded_success_records.append(marker)
                continue
            if (
                mode_changed
                and _is_known_geometry_endpoint_403(old_date, previous_mode)
            ):
                current[temporal_key].update(
                    {
                        "status": "pending",
                        "migrated_from_status": "failed_permanent",
                        "migrated_from_endpoint": str(
                            previous.get("metadata", {}).get(
                                "effective_endpoint",
                                previous.get("metadata", {}).get(
                                    "ohsome_endpoint", OHSOME_GEOMETRY_URL
                                ),
                            )
                        ),
                        "migration_reason": "known_geometry_endpoint_http_403",
                    }
                )
                reset_records.append(
                    {"filename": str(current["filename"]), "temporal_key": temporal_key}
                )
                continue
            if old_status in TERMINAL_STATUSES | FAILED_STATUSES:
                current[temporal_key] = copy.deepcopy(old_date)
                if old_status == "failed_permanent":
                    preserved_permanent += 1
                if old_status == "unavailable_before_history":
                    # No extraction occurred, so the record follows the new
                    # run semantics without claiming a legacy response mode.
                    current[temporal_key]["extraction_geometry"] = effective_mode
                    current[temporal_key]["effective_endpoint"] = skeleton["metadata"][
                        "effective_endpoint"
                    ]
                    current[temporal_key]["geometry_semantics"] = skeleton["metadata"][
                        "geometry_semantics"
                    ]
                else:
                    current[temporal_key].setdefault("extraction_geometry", previous_mode)
                    current[temporal_key].setdefault(
                        "effective_endpoint",
                        str(previous.get("metadata", {}).get("ohsome_endpoint", "")),
                    )
                    current[temporal_key].setdefault(
                        "geometry_semantics", GEOMETRY_SEMANTICS[previous_mode]
                    )

    previous_metadata = previous.get("metadata", {})
    previous_temporal_keys = manifest_temporal_keys(previous)
    current_temporal_keys = manifest_temporal_keys(skeleton)
    selection_changed = previous_temporal_keys != current_temporal_keys
    history = copy.deepcopy(previous_metadata.get("migration_history", []))
    if not isinstance(history, list):
        raise LEVIRValidationError("Existing migration_history is not a list")
    if previous_schema != MANIFEST_SCHEMA_VERSION or mode_changed or selection_changed:
        if mode_changed:
            migration_reason = "known_geometry_endpoint_403_bbox_fallback"
        elif previous_schema != MANIFEST_SCHEMA_VERSION:
            migration_reason = "manifest_schema_upgrade"
        else:
            migration_reason = "temporal_key_selection_changed"
        history.append(
            {
                "migrated_at": utc_now(),
                "generator_version": GENERATOR_VERSION,
                "from_schema_version": previous_schema,
                "to_schema_version": MANIFEST_SCHEMA_VERSION,
                "from_effective_extraction_geometry": previous_mode,
                "to_effective_extraction_geometry": effective_mode,
                "reason": migration_reason,
                "from_requested_temporal_keys": list(previous_temporal_keys),
                "to_requested_temporal_keys": list(current_temporal_keys),
                "reset_records": reset_records,
                "reset_record_count": len(reset_records),
                "validated_and_upgraded_success_records": upgraded_success_records,
                "validated_and_upgraded_success_record_count": len(upgraded_success_records),
                "invalid_success_records": invalid_success_records,
                "invalid_success_record_count": len(invalid_success_records),
                "preserved_permanent_error_count": preserved_permanent,
            }
        )
    skeleton["metadata"]["migration_history"] = history
    skeleton["metadata"]["initial_generated_at"] = str(
        previous_metadata.get("initial_generated_at", previous_metadata.get("generated_at", ""))
    )
    return skeleton


def _selected(sample: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if args.split and sample.get("split") not in args.split:
        return False
    if args.region and int(sample.get("region_id", -1)) not in args.region:
        return False
    if args.sample:
        wanted = {normalize_sample_key(value, keep_extension=True) for value in args.sample}
        if sample.get("filename") not in wanted:
            return False
    return True


def _status_for_exception(exc: BaseException) -> Tuple[str, str]:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = int(exc.response.status_code)
        status = "failed_retryable" if code in RETRYABLE_HTTP else "failed_permanent"
        return status, str(exc)
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return "failed_retryable", str(exc)
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "failed_permanent", str(exc)
    return "failed_retryable", f"{type(exc).__name__}: {exc}"


def fetch_one(
    client: OhsomeClient,
    output_root: Path,
    sample: Mapping[str, Any],
    temporal_key: str,
    extraction_geometry: str = "geometry",
    effective_endpoint: str = "",
    effective_http_method: str = "POST",
) -> Tuple[str, str, Dict[str, Any]]:
    filename = str(sample["filename"])
    stem = Path(filename).stem
    raw_path = output_root / "raw" / str(sample["split"]) / stem / f"{temporal_key}.geojson"
    date_record = copy.deepcopy(dict(sample[temporal_key]))
    mode = str(extraction_geometry).lower()
    endpoint = effective_endpoint or extraction_endpoint(client.base_url, mode)
    try:
        method = str(effective_http_method or "POST").upper()
        document, http_status = client.extract(
            sample["bbox"],
            date_record["query_timestamp"],
            mode,
            http_method=method,
        )
        extraction_validation = validate_extraction_document(
            document,
            mode,
            expected_timestamp=date_record["query_timestamp"],
            query_bbox=sample["bbox"],
        )
        summary = summarize_feature_collection(document)
        atomic_write_json(raw_path, document)
        feature_count = len(document["features"])
        date_record.update(
            {
                "status": "ok_empty" if feature_count == 0 else "ok",
                "feature_count": feature_count,
                "raw_geojson": path_relative_to(raw_path, output_root),
                "raw_sha256": sha256_file(raw_path),
                "error": None,
                "http_status": http_status,
                "generated_at": utc_now(),
                "extraction_geometry": mode,
                "effective_endpoint": endpoint,
                "effective_http_method": method,
                "geometry_semantics": GEOMETRY_SEMANTICS[mode],
                "response_api_version": extraction_validation["api_version"],
                "response_attribution": extraction_validation["attribution"],
                "unique_osm_id_count": extraction_validation["unique_osm_id_count"],
                "geometry_types": extraction_validation["geometry_types"],
                "osm_spatial_channel_order": list(OSM_SPATIAL_CHANNEL_ORDER),
                # This measures confidence in the timestamped query/cache
                # geometry, not physical feature completeness.  In particular,
                # ok_empty remains available context and is never a hard
                # negative building label.
                "osm_reliability": 0.75 if mode == "geometry" else 0.50,
                "osm_reliability_semantics": (
                    "query_and_geometry_confidence_not_physical_completeness"
                ),
                **summary,
            }
        )
    except BaseException as exc:
        status, error = _status_for_exception(exc)
        date_record.update(
            {
                "status": status,
                "feature_count": None,
                "raw_geojson": "",
                "raw_sha256": "",
                "error": error,
                "http_status": (
                    int(exc.response.status_code)
                    if isinstance(exc, requests.HTTPError) and exc.response is not None
                    else None
                ),
                "generated_at": utc_now(),
                "extraction_geometry": mode,
                "effective_endpoint": endpoint,
                "effective_http_method": str(effective_http_method or "POST").upper(),
                "geometry_semantics": GEOMETRY_SEMANTICS[mode],
            }
        )
    return filename, temporal_key, date_record


def validate_cached_record(
    output_root: Path,
    date_record: Mapping[str, Any],
    *,
    extraction_geometry: str = "",
    sample_bbox: Optional[Sequence[float]] = None,
) -> Optional[str]:
    status = date_record.get("status")
    if status not in SUCCESS_STATUSES:
        return None
    raw_relative = str(date_record.get("raw_geojson", ""))
    if not raw_relative:
        return "successful record lacks raw_geojson"
    raw_path = output_root / Path(raw_relative)
    try:
        document = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot read cached GeoJSON {raw_path}: {exc}"
    mode = str(extraction_geometry or date_record.get("extraction_geometry", "geometry")).lower()
    try:
        extraction_validation = validate_extraction_document(
            document,
            mode,
            expected_timestamp=str(date_record.get("query_timestamp", "")),
            query_bbox=sample_bbox,
        )
    except (TypeError, ValueError) as exc:
        return f"cached extraction validation failed for {raw_path}: {exc}"
    if len(document["features"]) != int(date_record.get("feature_count", -1)):
        return f"cached feature count mismatch: {raw_path}"
    if status == "ok_empty" and document["features"]:
        return f"ok_empty record has features: {raw_path}"
    if status == "ok" and not document["features"]:
        return f"ok record has no features: {raw_path}"
    stored_sha256 = str(date_record.get("raw_sha256", ""))
    if stored_sha256 and sha256_file(raw_path) != stored_sha256:
        return f"cached SHA-256 mismatch: {raw_path}"
    stored_identity_count = date_record.get("unique_osm_id_count")
    if stored_identity_count is not None and int(stored_identity_count) != int(
        extraction_validation["unique_osm_id_count"]
    ):
        return f"cached unique OSM identity count mismatch: {raw_path}"
    stored_attribution = str(date_record.get("response_attribution", ""))
    if stored_attribution and stored_attribution != extraction_validation["attribution"]:
        return f"cached attribution mismatch: {raw_path}"
    return None


def build_legacy_views(samples: Iterable[Mapping[str, Any]], temporal_key: str) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for sample in samples:
        record = sample[temporal_key]
        output[Path(sample["filename"]).stem] = {
            "status": record.get("status"),
            "osm_struct": record.get("osm_struct", [0.0] * 16),
            "osm_text": record.get("osm_text", ""),
            "feature_count": record.get("feature_count"),
            "query_timestamp": record.get("query_timestamp"),
            "image_month": record.get("image_month"),
            "raw_geojson": record.get("raw_geojson", ""),
            "raw_sha256": record.get("raw_sha256", ""),
            "extraction_geometry": record.get("extraction_geometry", ""),
            "effective_endpoint": record.get("effective_endpoint", ""),
            "geometry_semantics": record.get("geometry_semantics", ""),
        }
    return output


def coverage_report(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    samples = manifest_samples(manifest)
    report = summarize_statuses(samples)
    requested_keys = manifest_temporal_keys(manifest)
    paired = Counter()
    requested_complete = Counter()
    for sample in samples:
        both = sample["t1"].get("status") in SUCCESS_STATUSES and sample["t2"].get("status") in SUCCESS_STATUSES
        paired[str(sample["split"])] += int(both)
        requested_complete[str(sample["split"])] += int(
            all(sample[key].get("status") in SUCCESS_STATUSES for key in requested_keys)
        )
    report.update(
        {
            "schema_version": str(manifest.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            "generated_at": utc_now(),
            "total_samples": len(samples),
            "requested_temporal_keys": list(requested_keys),
            "requested_dates_successful_by_split": dict(sorted(requested_complete.items())),
            "paired_successful_by_split": dict(sorted(paired.items())),
            "requested_extraction_geometry": manifest.get("metadata", {}).get(
                "requested_extraction_geometry", "legacy_unspecified"
            ),
            "effective_extraction_geometry": _manifest_effective_geometry(manifest),
            "effective_endpoint": manifest.get("metadata", {}).get(
                "effective_endpoint", manifest.get("metadata", {}).get("ohsome_endpoint", "")
            ),
            "effective_http_method": manifest.get("metadata", {}).get(
                "effective_http_method", "POST"
            ),
            "fallback_used": manifest.get("metadata", {}).get("fallback_used", False),
            "geometry_semantics": manifest.get("metadata", {}).get(
                "geometry_semantics", GEOMETRY_SEMANTICS.get(_manifest_effective_geometry(manifest), "")
            ),
            "attribution": manifest.get("metadata", {}).get("attribution", OSM_ATTRIBUTION),
        }
    )
    return report


def validate_manifest_file(
    manifest_path: Path, require_complete: bool = False, check_cached: bool = True
) -> Dict[str, Any]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = str(document.get("schema_version", ""))
    supported_schemas = {
        MANIFEST_SCHEMA_VERSION,
        PREVIOUS_MANIFEST_SCHEMA_VERSION,
        LEGACY_MANIFEST_SCHEMA_VERSION,
    }
    if schema_version not in supported_schemas:
        raise LEVIRValidationError(
            f"Unexpected manifest schema {document.get('schema_version')!r}"
        )
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise LEVIRValidationError("OSM manifest metadata must be an object")
    effective_mode = _manifest_effective_geometry(document)
    if not effective_mode:
        raise LEVIRValidationError("OSM manifest lacks a recognizable extraction endpoint")
    if schema_version in {MANIFEST_SCHEMA_VERSION, PREVIOUS_MANIFEST_SCHEMA_VERSION}:
        required_metadata = {
            "requested_extraction_geometry",
            "effective_extraction_geometry",
            "effective_endpoint",
            "fallback_used",
            "fallback_reason",
            "capability_probe_time",
            "capability_probe_http_statuses",
            "geometry_semantics",
            "generator_version",
            "migration_history",
        }
        if schema_version == MANIFEST_SCHEMA_VERSION:
            required_metadata.add("requested_temporal_keys")
        missing_metadata = sorted(name for name in required_metadata if name not in metadata)
        if missing_metadata:
            raise LEVIRValidationError(
                "OSM manifest lacks capability provenance: " + ", ".join(missing_metadata)
            )
        if metadata.get("geometry_semantics") != GEOMETRY_SEMANTICS[effective_mode]:
            raise LEVIRValidationError("OSM manifest geometry_semantics does not match endpoint")
        if str(metadata.get("effective_endpoint", "")) != extraction_endpoint(
            str(metadata.get("effective_endpoint", "")).split("/elements/")[0], effective_mode
        ):
            raise LEVIRValidationError("OSM manifest effective_endpoint does not match selected mode")
        if not isinstance(metadata.get("capability_probe_http_statuses"), Mapping):
            raise LEVIRValidationError("capability_probe_http_statuses must be an object")
        if not isinstance(metadata.get("migration_history"), list):
            raise LEVIRValidationError("migration_history must be a list")
    if "OpenStreetMap contributors" not in str(metadata.get("attribution", "")):
        raise LEVIRValidationError("OSM manifest lacks required OpenStreetMap attribution")
    requested_keys = manifest_temporal_keys(document)
    samples = manifest_samples(document)
    if len(samples) != EXPECTED_TOTAL:
        raise LEVIRValidationError(f"Manifest contains {len(samples)} samples, expected {EXPECTED_TOTAL}")
    split_counts = Counter(item["split"] for item in samples)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise LEVIRValidationError(f"Manifest split counts {dict(split_counts)} != {EXPECTED_SPLIT_COUNTS}")
    allowed = TERMINAL_STATUSES | FAILED_STATUSES | {"pending"}
    errors = []
    for sample in samples:
        for temporal_key in TEMPORAL_KEYS:
            record = sample.get(temporal_key, {})
            status = record.get("status")
            if status not in allowed:
                errors.append(f"{sample['filename']} {temporal_key}: invalid status {status!r}")
            if require_complete and status == "pending":
                errors.append(f"{sample['filename']} {temporal_key}: still pending")
            requested = temporal_key in requested_keys
            if requested and status == NOT_REQUESTED_STATUS:
                errors.append(
                    f"{sample['filename']} {temporal_key}: requested date is marked not_requested"
                )
            if not requested and status != NOT_REQUESTED_STATUS:
                errors.append(
                    f"{sample['filename']} {temporal_key}: unrequested date has status {status!r}"
                )
            if status == NOT_REQUESTED_STATUS:
                if str(record.get("raw_geojson", "")) or str(record.get("raw_sha256", "")):
                    errors.append(
                        f"{sample['filename']} {temporal_key}: not_requested date references a cache"
                    )
                if record.get("feature_count") is not None:
                    errors.append(
                        f"{sample['filename']} {temporal_key}: not_requested date has a feature count"
                    )
            if schema_version in {
                MANIFEST_SCHEMA_VERSION,
                PREVIOUS_MANIFEST_SCHEMA_VERSION,
            } and status in SUCCESS_STATUSES:
                if record.get("extraction_geometry") != effective_mode:
                    errors.append(
                        f"{sample['filename']} {temporal_key}: extraction mode differs from manifest"
                    )
                if record.get("effective_endpoint") != metadata.get("effective_endpoint"):
                    errors.append(
                        f"{sample['filename']} {temporal_key}: endpoint differs from manifest"
                    )
                if record.get("geometry_semantics") != metadata.get("geometry_semantics"):
                    errors.append(
                        f"{sample['filename']} {temporal_key}: geometry semantics differ from manifest"
                    )
                if not str(record.get("raw_sha256", "")):
                    errors.append(f"{sample['filename']} {temporal_key}: missing raw SHA-256")
            if check_cached:
                cache_error = validate_cached_record(
                    manifest_path.parent,
                    record,
                    extraction_geometry=effective_mode,
                    sample_bbox=sample.get("bbox"),
                )
                if cache_error:
                    errors.append(f"{sample['filename']} {temporal_key}: {cache_error}")
            if status == "unavailable_before_history":
                history_start = document.get("metadata", {}).get("ohsome_history_start", "")
                if not history_start or not is_before_history(record.get("query_timestamp", ""), history_start):
                    errors.append(f"{sample['filename']} {temporal_key}: invalid history-boundary status")
    if errors:
        raise LEVIRValidationError(
            f"OSM manifest validation failed with {len(errors)} error(s): " + "; ".join(errors[:20])
        )
    return coverage_report(document)


def get_coordinates(args: argparse.Namespace, output_root: Path) -> Path:
    if args.coords_json:
        path = Path(args.coords_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Coordinate JSON not found: {path}")
        return path
    destination = output_root / "LEVIR_CD_name_coords.json"
    if destination.is_file():
        return destination
    response = requests.get(
        COORDINATE_SOURCE_URL,
        timeout=float(args.timeout),
        headers={"User-Agent": args.user_agent},
    )
    response.raise_for_status()
    document = response.json()
    atomic_write_json(destination, document)
    return destination


def obtain_metadata(args: argparse.Namespace, client: OhsomeClient) -> Dict[str, Any]:
    if args.history_start:
        return {
            "apiVersion": "user-supplied-history-boundary",
            "attribution": {"text": OSM_ATTRIBUTION, "url": "https://ohsome.org/copyrights"},
            "extractRegion": {
                "temporalExtent": {"fromTimestamp": args.history_start, "toTimestamp": "unknown"}
            },
        }
    return client.metadata()


def run_preflight(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    coordinate_path = get_coordinates(args, output_root)
    coordinates = load_coordinate_json(coordinate_path)
    records = build_source_records(coordinates, args.timestamp_policy)
    if args.data_root:
        validate_local_levir(args.data_root, coordinates, inspect_images=not args.skip_image_validation)
    client = make_client(args)
    metadata = obtain_metadata(args, client)
    capability = detect_extraction_capability(client, args.extraction_geometry)
    history_start, history_end, api_version, attribution = _metadata_fields(metadata)
    requested_keys = selected_temporal_keys(args.temporal_key)
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        for temporal_key in TEMPORAL_KEYS:
            if temporal_key not in requested_keys:
                state = NOT_REQUESTED_STATUS
            else:
                state = (
                    "unavailable_before_history"
                    if is_before_history(
                        item[temporal_key]["query_timestamp"], history_start
                    )
                    else "queryable"
                )
            counts[item["split"]][f"{temporal_key}:{state}"] += 1
    result = {
        "valid_coordinate_records": len(coordinates),
        "valid_local_data": bool(args.data_root),
        "coordinate_json": str(coordinate_path),
        "ohsome_api_version": api_version,
        "ohsome_history_start": history_start,
        "ohsome_history_end": history_end,
        "attribution": attribution,
        "requested_temporal_keys": list(requested_keys),
        "availability_by_split": {key: dict(value) for key, value in counts.items()},
        **capability,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def make_client(args: argparse.Namespace) -> OhsomeClient:
    return OhsomeClient(
        base_url=args.ohsome_base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        backoff_initial=args.backoff_initial,
        backoff_max=args.backoff_max,
        rate_limit_seconds=args.rate_limit_seconds,
        user_agent=args.user_agent,
    )


def plan_requests(
    manifest: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    retry_failed: bool,
) -> List[Tuple[Mapping[str, Any], str]]:
    """Plan resumable requests without redownloading validated successes."""

    planned: List[Tuple[Mapping[str, Any], str]] = []
    effective_mode = str(manifest["metadata"]["effective_extraction_geometry"])
    requested_keys = manifest_temporal_keys(manifest)
    for sample in selected:
        for temporal_key in requested_keys:
            record = sample[temporal_key]
            status = record.get("status")
            if status == NOT_REQUESTED_STATUS:
                raise LEVIRValidationError(
                    f"{sample['filename']} {temporal_key} is requested by manifest metadata "
                    "but marked not_requested"
                )
            if status == "unavailable_before_history":
                continue
            if status in SUCCESS_STATUSES:
                cache_error = validate_cached_record(
                    output_root,
                    record,
                    extraction_geometry=effective_mode,
                    sample_bbox=sample.get("bbox"),
                )
                if cache_error is None:
                    continue
                record["status"] = "failed_retryable"
                record["error"] = cache_error
                status = "failed_retryable"
            if status == "failed_permanent":
                continue
            if status == "failed_retryable" and not retry_failed:
                continue
            planned.append((sample, temporal_key))
    return planned


def run_generate(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    coordinate_path = get_coordinates(args, output_root)
    coordinates = load_coordinate_json(coordinate_path)
    if not args.data_root:
        raise LEVIRValidationError("--data-root is required before downloading historical OSM")
    validate_local_levir(args.data_root, coordinates, inspect_images=not args.skip_image_validation)
    records = build_source_records(coordinates, args.timestamp_policy)
    requested_keys = selected_temporal_keys(args.temporal_key)
    client = make_client(args)
    metadata = obtain_metadata(args, client)
    manifest_argument = str(args.manifest_name).strip()
    manifest_name = Path(manifest_argument).name
    if manifest_name != manifest_argument:
        raise LEVIRValidationError("--manifest-name must not contain path components")
    if not manifest_name.lower().endswith(".json"):
        raise LEVIRValidationError("--manifest-name must name a JSON file")
    manifest_path = output_root / manifest_name
    previous = None
    if manifest_path.is_file() and not args.no_resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    capability = capability_for_generate(args, client, previous)
    manifest = skeleton_manifest(
        records,
        metadata,
        args.timestamp_policy,
        coordinate_path,
        capability,
        temporal_keys=requested_keys,
    )
    manifest = _merge_resume(manifest, previous, output_root=output_root)
    selected = [item for item in manifest["samples"] if _selected(item, args)]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    planned = plan_requests(
        manifest, selected, output_root, retry_failed=bool(args.retry_failed)
    )

    dry_report = {
        "selected_sources": len(selected),
        "requested_temporal_keys": list(requested_keys),
        "planned_requests": len(planned),
        "skipped_or_pre_history_dates": len(selected) * len(requested_keys) - len(planned),
        "not_requested_dates": len(selected) * (len(TEMPORAL_KEYS) - len(requested_keys)),
        "history_start": manifest["metadata"]["ohsome_history_start"],
        "requested_extraction_geometry": manifest["metadata"]["requested_extraction_geometry"],
        "effective_extraction_geometry": manifest["metadata"]["effective_extraction_geometry"],
        "effective_endpoint": manifest["metadata"]["effective_endpoint"],
        "effective_http_method": manifest["metadata"].get("effective_http_method", "POST"),
        "fallback_used": manifest["metadata"]["fallback_used"],
        "geometry_semantics": manifest["metadata"]["geometry_semantics"],
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2))
        return 0

    print(
        json.dumps(
            {"extraction_selection": dry_report}, indent=2, ensure_ascii=False
        )
    )

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    index = {item["filename"]: item for item in manifest["samples"]}
    if planned:
        with ThreadPoolExecutor(max_workers=max(1, min(2, args.workers))) as executor:
            futures = {
                executor.submit(
                    fetch_one,
                    client,
                    output_root,
                    sample,
                    temporal_key,
                    manifest["metadata"]["effective_extraction_geometry"],
                    manifest["metadata"]["effective_endpoint"],
                    manifest["metadata"].get("effective_http_method", "POST"),
                ):
                (sample["filename"], temporal_key)
                for sample, temporal_key in planned
            }
            for future in as_completed(futures):
                filename, temporal_key, record = future.result()
                index[filename][temporal_key] = record
                manifest["metadata"]["updated_at"] = utc_now()
                atomic_write_json(manifest_path, manifest)
                print(f"{filename} {temporal_key}: {record['status']} ({record.get('feature_count')})")

    atomic_write_json(output_root / "levir_osm_t1.json", build_legacy_views(manifest["samples"], "t1"))
    atomic_write_json(output_root / "levir_osm_t2.json", build_legacy_views(manifest["samples"], "t2"))
    report = coverage_report(manifest)
    atomic_write_json(output_root / "generation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def run_validate(args: argparse.Namespace) -> int:
    report = validate_manifest_file(
        Path(args.manifest).expanduser().resolve(),
        require_complete=args.require_complete,
        check_cached=not args.skip_cached_validation,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coords-json", default="", help="Official LEVIR coordinate JSON; downloaded if omitted")
    parser.add_argument("--data-root", default="", help="LEVIR root containing train/val/test")
    parser.add_argument("--output-root", default="data_osm_t2_v2")
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Manifest filename within --output-root (path components are rejected)",
    )
    parser.add_argument("--timestamp-policy", choices=("month_start", "month_end"), default="month_end")
    parser.add_argument(
        "--temporal-key",
        choices=("both", "t1", "t2"),
        default="t2",
        help=(
            "Historical LEVIR date(s) to request from ohsome. Dates omitted by "
            "this selection are recorded as not_requested and never downloaded."
        ),
    )
    parser.add_argument("--ohsome-base-url", default=OHSOME_BASE_URL)
    parser.add_argument(
        "--extraction-geometry",
        choices=("auto", "geometry", "bbox"),
        default="auto",
        help=(
            "Extraction mode selected once per resumable run. 'auto' probes exact geometry "
            "and falls back to element bounding boxes only for endpoint-level 403/404/405."
        ),
    )
    parser.add_argument("--history-start", default="", help="Offline/test override; normally read from /metadata")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-initial", type=float, default=2.0)
    parser.add_argument("--backoff-max", type=float, default=90.0)
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument(
        "--user-agent",
        default="IEFT-LEVIR-temporal-context/1.0 (research; cached offline workflow)",
    )
    parser.add_argument("--skip-image-validation", action="store_true", help="Skip slow image header checks only")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Validate mapping/data and report history availability")
    add_common_options(preflight)
    preflight.set_defaults(func=run_preflight)

    generate = subparsers.add_parser("generate", help="Generate/resume cached historical OSM")
    add_common_options(generate)
    generate.add_argument("--split", action="append", choices=("train", "val", "test"))
    generate.add_argument("--region", action="append", type=int, choices=range(1, 21))
    generate.add_argument("--sample", action="append")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--workers", type=int, default=1, choices=(1, 2))
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--retry-failed", action="store_true")
    generate.add_argument("--no-resume", action="store_true")
    generate.set_defaults(func=run_generate)

    validate = subparsers.add_parser("validate", help="Validate manifest schema and cached GeoJSON")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--require-complete", action="store_true")
    validate.add_argument("--skip-cached-validation", action="store_true")
    validate.set_defaults(func=run_validate)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except (LEVIRValidationError, FileNotFoundError, requests.RequestException, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
