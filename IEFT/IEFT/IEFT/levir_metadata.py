"""Authoritative LEVIR-CD spatial and temporal metadata helpers.

This module is intentionally dependency-light.  Both offline auxiliary-data
preparation commands and the runtime dataset import it so filename/date/bbox
mapping cannot drift between the OSM and spectral workflows.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "levir-metadata-v1"
COORDINATE_SOURCE_URL = (
    "https://raw.githubusercontent.com/justchenhao/STANet/master/"
    "LEVIR_CD_spatial_info/LEVIR_CD_name_coords.json"
)
LEVIR_WEBSITE_URL = "https://justchenhao.github.io/LEVIR/"
LEVIR_PAPER_URL = "https://www.mdpi.com/2072-4292/12/10/1662"
EXPECTED_SPLIT_COUNTS = {"train": 445, "val": 64, "test": 128}
EXPECTED_TOTAL = 637

# Windows can briefly deny an otherwise valid atomic replacement while an
# antivirus/indexer or another reader releases its handle.  Keep the retry
# small and bounded: the completed temporary file remains on the same volume,
# and a persistent error is still surfaced to the caller.
_ATOMIC_REPLACE_ATTEMPTS = 6
_ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05
_ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.5


# Published LEVIR-CD regional month table.  Region 7 is deliberately retained
# exactly as published (T1=2017-11, T2=2018-06).
REGION_MONTHS: Dict[int, Tuple[str, str]] = {
    1: ("2002-12", "2013-11"),
    2: ("2002-12", "2013-11"),
    3: ("2002-12", "2013-11"),
    4: ("2002-02", "2017-11"),
    5: ("2003-03", "2017-01"),
    6: ("2003-03", "2017-01"),
    7: ("2017-11", "2018-06"),
    8: ("2008-02", "2018-01"),
    9: ("2006-04", "2017-01"),
    10: ("2003-03", "2017-01"),
    11: ("2003-03", "2015-07"),
    12: ("2009-03", "2017-02"),
    13: ("2003-03", "2017-01"),
    14: ("2003-03", "2012-08"),
    15: ("2003-03", "2013-11"),
    16: ("2006-04", "2016-02"),
    17: ("2009-02", "2017-01"),
    18: ("2009-02", "2017-01"),
    19: ("2011-03", "2017-01"),
    20: ("2012-08", "2017-01"),
}


# Inclusive indices within each official split.  None means the region has no
# member in that split.  These ranges map metadata only; runtime eligibility is
# always driven by generated manifest statuses.
REGION_SPLIT_RANGES: Dict[int, Dict[str, Optional[Tuple[int, int]]]] = {
    1: {"train": (1, 17), "val": (1, 3), "test": (1, 6)},
    2: {"train": (18, 33), "val": (4, 5), "test": (7, 9)},
    3: {"train": (34, 47), "val": (6, 7), "test": (10, 14)},
    4: {"train": (48, 60), "val": (8, 9), "test": (15, 18)},
    5: {"train": (61, 82), "val": (10, 13), "test": (19, 23)},
    6: {"train": (83, 111), "val": (14, 17), "test": (24, 31)},
    7: {"train": (112, 128), "val": (18, 20), "test": (32, 34)},
    8: {"train": (129, 158), "val": (21, 21), "test": (35, 47)},
    9: {"train": (159, 221), "val": (22, 33), "test": (48, 66)},
    10: {"train": (222, 249), "val": (34, 36), "test": (67, 74)},
    11: {"train": (250, 256), "val": None, "test": (75, 75)},
    12: {"train": (257, 277), "val": (37, 38), "test": (76, 82)},
    13: {"train": (278, 297), "val": (39, 43), "test": (83, 87)},
    14: {"train": (298, 310), "val": (44, 44), "test": (88, 93)},
    15: {"train": (311, 321), "val": (45, 46), "test": (94, 100)},
    16: {"train": (322, 337), "val": (47, 48), "test": (101, 106)},
    17: {"train": (338, 358), "val": (49, 54), "test": (107, 109)},
    18: {"train": (359, 402), "val": (55, 55), "test": (110, 120)},
    19: {"train": (403, 430), "val": (56, 62), "test": (121, 127)},
    20: {"train": (431, 445), "val": (63, 64), "test": (128, 128)},
}

EXPECTED_REGION_TOTALS = {
    1: 26, 2: 21, 3: 21, 4: 19, 5: 31, 6: 41, 7: 23, 8: 44,
    9: 94, 10: 39, 11: 8, 12: 30, 13: 30, 14: 20, 15: 20,
    16: 24, 17: 30, 18: 56, 19: 42, 20: 18,
}

_SOURCE_RE = re.compile(r"^(train|val|test)_(\d+)(?:\.[A-Za-z0-9]+)?$", re.IGNORECASE)
_TILE_RE = re.compile(
    r"^(train|val|test)_(\d+)(?:\.[A-Za-z0-9]+)?(?:_y\d+_x\d+|_r\d+_c\d+)$",
    re.IGNORECASE,
)


class LEVIRValidationError(ValueError):
    """Raised when official or local LEVIR metadata fail validation."""


def natural_sample_sort_key(value: str) -> Tuple[int, int, str]:
    key = normalize_sample_key(value, keep_extension=False)
    match = _SOURCE_RE.match(key)
    if not match:
        return (99, 0, str(value))
    split, index = match.groups()
    return ({"train": 0, "val": 1, "test": 2}[split.lower()], int(index), key)


def normalize_sample_key(value: Any, keep_extension: bool = False) -> str:
    """Normalize a source filename, stem, or tile key to its source-pair key."""

    raw = Path(str(value).replace("\\", "/")).name.strip()
    raw = re.sub(r"(?:_y\d+_x\d+|_r\d+_c\d+)$", "", raw, flags=re.IGNORECASE)
    suffix = Path(raw).suffix
    if suffix:
        raw = raw[: -len(suffix)]
    match = _SOURCE_RE.match(raw)
    if not match:
        raise LEVIRValidationError(f"Unrecognized LEVIR sample key: {value!r}")
    split, index = match.groups()
    normalized = f"{split.lower()}_{int(index)}"
    return normalized + ".png" if keep_extension else normalized


def split_and_index(value: Any) -> Tuple[str, int]:
    key = normalize_sample_key(value, keep_extension=False)
    match = _SOURCE_RE.match(key)
    assert match is not None
    return match.group(1).lower(), int(match.group(2))


def build_filename_region_map() -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for region_id, ranges in REGION_SPLIT_RANGES.items():
        for split in ("train", "val", "test"):
            bounds = ranges[split]
            if bounds is None:
                continue
            start, end = bounds
            for index in range(start, end + 1):
                filename = f"{split}_{index}.png"
                if filename in mapping:
                    raise LEVIRValidationError(f"Duplicate region assignment: {filename}")
                mapping[filename] = region_id
    validate_region_mapping(mapping)
    return mapping


def validate_region_mapping(mapping: Mapping[str, int]) -> None:
    errors: List[str] = []
    split_counts = Counter(split_and_index(name)[0] for name in mapping)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        errors.append(f"split counts {dict(split_counts)} != {EXPECTED_SPLIT_COUNTS}")
    if len(mapping) != EXPECTED_TOTAL:
        errors.append(f"total {len(mapping)} != {EXPECTED_TOTAL}")
    region_counts = Counter(mapping.values())
    if dict(sorted(region_counts.items())) != EXPECTED_REGION_TOTALS:
        errors.append(
            f"region counts {dict(sorted(region_counts.items()))} != {EXPECTED_REGION_TOTALS}"
        )
    expected_names = {
        f"{split}_{index}.png"
        for split, count in EXPECTED_SPLIT_COUNTS.items()
        for index in range(1, count + 1)
    }
    missing = sorted(expected_names - set(mapping), key=natural_sample_sort_key)
    extra = sorted(set(mapping) - expected_names, key=natural_sample_sort_key)
    if missing:
        errors.append(f"missing assignments: {missing[:10]}")
    if extra:
        errors.append(f"extra assignments: {extra[:10]}")
    if errors:
        raise LEVIRValidationError("Invalid LEVIR region mapping: " + "; ".join(errors))


FILENAME_REGION_MAP = build_filename_region_map()


def region_for_sample(value: Any) -> int:
    filename = normalize_sample_key(value, keep_extension=True)
    try:
        return FILENAME_REGION_MAP[filename]
    except KeyError as exc:
        raise LEVIRValidationError(f"No published region mapping for {filename}") from exc


def validate_month(month: str) -> Tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(month))
    if not match:
        raise LEVIRValidationError(f"Month must be YYYY-MM, got {month!r}")
    year, mon = (int(x) for x in match.groups())
    if not 1 <= mon <= 12:
        raise LEVIRValidationError(f"Invalid month: {month!r}")
    return year, mon


def month_date_bounds(month: str) -> Tuple[date, date]:
    year, mon = validate_month(month)
    return date(year, mon, 1), date(year, mon, calendar.monthrange(year, mon)[1])


def month_timestamp(month: str, policy: str = "month_end") -> str:
    start, end = month_date_bounds(month)
    if policy == "month_start":
        chosen = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    elif policy == "month_end":
        chosen = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        raise LEVIRValidationError(
            f"Unsupported timestamp policy {policy!r}; expected month_start or month_end"
        )
    return chosen.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_timestamp(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def coordinate_record_to_bbox(coords: Sequence[Any]) -> List[float]:
    """Convert official [lon_left_up, lat_left_up, lon_right_bottom, lat_right_bottom]."""

    if not isinstance(coords, Sequence) or isinstance(coords, (str, bytes)) or len(coords) != 4:
        raise LEVIRValidationError(f"Coordinate record must contain four values, got {coords!r}")
    try:
        west, north, east, south = (float(x) for x in coords)
    except (TypeError, ValueError) as exc:
        raise LEVIRValidationError(f"Non-numeric coordinate record: {coords!r}") from exc
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise LEVIRValidationError(f"Longitude outside [-180, 180]: {coords!r}")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise LEVIRValidationError(f"Latitude outside [-90, 90]: {coords!r}")
    if not west < east or not south < north:
        raise LEVIRValidationError(
            f"Invalid bbox orientation (west,south,east,north): {[west, south, east, north]}"
        )
    return [west, south, east, north]


def load_coordinate_json(path: os.PathLike[str] | str) -> Dict[str, List[float]]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LEVIRValidationError(f"Cannot read coordinate JSON {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise LEVIRValidationError("Coordinate JSON root must be an object")
    normalized: Dict[str, List[float]] = {}
    errors: List[str] = []
    for original_name, coords in raw.items():
        try:
            filename = normalize_sample_key(original_name, keep_extension=True)
            if filename in normalized:
                raise LEVIRValidationError(f"duplicate normalized key {filename}")
            normalized[filename] = coordinate_record_to_bbox(coords)
        except LEVIRValidationError as exc:
            errors.append(f"{original_name!r}: {exc}")
    expected = set(FILENAME_REGION_MAP)
    missing = sorted(expected - set(normalized), key=natural_sample_sort_key)
    extra = sorted(set(normalized) - expected, key=natural_sample_sort_key)
    if len(normalized) != EXPECTED_TOTAL:
        errors.append(f"record count {len(normalized)} != {EXPECTED_TOTAL}")
    if missing:
        errors.append(f"missing coordinate records: {missing[:10]}")
    if extra:
        errors.append(f"unexpected coordinate records: {extra[:10]}")
    if errors:
        raise LEVIRValidationError("Invalid official coordinate JSON: " + "; ".join(errors))
    return normalized


def build_source_records(
    coordinates: Mapping[str, Sequence[float]], timestamp_policy: str = "month_end"
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for filename in sorted(FILENAME_REGION_MAP, key=natural_sample_sort_key):
        if filename not in coordinates:
            raise LEVIRValidationError(f"Missing coordinate record for {filename}")
        bbox_raw = coordinates[filename]
        # load_coordinate_json has already converted records; callers can also
        # provide official raw records, detected by latitude orientation.
        if len(bbox_raw) != 4:
            raise LEVIRValidationError(f"Invalid bbox for {filename}: {bbox_raw!r}")
        values = [float(x) for x in bbox_raw]
        if values[0] < values[2] and values[1] < values[3]:
            bbox = values
        else:
            bbox = coordinate_record_to_bbox(values)
        region_id = FILENAME_REGION_MAP[filename]
        t1_month, t2_month = REGION_MONTHS[region_id]
        split, _ = split_and_index(filename)
        records.append(
            {
                "filename": filename,
                "source_pair_id": Path(filename).stem,
                "split": split,
                "region_id": region_id,
                "bbox": bbox,
                "t1": {
                    "image_month": t1_month,
                    "query_timestamp": month_timestamp(t1_month, timestamp_policy),
                },
                "t2": {
                    "image_month": t2_month,
                    "query_timestamp": month_timestamp(t2_month, timestamp_policy),
                },
            }
        )
    return records


def validate_local_levir(
    data_root: os.PathLike[str] | str,
    coordinate_names: Optional[Iterable[str]] = None,
    expected_size: Tuple[int, int] = (1024, 1024),
    inspect_images: bool = True,
) -> Dict[str, Any]:
    """Strictly validate local A/B/label split membership and image headers."""

    root = Path(data_root)
    errors: List[str] = []
    report: Dict[str, Any] = {"root": str(root), "splits": {}, "errors": errors}
    official_names = set(FILENAME_REGION_MAP)
    if coordinate_names is not None:
        normalized_coords = {
            normalize_sample_key(name, keep_extension=True) for name in coordinate_names
        }
        if normalized_coords != official_names:
            errors.append("Coordinate names do not match the complete official 637-name mapping")

    Image = None
    if inspect_images:
        try:
            from PIL import Image as PILImage

            Image = PILImage
        except ImportError as exc:
            raise LEVIRValidationError("Pillow is required for --data-root image validation") from exc

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        expected = {f"{split}_{index}.png" for index in range(1, expected_count + 1)}
        split_report: Dict[str, Any] = {"expected": expected_count, "parts": {}}
        names_by_part: Dict[str, set[str]] = {}
        for part in ("A", "B", "label"):
            directory = root / split / part
            if not directory.is_dir():
                errors.append(f"Missing directory: {directory}")
                names: set[str] = set()
            else:
                names = {p.name for p in directory.iterdir() if p.is_file()}
            names_by_part[part] = names
            missing = sorted(expected - names, key=natural_sample_sort_key)
            extra = sorted(names - expected, key=natural_sample_sort_key)
            split_report["parts"][part] = {
                "count": len(names), "missing": missing, "extra": extra
            }
            if missing or extra:
                errors.append(
                    f"{split}/{part}: missing={missing[:10]} extra={extra[:10]}"
                )
            if inspect_images and Image is not None and directory.is_dir():
                expected_mode = "L" if part == "label" else "RGB"
                for name in sorted(names & expected, key=natural_sample_sort_key):
                    path = directory / name
                    try:
                        with Image.open(path) as image:
                            if image.size != expected_size:
                                errors.append(f"{path}: size {image.size} != {expected_size}")
                            if image.mode != expected_mode:
                                errors.append(f"{path}: mode {image.mode} != {expected_mode}")
                            image.verify()
                    except Exception as exc:  # Pillow raises several format-specific exceptions.
                        errors.append(f"{path}: unreadable ({exc})")
        if not (names_by_part["A"] == names_by_part["B"] == names_by_part["label"]):
            errors.append(f"{split}: A/B/label filename sets are not identical")
        split_report["aligned_count"] = len(
            names_by_part["A"] & names_by_part["B"] & names_by_part["label"]
        )
        report["splits"][split] = split_report

    local_all = {
        name
        for split in report["splits"].values()
        for part in split["parts"].values()
        for name in []  # membership errors are recorded above; keeps report JSON compact
    }
    del local_all
    if errors:
        raise LEVIRValidationError(
            f"Local LEVIR validation failed with {len(errors)} error(s): " + "; ".join(errors[:20])
        )
    report["valid"] = True
    report["total_aligned"] = sum(x["aligned_count"] for x in report["splits"].values())
    return report


def tile_bbox(
    bbox: Sequence[float], x: int, y: int, tile_w: int, tile_h: int, width: int, height: int
) -> List[float]:
    west, south, east, north = (float(v) for v in bbox)
    if not west < east or not south < north:
        raise LEVIRValidationError(f"Invalid source bbox: {bbox!r}")
    if min(x, y, tile_w, tile_h) < 0 or tile_w <= 0 or tile_h <= 0:
        raise LEVIRValidationError("Tile coordinates and dimensions must be non-negative/positive")
    if x + tile_w > width or y + tile_h > height:
        raise LEVIRValidationError("Tile extends beyond source grid")
    tile_west = west + (x / width) * (east - west)
    tile_east = west + ((x + tile_w) / width) * (east - west)
    tile_north = north - (y / height) * (north - south)
    tile_south = north - ((y + tile_h) / height) * (north - south)
    return [tile_west, tile_south, tile_east, tile_north]


def calendar_month_window(month: str, expansion_days: int = 0) -> Tuple[date, date]:
    start, end = month_date_bounds(month)
    expansion = timedelta(days=max(0, int(expansion_days)))
    return start - expansion, end + expansion


def non_overlapping_windows(
    t1_month: str, t2_month: str, requested_expansion_days: int
) -> Tuple[Tuple[date, date], Tuple[date, date], int]:
    """Return symmetric windows, shrinking expansion if dates would overlap."""

    expansion = max(0, int(requested_expansion_days))
    while expansion >= 0:
        first = calendar_month_window(t1_month, expansion)
        second = calendar_month_window(t2_month, expansion)
        if first[1] < second[0]:
            return first, second, expansion
        expansion -= 1
    raise LEVIRValidationError(f"Published T1/T2 months overlap: {t1_month}, {t2_month}")


def is_before_history(query_timestamp: str, history_start: str) -> bool:
    return parse_utc_timestamp(query_timestamp) < parse_utc_timestamp(history_start)


def manifest_samples(document: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = document.get("samples", document.get("records", document))
    if isinstance(raw, list):
        samples = raw
    elif isinstance(raw, Mapping):
        samples = []
        for key, value in raw.items():
            if key in {"schema_version", "metadata", "generation_report"}:
                continue
            if not isinstance(value, Mapping):
                continue
            sample = dict(value)
            sample.setdefault("filename", normalize_sample_key(key, keep_extension=True))
            samples.append(sample)
    else:
        raise LEVIRValidationError("Manifest must contain a samples list or keyed record object")
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise LEVIRValidationError("Manifest sample entries must be objects")
        item = dict(sample)
        key = normalize_sample_key(
            item.get("filename", item.get("source_pair_id", "")), keep_extension=True
        )
        if key in seen:
            raise LEVIRValidationError(f"Duplicate manifest sample: {key}")
        seen.add(key)
        item["filename"] = key
        item.setdefault("source_pair_id", Path(key).stem)
        item.setdefault("split", split_and_index(key)[0])
        normalized.append(item)
    return sorted(normalized, key=lambda item: natural_sample_sort_key(item["filename"]))


def load_manifest(path: os.PathLike[str] | str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LEVIRValidationError(f"Cannot read manifest {source}: {exc}") from exc
    if not isinstance(document, MutableMapping):
        raise LEVIRValidationError(f"Manifest root must be an object: {source}")
    samples = manifest_samples(document)
    return dict(document), {Path(item["filename"]).stem: item for item in samples}


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        delay = _ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
        for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary_name, destination)
                break
            except PermissionError:
                if attempt + 1 >= _ATOMIC_REPLACE_ATTEMPTS:
                    raise
                time.sleep(delay)
                delay = min(delay * 2.0, _ATOMIC_REPLACE_MAX_DELAY_SECONDS)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_relative_to(path: os.PathLike[str] | str, root: os.PathLike[str] | str) -> str:
    try:
        relative = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError as exc:
        raise LEVIRValidationError(f"Path {path} is outside manifest root {root}") from exc
    return relative.as_posix()


def summarize_statuses(samples: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"by_split": {}, "by_region": {}}
    split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    region_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        split = str(sample.get("split", "unknown"))
        region = str(sample.get("region_id", "unknown"))
        for temporal_key in ("t1", "t2"):
            record = sample.get(temporal_key, {})
            status = str(record.get("status", "missing")) if isinstance(record, Mapping) else "missing"
            split_counts[split][f"{temporal_key}:{status}"] += 1
            region_counts[region][f"{temporal_key}:{status}"] += 1
    report["by_split"] = {k: dict(sorted(v.items())) for k, v in sorted(split_counts.items())}
    report["by_region"] = {
        k: dict(sorted(v.items())) for k, v in sorted(region_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
    }
    return report


__all__ = [
    "COORDINATE_SOURCE_URL",
    "EXPECTED_REGION_TOTALS",
    "EXPECTED_SPLIT_COUNTS",
    "EXPECTED_TOTAL",
    "FILENAME_REGION_MAP",
    "LEVIRValidationError",
    "REGION_MONTHS",
    "REGION_SPLIT_RANGES",
    "SCHEMA_VERSION",
    "atomic_write_json",
    "build_filename_region_map",
    "build_source_records",
    "calendar_month_window",
    "coordinate_record_to_bbox",
    "is_before_history",
    "load_coordinate_json",
    "load_manifest",
    "manifest_samples",
    "month_date_bounds",
    "month_timestamp",
    "natural_sample_sort_key",
    "non_overlapping_windows",
    "normalize_sample_key",
    "parse_utc_timestamp",
    "path_relative_to",
    "region_for_sample",
    "sha256_file",
    "split_and_index",
    "summarize_statuses",
    "tile_bbox",
    "validate_local_levir",
    "validate_region_mapping",
]
