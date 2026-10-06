"""Dependency-light historical OSM GeoJSON selection and summarization.

The offline generator and runtime loader share this module so a cached source
snapshot is summarized identically at source and tile scale. It contains no
HTTP client and is safe to import in DataLoader workers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


OSM_SPATIAL_CHANNEL_ORDER = (
    "building",
    "transport",
    "water",
    "vegetation",
    "land_use",
)


def feature_tags(feature: Mapping[str, Any]) -> Dict[str, Any]:
    properties = feature.get("properties", {})
    if not isinstance(properties, Mapping):
        return {}
    nested = properties.get("tags")
    if isinstance(nested, Mapping):
        return {str(key): value for key, value in nested.items()}
    return {
        str(key): value
        for key, value in properties.items()
        if not str(key).startswith("@")
        and key not in {"metadata", "type", "geometry", "id", "osmId"}
    }


def _coordinate_pairs(value: Any) -> Iterator[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)):
        return
    if (
        len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return
    for child in value:
        yield from _coordinate_pairs(child)


def geometry_bbox(geometry: Optional[Mapping[str, Any]]) -> Optional[List[float]]:
    """Return ``[west, south, east, north]`` for any GeoJSON geometry."""

    if not isinstance(geometry, Mapping):
        return None
    if geometry.get("type") == "GeometryCollection":
        pairs = (
            pair
            for child in geometry.get("geometries", [])
            if isinstance(child, Mapping)
            for pair in _coordinate_pairs(child.get("coordinates", []))
        )
    else:
        pairs = _coordinate_pairs(geometry.get("coordinates", []))
    west = south = float("inf")
    east = north = float("-inf")
    found = False
    for longitude, latitude in pairs:
        found = True
        west = min(west, longitude)
        east = max(east, longitude)
        south = min(south, latitude)
        north = max(north, latitude)
    return [west, south, east, north] if found else None


def is_axis_aligned_bbox_polygon(geometry: Optional[Mapping[str, Any]]) -> bool:
    """Return whether a geometry is a GeoJSON polygon encoding one bbox.

    The ohsome ``/elements/bbox`` endpoint represents every OSM element by an
    axis-aligned Polygon, including degenerate polygons for point elements.
    This check deliberately accepts those point/line degeneracies while
    rejecting arbitrary polygons that could be mistaken for endpoint output.
    """

    if not isinstance(geometry, Mapping) or geometry.get("type") != "Polygon":
        return False
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 1:
        return False
    ring = coordinates[0]
    if not isinstance(ring, list) or len(ring) < 4:
        return False
    pairs = list(_coordinate_pairs(ring))
    if len(pairs) != len(ring) or len(pairs) < 4:
        return False
    if pairs[0] != pairs[-1]:
        return False
    if not all(math.isfinite(value) for pair in pairs for value in pair):
        return False
    west = min(pair[0] for pair in pairs)
    east = max(pair[0] for pair in pairs)
    south = min(pair[1] for pair in pairs)
    north = max(pair[1] for pair in pairs)

    def matches(value: float, first: float, second: float) -> bool:
        return math.isclose(value, first, rel_tol=0.0, abs_tol=1e-12) or math.isclose(
            value, second, rel_tol=0.0, abs_tol=1e-12
        )

    if not all(matches(x, west, east) and matches(y, south, north) for x, y in pairs):
        return False
    expected_corners = {
        (west, south),
        (west, north),
        (east, south),
        (east, north),
    }
    actual_corners = set(pairs)
    return expected_corners.issubset(actual_corners)


def bboxes_intersect(first: Sequence[float], second: Sequence[float]) -> bool:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("Bounding boxes must contain west,south,east,north")
    a_west, a_south, a_east, a_north = (float(value) for value in first)
    b_west, b_south, b_east, b_north = (float(value) for value in second)
    if not a_west <= a_east or not a_south <= a_north:
        raise ValueError(f"Invalid first bbox: {list(first)!r}")
    if not b_west < b_east or not b_south < b_north:
        raise ValueError(f"Invalid second bbox: {list(second)!r}")
    return not (
        a_east < b_west
        or a_west > b_east
        or a_north < b_south
        or a_south > b_north
    )


def select_features_for_bbox(
    document: Mapping[str, Any], bbox: Sequence[float]
) -> Dict[str, Any]:
    """Select cached features whose extent intersects a tile bbox locally.

    Ohsome already clips geometry to the source bbox. Extent intersection is a
    dependency-free, conservative selection for features crossing tile edges.
    """

    features = document.get("features", [])
    if document.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise ValueError("OSM cache must be a GeoJSON FeatureCollection")
    selected: List[Mapping[str, Any]] = []
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        extent = geometry_bbox(feature.get("geometry"))
        if extent is not None and bboxes_intersect(extent, bbox):
            selected.append(feature)
    return {"type": "FeatureCollection", "features": selected}


def summarize_feature_collection(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Create the deterministic legacy-compatible 16-D OSM representation."""

    features = document.get("features", [])
    if not isinstance(features, list):
        raise ValueError("ohsome response field 'features' must be a list")
    counts: Counter[str] = Counter()
    values: Dict[str, Counter[str]] = defaultdict(Counter)
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        tags = feature_tags(feature)
        if "building" in tags:
            counts["building"] += 1
            values["building"][str(tags["building"])] += 1
        if "highway" in tags:
            counts["road"] += 1
            values["highway"][str(tags["highway"])] += 1
        if "railway" in tags:
            counts["railway"] += 1
            values["railway"][str(tags["railway"])] += 1
        if (
            "waterway" in tags
            or tags.get("natural") in {"water", "wetland", "bay"}
            or "water" in tags
        ):
            counts["water"] += 1
        if (
            tags.get("natural") in {"wood", "grassland", "scrub", "heath", "tree_row"}
            or tags.get("landuse") in {"forest", "grass", "meadow", "orchard", "vineyard"}
            or tags.get("leisure") in {"park", "garden", "nature_reserve"}
        ):
            counts["vegetation"] += 1
        if tags.get("landuse") == "residential":
            counts["residential"] += 1
        if tags.get("landuse") in {"industrial", "commercial", "retail"}:
            counts["industrial_commercial"] += 1
        if tags.get("landuse") == "construction" or tags.get("building") == "construction":
            counts["construction"] += 1
        if "amenity" in tags:
            counts["amenity"] += 1
            values["amenity"][str(tags["amenity"])] += 1

    def present(name: str) -> float:
        return float(counts[name] > 0)

    def normalized(name: str, cap: float) -> float:
        return min(float(counts[name]), cap) / cap

    struct = [0.0] * 16
    struct[0] = present("building")
    struct[1] = present("road")
    struct[2] = present("railway")
    struct[3] = present("water")
    struct[4] = present("vegetation")
    struct[5] = present("residential")
    struct[6] = present("industrial_commercial")
    struct[7] = present("construction")
    struct[8] = present("amenity")
    struct[9] = normalized("building", 120.0)
    struct[10] = min(1.0, (counts["road"] + counts["railway"]) / 180.0)
    struct[11] = normalized("water", 30.0)
    struct[12] = normalized("vegetation", 60.0)
    struct[13] = min(
        1.0,
        0.55 * struct[0] + 0.25 * struct[5] + 0.20 * struct[6] + 0.35 * struct[9],
    )
    struct[14] = min(1.0, 0.55 * struct[1] + 0.20 * struct[2] + 0.25 * struct[10])
    struct[15] = min(
        1.0,
        max(
            0.0,
            0.50 * struct[13]
            + 0.30 * struct[14]
            - 0.20 * struct[3]
            - 0.10 * struct[4]
            + 0.25 * struct[7],
        ),
    )
    phrases = [
        f"{counts[key]} {label}"
        for key, label in (
            ("building", "buildings"),
            ("road", "roads"),
            ("railway", "railways"),
            ("water", "water features"),
            ("vegetation", "vegetation or parks"),
            ("residential", "residential land use"),
            ("industrial_commercial", "industrial or commercial land use"),
            ("construction", "construction"),
            ("amenity", "amenities"),
        )
        if counts[key]
    ]
    return {
        "osm_struct": struct,
        "osm_text": "Historical OSM snapshot: "
        + (", ".join(phrases) if phrases else "no matching mapped features"),
        "counts": dict(sorted(counts.items())),
        "types": {
            key: [name for name, _ in counter.most_common(10)]
            for key, counter in sorted(values.items())
        },
    }


def summarize_tile(document: Mapping[str, Any], bbox: Sequence[float]) -> Dict[str, Any]:
    selected = select_features_for_bbox(document, bbox)
    summary = summarize_feature_collection(selected)
    summary["tile_feature_count"] = len(selected["features"])
    return summary


def _spatial_categories(tags: Mapping[str, Any]) -> Tuple[int, ...]:
    """Map one OSM tag dictionary to the five retained guidance channels.

    Channels are deliberately non-exclusive.  For example, a forest polygon is
    useful both as vegetation and as mapped land use.  These channels express
    mapped context only; an all-zero map is never interpreted as proof that a
    physical feature is absent.
    """

    categories: List[int] = []
    if "building" in tags:
        categories.append(0)
    if "highway" in tags or "railway" in tags or "public_transport" in tags:
        categories.append(1)
    if (
        "waterway" in tags
        or "water" in tags
        or tags.get("natural") in {"water", "wetland", "bay"}
    ):
        categories.append(2)
    if (
        tags.get("natural") in {"wood", "grassland", "scrub", "heath", "tree_row"}
        or tags.get("landuse")
        in {"forest", "grass", "meadow", "orchard", "vineyard", "recreation_ground"}
        or tags.get("leisure") in {"park", "garden", "nature_reserve"}
    ):
        categories.append(3)
    if "landuse" in tags or "amenity" in tags or "leisure" in tags:
        categories.append(4)
    return tuple(categories)


def _geometry_parts(geometry: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, Any]]:
    """Yield simple GeoJSON geometry parts without an optional GIS dependency."""

    if not isinstance(geometry, Mapping):
        return
    geometry_type = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates")
    if geometry_type == "GeometryCollection":
        for child in geometry.get("geometries", []):
            if isinstance(child, Mapping):
                yield from _geometry_parts(child)
    elif geometry_type == "Polygon":
        if isinstance(coordinates, list) and coordinates:
            yield "polygon", coordinates[0]
    elif geometry_type == "MultiPolygon":
        if isinstance(coordinates, list):
            for polygon in coordinates:
                if isinstance(polygon, list) and polygon:
                    yield "polygon", polygon[0]
    elif geometry_type == "LineString":
        yield "line", coordinates
    elif geometry_type == "MultiLineString":
        if isinstance(coordinates, list):
            for line in coordinates:
                yield "line", line
    elif geometry_type == "Point":
        yield "point", coordinates
    elif geometry_type == "MultiPoint":
        if isinstance(coordinates, list):
            for point in coordinates:
                yield "point", point


def rasterize_feature_collection(
    document: Mapping[str, Any],
    bbox: Sequence[float],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize cached OSM geometry to tile-aligned five-channel guidance.

    The function is deterministic, offline-only, and intentionally lightweight.
    Exact ohsome geometries are drawn as polygons/lines/points.  If a manifest
    records the documented ``elements/bbox`` fallback, its axis-aligned feature
    extents are drawn conservatively and retain that provenance in the manifest.
    """

    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("OSM raster dimensions must be positive")
    if len(bbox) != 4:
        raise ValueError("OSM raster bbox must contain west,south,east,north")
    west, south, east, north = (float(value) for value in bbox)
    if not west < east or not south < north:
        raise ValueError(f"Invalid OSM raster bbox: {list(bbox)!r}")

    selected = select_features_for_bbox(document, bbox)
    images = [Image.new("L", (int(width), int(height)), 0) for _ in OSM_SPATIAL_CHANNEL_ORDER]
    draws = [ImageDraw.Draw(image) for image in images]

    def pixel(point: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            longitude, latitude = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            return None
        x = (longitude - west) / (east - west) * max(0, int(width) - 1)
        y = (north - latitude) / (north - south) * max(0, int(height) - 1)
        return x, y

    line_width = max(1, int(round(min(int(height), int(width)) / 128.0)))
    point_radius = max(1, line_width)
    for feature in selected["features"]:
        if not isinstance(feature, Mapping):
            continue
        categories = _spatial_categories(feature_tags(feature))
        if not categories:
            continue
        for part_type, raw_part in _geometry_parts(feature.get("geometry")):
            if part_type == "point":
                point_value = pixel(raw_part)
                if point_value is None:
                    continue
                x, y = point_value
                shape = (x - point_radius, y - point_radius, x + point_radius, y + point_radius)
                for category in categories:
                    draws[category].ellipse(shape, fill=255)
                continue
            if not isinstance(raw_part, (list, tuple)):
                continue
            points = [value for value in (pixel(item) for item in raw_part) if value is not None]
            if not points:
                continue
            if part_type == "polygon" and len(points) >= 3:
                for category in categories:
                    draws[category].polygon(points, fill=255)
            elif len(points) >= 2:
                for category in categories:
                    draws[category].line(points, fill=255, width=line_width)
            else:
                x, y = points[0]
                shape = (x - point_radius, y - point_radius, x + point_radius, y + point_radius)
                for category in categories:
                    draws[category].ellipse(shape, fill=255)

    return np.stack(
        [(np.asarray(image, dtype=np.uint8) > 0).astype(np.float32) for image in images],
        axis=-1,
    )


__all__ = [
    "bboxes_intersect",
    "feature_tags",
    "geometry_bbox",
    "is_axis_aligned_bbox_polygon",
    "OSM_SPATIAL_CHANNEL_ORDER",
    "rasterize_feature_collection",
    "select_features_for_bbox",
    "summarize_feature_collection",
    "summarize_tile",
]
