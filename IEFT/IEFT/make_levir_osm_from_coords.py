import os
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional

import requests


DEFAULT_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def bbox_from_coords(coords: List[float]) -> Tuple[float, float, float, float]:
    """
    Input format from LEVIR_CD_name_coords.json:
    [lon_left_up, lat_left_up, lon_right_bottom, lat_right_bottom]

    Returns Overpass bbox:
    (south, west, north, east)
    """
    if len(coords) != 4:
        raise ValueError(f"Expected 4 coords, got {coords}")
    lon_lu, lat_lu, lon_rb, lat_rb = coords
    west = min(lon_lu, lon_rb)
    east = max(lon_lu, lon_rb)
    south = min(lat_lu, lat_rb)
    north = max(lat_lu, lat_rb)
    return south, west, north, east


def strip_png(name: str) -> str:
    return os.path.splitext(name)[0]


def safe_get_tags(element: Dict[str, Any]) -> Dict[str, str]:
    tags = element.get("tags", {})
    return tags if isinstance(tags, dict) else {}


def build_filtered_query(south: float, west: float, north: float, east: float) -> str:
    """
    Much lighter than querying all nodes/ways/relations in the bbox.
    Only asks for the tags we actually summarize.
    """
    return f"""
    [out:json][timeout:40];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});

      way["highway"]({south},{west},{north},{east});
      relation["highway"]({south},{west},{north},{east});

      way["railway"]({south},{west},{north},{east});
      relation["railway"]({south},{west},{north},{east});

      way["landuse"]({south},{west},{north},{east});
      relation["landuse"]({south},{west},{north},{east});

      way["natural"]({south},{west},{north},{east});
      relation["natural"]({south},{west},{north},{east});

      way["waterway"]({south},{west},{north},{east});
      relation["waterway"]({south},{west},{north},{east});

      node["amenity"]({south},{west},{north},{east});
      way["amenity"]({south},{west},{north},{east});
      relation["amenity"]({south},{west},{north},{east});

      way["leisure"]({south},{west},{north},{east});
      relation["leisure"]({south},{west},{north},{east});
    );
    out tags center qt;
    """


def request_overpass_with_retry(
    south: float,
    west: float,
    north: float,
    east: float,
    endpoints: List[str],
    max_retries: int = 6,
    base_sleep: float = 4.0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    query = build_filtered_query(south, west, north, east)
    sess = session or requests.Session()
    last_err = None

    for attempt in range(max_retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            resp = sess.post(
                endpoint,
                data={"data": query},
                timeout=120,
                headers={"User-Agent": "LEVIR-OSM-Builder/1.0"},
            )

            if resp.status_code == 200:
                return resp.json()

            # 429 / 504 / temporary failures -> backoff and retry
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"{resp.status_code} from {endpoint}")
                sleep_s = base_sleep * (2 ** min(attempt, 4))
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()

        except requests.RequestException as e:
            last_err = e
            sleep_s = base_sleep * (2 ** min(attempt, 4))
            time.sleep(sleep_s)

    raise RuntimeError(f"Overpass failed after {max_retries} retries: {last_err}")


def summarize_osm(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    building_count = 0
    road_count = 0
    water_count = 0
    vegetation_count = 0
    construction_count = 0
    industrial_count = 0
    residential_count = 0
    amenity_count = 0
    railway_count = 0

    building_types = set()
    road_types = set()
    landuse_types = set()
    natural_types = set()
    amenity_types = set()

    for el in elements:
        tags = safe_get_tags(el)

        if "building" in tags:
            building_count += 1
            building_types.add(tags.get("building", "yes"))

        if "highway" in tags:
            road_count += 1
            road_types.add(tags.get("highway", "road"))

        if "railway" in tags:
            railway_count += 1

        if "amenity" in tags:
            amenity_count += 1
            amenity_types.add(tags.get("amenity", ""))

        landuse = tags.get("landuse", "")
        natural = tags.get("natural", "")
        waterway = tags.get("waterway", "")
        leisure = tags.get("leisure", "")

        if landuse:
            landuse_types.add(landuse)
        if natural:
            natural_types.add(natural)

        if landuse in {"forest", "grass", "meadow", "farmland", "orchard", "vineyard"}:
            vegetation_count += 1
        if natural in {"wood", "tree_row", "grassland", "scrub", "heath", "wetland"}:
            vegetation_count += 1
        if leisure in {"park", "garden", "golf_course", "pitch"}:
            vegetation_count += 1

        if natural in {"water", "wetland"} or waterway or landuse in {"reservoir", "basin"}:
            water_count += 1

        if landuse == "construction" or tags.get("building") == "construction":
            construction_count += 1

        if landuse in {"industrial", "commercial"}:
            industrial_count += 1

        if landuse == "residential":
            residential_count += 1

    return {
        "building_count": building_count,
        "road_count": road_count,
        "railway_count": railway_count,
        "water_count": water_count,
        "vegetation_count": vegetation_count,
        "construction_count": construction_count,
        "industrial_count": industrial_count,
        "residential_count": residential_count,
        "amenity_count": amenity_count,
        "building_types": sorted([x for x in building_types if x]),
        "road_types": sorted([x for x in road_types if x]),
        "landuse_types": sorted([x for x in landuse_types if x]),
        "natural_types": sorted([x for x in natural_types if x]),
        "amenity_types": sorted([x for x in amenity_types if x]),
    }


def bucketize_presence(count: int, low: int, med: int, high: int) -> str:
    if count <= low:
        return "very low"
    if count <= med:
        return "low"
    if count <= high:
        return "moderate"
    return "high"


def make_phrases(summary: Dict[str, Any]) -> List[str]:
    phrases = []

    built = bucketize_presence(summary["building_count"], 2, 8, 20)
    roads = bucketize_presence(summary["road_count"] + summary["railway_count"], 1, 5, 12)
    veg = bucketize_presence(summary["vegetation_count"], 1, 4, 10)

    phrases.append(f"{built} built-up presence")
    phrases.append(f"{roads} transport structure")
    phrases.append(f"{veg} vegetation context")

    if summary["residential_count"] > 0:
        phrases.append("residential context")
    if summary["industrial_count"] > 0:
        phrases.append("industrial or commercial context")
    if summary["construction_count"] > 0:
        phrases.append("construction-related context")
    if summary["water_count"] > 0:
        phrases.append("water-related context")
    if summary["amenity_count"] > 0:
        phrases.append("amenity-related context")

    return phrases[:6]


def make_tags(summary: Dict[str, Any]) -> List[str]:
    tags = []

    if summary["building_count"] > 0:
        tags.append("building")
    if summary["road_count"] > 0:
        tags.append("road")
    if summary["railway_count"] > 0:
        tags.append("railway")
    if summary["vegetation_count"] > 0:
        tags.append("vegetation")
    if summary["water_count"] > 0:
        tags.append("water")
    if summary["residential_count"] > 0:
        tags.append("residential")
    if summary["industrial_count"] > 0:
        tags.append("industrial")
    if summary["construction_count"] > 0:
        tags.append("construction")
    if summary["amenity_count"] > 0:
        tags.append("amenity")

    return tags


def make_text_v21(summary: Dict[str, Any]) -> str:
    built = bucketize_presence(summary["building_count"], 2, 8, 20)
    roads = bucketize_presence(summary["road_count"] + summary["railway_count"], 1, 5, 12)
    veg = bucketize_presence(summary["vegetation_count"], 1, 4, 10)

    extra = []
    if summary["residential_count"] > 0:
        extra.append("residential context")
    if summary["industrial_count"] > 0:
        extra.append("industrial or commercial context")
    if summary["construction_count"] > 0:
        extra.append("construction-related area")
    if summary["water_count"] > 0:
        extra.append("water-related area")
    if summary["amenity_count"] > 0:
        extra.append("amenity-related area")

    extra_text = ", with " + ", ".join(extra[:2]) if extra else ""
    return (
        f"This patch shows {built} built-up presence, "
        f"{roads} transport structure, and {veg} vegetation{extra_text}."
    )


def make_summary_label(summary: Dict[str, Any]) -> str:
    if summary["residential_count"] > 0 and summary["building_count"] > 5:
        return "residential neighborhood"
    if summary["industrial_count"] > 0:
        return "industrial or commercial area"
    if summary["water_count"] > 0 and summary["building_count"] == 0:
        return "water-related open area"
    if summary["vegetation_count"] > 0 and summary["building_count"] <= 2:
        return "vegetated open area"
    if summary["building_count"] > 0:
        return "built-up area"
    return "sparse undeveloped area"


def make_source_text(summary: Dict[str, Any]) -> str:
    parts = []
    if summary["building_types"]:
        parts.append("building=" + ",".join(summary["building_types"][:5]))
    if summary["road_types"]:
        parts.append("highway=" + ",".join(summary["road_types"][:5]))
    if summary["landuse_types"]:
        parts.append("landuse=" + ",".join(summary["landuse_types"][:5]))
    if summary["natural_types"]:
        parts.append("natural=" + ",".join(summary["natural_types"][:5]))
    if summary["amenity_types"]:
        parts.append("amenity=" + ",".join(summary["amenity_types"][:5]))
    return " ; ".join(parts)


def build_entry(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text_v21": make_text_v21(summary),
        "summary": make_summary_label(summary),
        "source_text": make_source_text(summary),
        "tags": make_tags(summary),
        "phrases": make_phrases(summary),
        "osm_struct_hint": {
            "building_count": summary["building_count"],
            "road_count": summary["road_count"],
            "railway_count": summary["railway_count"],
            "water_count": summary["water_count"],
            "vegetation_count": summary["vegetation_count"],
            "construction_count": summary["construction_count"],
            "industrial_count": summary["industrial_count"],
            "residential_count": summary["residential_count"],
            "amenity_count": summary["amenity_count"],
        },
    }


def should_retry_failed(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return True
    summary = str(entry.get("summary", "")).lower()
    return "failed" in summary or entry.get("text_v21") == "no_osm_context"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords_json", type=str, required=True, help="Path to LEVIR_CD_name_coords.json")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output LEVIR OSM JSON")
    parser.add_argument("--sleep_sec", type=float, default=4.0, help="Pause between items")
    parser.add_argument("--max_retries", type=int, default=6, help="Retries per sample")
    parser.add_argument("--resume", action="store_true", help="Resume if output already exists")
    parser.add_argument("--retry_failed", action="store_true", help="When resuming, retry entries that previously failed")
    parser.add_argument("--start_index", type=int, default=0, help="Start from this sorted index")
    parser.add_argument("--limit", type=int, default=-1, help="Process at most this many items")
    parser.add_argument("--endpoints", nargs="*", default=DEFAULT_ENDPOINTS, help="Overpass endpoints to rotate")
    args = parser.parse_args()

    with open(args.coords_json, "r", encoding="utf-8") as f:
        coords_map = json.load(f)

    if not isinstance(coords_map, dict):
        raise ValueError("coords_json must contain a dictionary")

    out = {}
    if args.resume and os.path.isfile(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            out = json.load(f)

    keys = sorted(coords_map.keys())
    if args.start_index > 0:
        keys = keys[args.start_index:]
    if args.limit > 0:
        keys = keys[:args.limit]

    total = len(keys)
    sess = requests.Session()

    for idx, name in enumerate(keys, start=1):
        stem = strip_png(name)

        if args.resume and stem in out:
            if args.retry_failed and should_retry_failed(out[stem]):
                pass
            else:
                print(f"[{idx}/{total}] skip {stem}")
                continue

        coords = coords_map[name]
        south, west, north, east = bbox_from_coords(coords)

        try:
            data = request_overpass_with_retry(
                south=south,
                west=west,
                north=north,
                east=east,
                endpoints=args.endpoints,
                max_retries=args.max_retries,
                base_sleep=max(1.0, args.sleep_sec),
                session=sess,
            )
            elements = data.get("elements", [])
            summary = summarize_osm(elements)
            out[stem] = build_entry(summary)
            print(f"[{idx}/{total}] ok {stem} | elements={len(elements)}")
        except Exception as e:
            out[stem] = {
                "text_v21": "no_osm_context",
                "summary": "osm query failed",
                "source_text": str(e),
                "tags": [],
                "phrases": [],
            }
            print(f"[{idx}/{total}] fail {stem} | {e}")

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        time.sleep(max(0.0, args.sleep_sec))

    print(f"[DONE] wrote {args.output_json}")


if __name__ == "__main__":
    main()