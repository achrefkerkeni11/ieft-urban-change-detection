"""Canonical full-scene visual exporter with the historical horizontal panel style.

This exporter is intentionally VISUALIZATION-ONLY.

It does NOT:
- reload the model,
- rerun GPU inference,
- tune a threshold,
- apply the historical component-completion / precision / no-change post-processing.

Instead it consumes the already-generated canonical full-scene NPZ archives
from ``predict_full_scenes.py`` and the frozen official TEST report from
``evaluate_full_scenes.py``.

The probability array is auto-discovered and accepted only if thresholding it
with the frozen TEST threshold reproduces the official TEST confusion matrix
exactly.  This keeps the visual output scientifically aligned with the reported
IoU/F1/Precision/Recall/OA.

The panel layout intentionally follows the old exporter style: one long
horizontal row with white title bars.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Preference only. A key is NEVER accepted just because of its name: it must
# reproduce the official confusion matrix exactly at the frozen threshold.
PREFERRED_PROBABILITY_NAMES = (
    "probability",
    "probabilities",
    "prob",
    "prob_map",
    "full_scene_prob",
    "full_scene_probability",
    "scene_prob",
    "scene_probability",
    "semantic_prob",
    "semantic_probability",
    "stitched_prob",
    "stitched_probability",
    "change_prob",
    "change_probability",
    "pred_prob",
    "pred_prob_raw",
    "change_refined_map_up",
)

LOW_PRIORITY_TERMS = (
    "center",
    "offset",
    "instance",
    "coverage",
    "weight",
    "valid",
    "mask",
    "label",
    "ground",
    "gt",
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def float01_to_uint8_gray(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.asarray(arr01, dtype=np.float32)
    arr01 = np.nan_to_num(arr01, nan=0.0, posinf=0.0, neginf=0.0)
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0).round().astype(np.uint8)


def binary_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return ((np.asarray(mask) > 0).astype(np.uint8) * 255)


def save_png_gray(arr01: np.ndarray, out_path: str | Path) -> None:
    Image.fromarray(float01_to_uint8_gray(arr01), mode="L").save(out_path)


def save_png_binary(mask: np.ndarray, out_path: str | Path) -> None:
    Image.fromarray(binary_mask_to_uint8(mask), mode="L").save(out_path)


def save_png_rgb(arr_uint8: np.ndarray, out_path: str | Path) -> None:
    Image.fromarray(np.asarray(arr_uint8, dtype=np.uint8), mode="RGB").save(out_path)


def make_binary_overlay(
    base_rgb: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float = 0.55,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.float32)
    mask = (np.asarray(binary_mask) > 0).astype(np.float32)[..., None]
    paint = np.zeros_like(base, dtype=np.float32)
    paint[..., 0] = float(color[0])
    paint[..., 1] = float(color[1])
    paint[..., 2] = float(color[2])
    overlay = base * (1.0 - mask * alpha) + paint * (mask * alpha)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _load_font(font_size: int):
    font_candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def add_title_bar(
    img: np.ndarray,
    title: str,
    bar_h: int = 40,
    font_size: int = 18,
) -> np.ndarray:
    """Same white title-bar visual style used by the historical exporter."""
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w, c = img.shape
    canvas = np.full((h + bar_h, w, c), 255, dtype=np.uint8)
    canvas[bar_h:] = img
    pil = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size)
    draw.text(
        (8, max(4, (bar_h - font_size) // 2)),
        title,
        fill=(0, 0, 0),
        font=font,
    )
    return np.asarray(pil)


def make_panel(
    images_with_titles: Sequence[Tuple[str, np.ndarray]],
    pad: int = 8,
    title_bar_h: int = 40,
    font_size: int = 18,
) -> np.ndarray:
    """Historical one-row horizontal panel layout."""
    prepared = [
        add_title_bar(
            img,
            title,
            bar_h=title_bar_h,
            font_size=font_size,
        )
        for title, img in images_with_titles
    ]

    max_h = max(p.shape[0] for p in prepared)
    total_w = sum(p.shape[1] for p in prepared) + pad * (len(prepared) + 1)

    panel = np.full(
        (max_h + 2 * pad, total_w, 3),
        255,
        dtype=np.uint8,
    )

    x = pad
    for img in prepared:
        h, w, _ = img.shape
        y = pad + (max_h - h) // 2
        panel[y : y + h, x : x + w] = img
        x += w + pad

    return panel


def natural_scene_key(scene_id: str) -> Tuple[str, int]:
    head, sep, tail = scene_id.rpartition("_")
    if sep and tail.isdigit():
        return head, int(tail)
    return scene_id, -1


def _is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTS)


# ---------------------------------------------------------------------------
# LEVIR source files
# ---------------------------------------------------------------------------

def find_split_root(data_root: Path, split: str) -> Path:
    for candidate in (split, split.lower(), split.upper(), split.capitalize()):
        path = data_root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find split '{split}' under {data_root}"
    )


def find_subdir(split_root: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = split_root / name
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find any of {list(names)} under {split_root}"
    )


def find_scene_file(directory: Path, scene_id: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{scene_id}{ext}"
        if path.is_file():
            return path

    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and path.stem == scene_id
    ]

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find image for scene '{scene_id}' in {directory}"
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_binary(path: Path) -> np.ndarray:
    return (
        np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Temporal-delta visualization from the historical exporter
# ---------------------------------------------------------------------------

def _gradient_magnitude(gray01: np.ndarray) -> np.ndarray:
    sx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=np.float32,
    )
    sy = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=np.float32,
    )
    gx = ndi.convolve(
        gray01.astype(np.float32),
        sx,
        mode="reflect",
    )
    gy = ndi.convolve(
        gray01.astype(np.float32),
        sy,
        mode="reflect",
    )
    return np.sqrt(gx * gx + gy * gy + 1e-6).astype(np.float32)


def robust_normalize(
    arr: np.ndarray,
    q_low: float = 0.02,
    q_high: float = 0.98,
    eps: float = 1e-6,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))

    if abs(hi - lo) < eps:
        mn = float(arr.min())
        mx = float(arr.max())
        if abs(mx - mn) < eps:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip(
            (arr - mn) / (mx - mn + eps),
            0.0,
            1.0,
        )

    return np.clip(
        (arr - lo) / (hi - lo + eps),
        0.0,
        1.0,
    )


def compute_temporal_delta_support(
    t1_rgb_u8: np.ndarray,
    t2_rgb_u8: np.ndarray,
) -> np.ndarray:
    """Historical visual temporal-delta map; never modifies predictions."""
    t1 = t1_rgb_u8.astype(np.float32) / 255.0
    t2 = t2_rgb_u8.astype(np.float32) / 255.0

    diff_rgb = np.abs(t2 - t1).mean(axis=2)

    g1 = t1.mean(axis=2)
    g2 = t2.mean(axis=2)

    e1 = _gradient_magnitude(g1)
    e2 = _gradient_magnitude(g2)
    edge_diff = np.abs(e2 - e1)

    delta = robust_normalize(
        0.68 * robust_normalize(diff_rgb)
        + 0.32 * robust_normalize(edge_diff)
    )
    return delta


# ---------------------------------------------------------------------------
# Official report + prediction manifest
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return raw


def load_official_report(path: Path) -> Dict[str, Any]:
    report = load_json(path)
    evaluation = report.get("evaluation")

    if not isinstance(evaluation, dict):
        raise RuntimeError(
            f"Official report has no 'evaluation' object: {path}"
        )

    return report


def load_prediction_manifest(prediction_dir: Path) -> Dict[str, Any]:
    manifest_path = prediction_dir / "prediction_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing prediction manifest: {manifest_path}"
        )

    manifest = load_json(manifest_path)
    archives = manifest.get("archives")

    if not isinstance(archives, list) or not archives:
        raise RuntimeError(
            f"prediction_manifest.json has no non-empty 'archives' list: "
            f"{manifest_path}"
        )

    return manifest


def canonical_manifest_rows(
    prediction_dir: Path,
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_scene_ids = set()
    seen_archives = set()

    for item in manifest["archives"]:
        if not isinstance(item, dict):
            raise RuntimeError(
                "Invalid non-object archive entry in prediction manifest."
            )

        scene_id = str(item.get("scene_id", "")).strip()
        archive_name = str(item.get("archive", "")).strip()

        if not scene_id:
            raise RuntimeError(
                f"Manifest entry has no scene_id: {item}"
            )
        if not archive_name:
            raise RuntimeError(
                f"Manifest entry has no archive: {item}"
            )
        if scene_id in seen_scene_ids:
            raise RuntimeError(
                f"Duplicate scene_id in manifest: {scene_id}"
            )
        if archive_name in seen_archives:
            raise RuntimeError(
                f"Duplicate archive in manifest: {archive_name}"
            )

        archive_path = prediction_dir / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Archive listed by manifest is missing: {archive_path}"
            )

        source_height = int(item.get("source_height", 0))
        source_width = int(item.get("source_width", 0))

        if source_height <= 0 or source_width <= 0:
            raise RuntimeError(
                f"Invalid source dimensions for {scene_id}: "
                f"{source_height} x {source_width}"
            )

        row = dict(item)
        row["scene_id"] = scene_id
        row["archive"] = archive_name
        row["archive_path"] = archive_path
        row["source_height"] = source_height
        row["source_width"] = source_width
        rows.append(row)

        seen_scene_ids.add(scene_id)
        seen_archives.add(archive_name)

    rows.sort(key=lambda row: natural_scene_key(row["scene_id"]))
    return rows


# ---------------------------------------------------------------------------
# NPZ probability discovery
# ---------------------------------------------------------------------------

def safe_read_npz_numeric(
    npz: np.lib.npyio.NpzFile,
    key: str,
) -> Optional[np.ndarray]:
    """Read one key without letting object metadata invalidate the whole NPZ."""
    try:
        arr = npz[key]
    except Exception:
        return None

    try:
        arr = np.asarray(arr)
    except Exception:
        return None

    if not np.issubdtype(arr.dtype, np.number):
        return None

    return arr


def squeeze_scene_array(
    arr: np.ndarray,
    expected_hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    arr = np.squeeze(np.asarray(arr))

    if arr.ndim != 2:
        return None
    if tuple(arr.shape) != tuple(expected_hw):
        return None

    return arr


def discover_candidate_keys(
    first_archive: Path,
    expected_hw: Tuple[int, int],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    with np.load(first_archive, allow_pickle=False) as npz:
        all_keys = list(npz.files)

        for key in all_keys:
            raw = safe_read_npz_numeric(npz, key)
            if raw is None:
                continue

            arr = squeeze_scene_array(raw, expected_hw)
            if arr is None:
                continue

            if not np.issubdtype(arr.dtype, np.floating):
                continue

            arr32 = arr.astype(np.float32, copy=False)

            if not np.isfinite(arr32).all():
                continue

            amin = float(arr32.min())
            amax = float(arr32.max())

            # Canonical final scene probability is bounded to [0, 1].
            if amin < -1e-5 or amax > 1.00001:
                continue

            candidates.append(
                {
                    "key": key,
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "min": amin,
                    "max": amax,
                    "mean": float(arr32.mean()),
                }
            )

    if not candidates:
        with np.load(first_archive, allow_pickle=False) as npz:
            all_keys = list(npz.files)

        raise RuntimeError(
            "No 2-D floating [0,1] full-scene candidate was found in "
            f"{first_archive.name}.\nAvailable NPZ keys: {all_keys}"
        )

    return candidates


def load_probability(
    archive_path: Path,
    key: str,
    expected_hw: Tuple[int, int],
) -> np.ndarray:
    with np.load(archive_path, allow_pickle=False) as npz:
        if key not in npz.files:
            raise KeyError(
                f"Key '{key}' missing from {archive_path.name}"
            )

        raw = safe_read_npz_numeric(npz, key)

        if raw is None:
            raise RuntimeError(
                f"Key '{key}' is not safely readable numeric data in "
                f"{archive_path.name}"
            )

    arr = squeeze_scene_array(raw, expected_hw)

    if arr is None:
        raise RuntimeError(
            f"{archive_path.name}:{key} does not have expected shape "
            f"{expected_hw}; raw shape={np.asarray(raw).shape}"
        )

    arr = arr.astype(np.float32, copy=False)

    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"Non-finite values in {archive_path.name}:{key}"
        )

    return np.clip(arr, 0.0, 1.0)


def preference_rank(key: str) -> Tuple[int, int, str]:
    lower = key.lower()

    for index, preferred in enumerate(PREFERRED_PROBABILITY_NAMES):
        if lower == preferred.lower():
            return 0, index, lower

    if "semantic" in lower and (
        "prob" in lower or "map" in lower
    ):
        return 1, 0, lower

    if "refined" in lower and (
        "prob" in lower or "map" in lower
    ):
        return 1, 1, lower

    if "prob" in lower:
        return 2, 0, lower

    if any(term in lower for term in LOW_PRIORITY_TERMS):
        return 4, 0, lower

    return 3, 0, lower


# ---------------------------------------------------------------------------
# Metrics / scientific verification
# ---------------------------------------------------------------------------

def confusion(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, int]:
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)

    return {
        "tp": int(np.logical_and(p, g).sum()),
        "fp": int(np.logical_and(p, ~g).sum()),
        "fn": int(np.logical_and(~p, g).sum()),
        "tn": int(np.logical_and(~p, ~g).sum()),
    }


def add_counts(
    total: Dict[str, int],
    current: Dict[str, int],
) -> None:
    for key in ("tp", "fp", "fn", "tn"):
        total[key] += int(current[key])


def metrics_from_counts(
    counts: Dict[str, int],
) -> Dict[str, float]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tn = int(counts["tn"])
    eps = 1e-12

    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": (2.0 * tp) / (2.0 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "oa": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def official_confusion(
    report: Dict[str, Any],
) -> Dict[str, int]:
    raw = report["evaluation"]["confusion"]

    return {
        "tp": int(raw["tp"]),
        "fp": int(raw["fp"]),
        "fn": int(raw["fn"]),
        "tn": int(raw["tn"]),
    }


def count_distance(
    candidate: Dict[str, int],
    official: Dict[str, int],
) -> int:
    return sum(
        abs(int(candidate[key]) - int(official[key]))
        for key in ("tp", "fp", "fn", "tn")
    )


def identify_probability_key(
    manifest_rows: List[Dict[str, Any]],
    gt_dir: Path,
    threshold: float,
    official_counts: Dict[str, int],
    diagnostic_path: Path,
) -> str:
    first_row = manifest_rows[0]
    first_archive = Path(first_row["archive_path"])
    expected_hw = (
        int(first_row["source_height"]),
        int(first_row["source_width"]),
    )

    candidates = discover_candidate_keys(
        first_archive,
        expected_hw,
    )

    print("[DISCOVERY] Full-scene [0,1] candidates:")
    for candidate in candidates:
        print(
            f"  - {candidate['key']} | "
            f"dtype={candidate['dtype']} | "
            f"shape={candidate['shape']} | "
            f"min={candidate['min']:.6f} | "
            f"max={candidate['max']:.6f} | "
            f"mean={candidate['mean']:.6f}"
        )

    diagnostics: List[Dict[str, Any]] = []
    exact_keys: List[str] = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        key = str(candidate["key"])
        total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        compatible = True
        failure = ""

        print(
            f"[DISCOVERY] Verifying {candidate_index}/{len(candidates)}: "
            f"{key}"
        )

        for scene_index, row in enumerate(manifest_rows, start=1):
            scene_id = str(row["scene_id"])
            archive_path = Path(row["archive_path"])
            expected_hw = (
                int(row["source_height"]),
                int(row["source_width"]),
            )

            try:
                prob = load_probability(
                    archive_path,
                    key,
                    expected_hw,
                )
            except Exception as exc:
                compatible = False
                failure = f"{type(exc).__name__}: {exc}"
                break

            gt = load_binary(
                find_scene_file(gt_dir, scene_id)
            )

            if gt.shape != prob.shape:
                compatible = False
                failure = (
                    f"Shape mismatch {scene_id}: "
                    f"GT={gt.shape}, prob={prob.shape}"
                )
                break

            pred = (prob >= threshold).astype(np.uint8)
            add_counts(total, confusion(pred, gt))

            if scene_index % 32 == 0 or scene_index == len(manifest_rows):
                print(
                    f"    {key}: {scene_index}/"
                    f"{len(manifest_rows)} scenes"
                )

        exact = compatible and total == official_counts
        distance = (
            count_distance(total, official_counts)
            if compatible
            else None
        )

        diagnostics.append(
            {
                "key": key,
                "first_archive_profile": candidate,
                "compatible_all_scenes": compatible,
                "failure": failure,
                "threshold": threshold,
                "confusion": total if compatible else None,
                "official_confusion": official_counts,
                "absolute_count_distance": distance,
                "exact_official_confusion_match": exact,
            }
        )

        if exact:
            exact_keys.append(key)
            print(
                f"[DISCOVERY][EXACT] {key} reproduces the official "
                "TEST confusion exactly."
            )
        elif compatible:
            print(
                f"[DISCOVERY][NO MATCH] {key}: "
                f"count_distance={distance:,}"
            )
        else:
            print(
                f"[DISCOVERY][INCOMPATIBLE] {key}: {failure}"
            )

    diagnostic_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "official_confusion": official_counts,
                "candidates": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not exact_keys:
        nearest = [
            row
            for row in diagnostics
            if row["compatible_all_scenes"]
            and row["absolute_count_distance"] is not None
        ]
        nearest.sort(
            key=lambda row: int(row["absolute_count_distance"])
        )

        raise RuntimeError(
            "No NPZ probability array reproduces the official TEST "
            f"confusion at frozen threshold={threshold}.\n"
            f"Nearest candidates: "
            f"{[(row['key'], row['absolute_count_distance']) for row in nearest[:5]]}\n"
            f"Diagnostic: {diagnostic_path}"
        )

    exact_keys.sort(key=preference_rank)
    selected = exact_keys[0]

    if len(exact_keys) > 1:
        print(
            "[DISCOVERY] Multiple exact candidates: "
            f"{exact_keys}. Selected '{selected}' by probability-name "
            "preference."
        )

    return selected


def verify_official_result(
    counts: Dict[str, int],
    metrics: Dict[str, float],
    report: Dict[str, Any],
) -> None:
    expected_counts = official_confusion(report)

    if counts != expected_counts:
        raise RuntimeError(
            "Exported masks do NOT reproduce the official confusion.\n"
            f"Calculated: {counts}\nOfficial:   {expected_counts}"
        )

    official_metrics = report["evaluation"]["metrics"]

    for key in ("iou", "f1", "precision", "recall", "oa"):
        local_value = float(metrics[key])
        official_value = float(official_metrics[key])

        if abs(local_value - official_value) > 1e-12:
            raise RuntimeError(
                f"Metric mismatch for {key}: "
                f"calculated={local_value}, official={official_value}"
            )


# ---------------------------------------------------------------------------
# Error visualization
# ---------------------------------------------------------------------------

def make_error_map(
    pred: np.ndarray,
    gt: np.ndarray,
) -> np.ndarray:
    """Black=TN, green=TP, red=FP, blue=FN."""
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)

    out = np.zeros((*p.shape, 3), dtype=np.uint8)
    out[p & g] = (0, 220, 0)       # TP
    out[p & ~g] = (240, 0, 0)      # FP
    out[~p & g] = (0, 80, 255)     # FN
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction_dir",
        "--prediction-dir",
        dest="prediction_dir",
        type=str,
        required=True,
        help="Canonical full-scene directory produced by predict_full_scenes.py",
    )
    parser.add_argument(
        "--data_root",
        "--data-root",
        dest="data_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Official evaluate_full_scenes TEST report JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fixed_threshold",
        "--fixed-threshold",
        dest="fixed_threshold",
        type=float,
        default=None,
        help=(
            "Optional explicit frozen threshold. If omitted, the official "
            "report threshold is used. A mismatch with the report is refused."
        ),
    )
    parser.add_argument(
        "--binary_overlay_alpha",
        "--binary-overlay-alpha",
        dest="binary_overlay_alpha",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--panel_title_bar_h",
        "--panel-title-bar-h",
        dest="panel_title_bar_h",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--panel_font_size",
        "--panel-font-size",
        dest="panel_font_size",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--top_k",
        "--top-k",
        dest="top_k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    prediction_dir = Path(args.prediction_dir).resolve()
    data_root = Path(args.data_root).resolve()
    report_path = Path(args.report).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_dir}"
        )
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data root not found: {data_root}"
        )
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Official report not found: {report_path}"
        )

    report = load_official_report(report_path)
    evaluation = report["evaluation"]

    report_split = str(evaluation.get("split", "")).strip().lower()
    if report_split and report_split != str(args.split).lower():
        raise RuntimeError(
            f"Split mismatch: CLI={args.split}, report={report_split}"
        )

    report_threshold = float(evaluation["threshold"])
    threshold = (
        report_threshold
        if args.fixed_threshold is None
        else float(args.fixed_threshold)
    )

    if abs(threshold - report_threshold) > 1e-12:
        raise RuntimeError(
            f"Threshold mismatch: CLI={threshold}, "
            f"official report={report_threshold}. "
            "Refusing non-canonical export."
        )

    # If a previous failed export created only empty directories, remove/rebuild.
    if out_dir.exists():
        existing_files = [
            p for p in out_dir.rglob("*") if p.is_file()
        ]
        if existing_files:
            if not args.overwrite:
                raise RuntimeError(
                    f"Output directory already contains files: {out_dir}\n"
                    "Use --overwrite only if you intentionally want to "
                    "replace this visualization export."
                )
            shutil.rmtree(out_dir)
        else:
            shutil.rmtree(out_dir)

    ensure_dir(out_dir)

    subdirs = [
        "npz",
        "panels",
        "preview_t1",
        "preview_t2",
        "prob",
        "temporal_delta",
        "gt_mask",
        "mask",
        "error_map",
        "overlay_pred",
        "overlay_gt",
        "best_iou_panels",
        "worst_iou_panels",
    ]

    for sub in subdirs:
        ensure_dir(out_dir / sub)

    manifest = load_prediction_manifest(prediction_dir)
    manifest_rows = canonical_manifest_rows(
        prediction_dir,
        manifest,
    )

    expected_scene_count = int(evaluation["scene_count"])

    if len(manifest_rows) != expected_scene_count:
        raise RuntimeError(
            f"Manifest scene count={len(manifest_rows)}, "
            f"official report scene count={expected_scene_count}"
        )

    split_root = find_split_root(data_root, args.split)
    dir_a = find_subdir(split_root, ("A", "a"))
    dir_b = find_subdir(split_root, ("B", "b"))
    dir_l = find_subdir(
        split_root,
        ("label", "Label", "LABEL", "labels", "Labels"),
    )

    expected_counts = official_confusion(report)
    discovery_json = out_dir / "probability_key_discovery.json"

    print("=" * 96)
    print("FINAL CANONICAL FULL-SCENE VISUAL EXPORT — HISTORICAL PANEL STYLE")
    print("=" * 96)
    print(f"Prediction dir      : {prediction_dir}")
    print(f"Output dir          : {out_dir}")
    print(f"Scenes              : {len(manifest_rows)}")
    print(f"Frozen threshold    : {threshold:.3f}")
    print("GPU inference       : NO")
    print("Threshold tuning    : NO")
    print("Legacy postprocess  : NO")
    print("Panel layout        : historical one-row horizontal style")
    print("=" * 96)

    selected_key = identify_probability_key(
        manifest_rows=manifest_rows,
        gt_dir=dir_l,
        threshold=threshold,
        official_counts=expected_counts,
        diagnostic_path=discovery_json,
    )

    print()
    print(
        f"[VERIFIED] Canonical full-scene probability key: "
        f"{selected_key}"
    )
    print()

    rows: List[Dict[str, Any]] = []
    global_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    for row in tqdm(
        manifest_rows,
        desc=f"Export {args.split}",
    ):
        patch_id = str(row["scene_id"])
        archive_path = Path(row["archive_path"])
        expected_hw = (
            int(row["source_height"]),
            int(row["source_width"]),
        )

        t1_full = load_rgb(
            find_scene_file(dir_a, patch_id)
        )
        t2_full = load_rgb(
            find_scene_file(dir_b, patch_id)
        )
        gt_full = load_binary(
            find_scene_file(dir_l, patch_id)
        )

        prob = load_probability(
            archive_path,
            selected_key,
            expected_hw,
        )

        if (
            t1_full.shape[:2] != prob.shape
            or t2_full.shape[:2] != prob.shape
            or gt_full.shape != prob.shape
        ):
            raise RuntimeError(
                f"{patch_id}: shape mismatch: "
                f"T1={t1_full.shape}, T2={t2_full.shape}, "
                f"GT={gt_full.shape}, prob={prob.shape}"
            )

        # This is the AUTHORITATIVE semantic prediction.
        final_mask = (prob >= threshold).astype(np.uint8)

        # Purely visual diagnostic, identical concept to old exporter.
        temporal_delta = compute_temporal_delta_support(
            t1_full,
            t2_full,
        )

        scene_counts = confusion(
            final_mask,
            gt_full,
        )
        scene_metrics = metrics_from_counts(
            scene_counts
        )
        add_counts(
            global_counts,
            scene_counts,
        )

        error_map = make_error_map(
            final_mask,
            gt_full,
        )

        overlay_pred = make_binary_overlay(
            t2_full,
            final_mask,
            alpha=float(args.binary_overlay_alpha),
            color=(255, 0, 0),
        )

        overlay_gt = make_binary_overlay(
            t2_full,
            gt_full,
            alpha=float(args.binary_overlay_alpha),
            color=(255, 0, 0),
        )

        # Individual images, using the familiar historical folder names.
        save_png_rgb(
            t1_full,
            out_dir / "preview_t1" / f"{patch_id}.png",
        )
        save_png_rgb(
            t2_full,
            out_dir / "preview_t2" / f"{patch_id}.png",
        )
        save_png_gray(
            prob,
            out_dir / "prob" / f"{patch_id}.png",
        )
        save_png_gray(
            temporal_delta,
            out_dir / "temporal_delta" / f"{patch_id}.png",
        )
        save_png_binary(
            gt_full,
            out_dir / "gt_mask" / f"{patch_id}.png",
        )
        save_png_binary(
            final_mask,
            out_dir / "mask" / f"{patch_id}.png",
        )
        save_png_rgb(
            error_map,
            out_dir / "error_map" / f"{patch_id}.png",
        )
        save_png_rgb(
            overlay_pred,
            out_dir / "overlay_pred" / f"{patch_id}.png",
        )
        save_png_rgb(
            overlay_gt,
            out_dir / "overlay_gt" / f"{patch_id}.png",
        )

        # Small archive for convenient later inspection.
        np.savez_compressed(
            out_dir / "npz" / f"{patch_id}.npz",
            patch_id=np.asarray(patch_id),
            gt_mask=gt_full.astype(np.uint8),
            prob_map=prob.astype(np.float32),
            temporal_delta=temporal_delta.astype(np.float32),
            pred_mask=final_mask.astype(np.uint8),
            error_map=error_map.astype(np.uint8),
            threshold=np.float32(threshold),
            tp=np.int64(scene_counts["tp"]),
            fp=np.int64(scene_counts["fp"]),
            fn=np.int64(scene_counts["fn"]),
            tn=np.int64(scene_counts["tn"]),
            iou=np.float64(scene_metrics["iou"]),
            f1=np.float64(scene_metrics["f1"]),
            precision=np.float64(scene_metrics["precision"]),
            recall=np.float64(scene_metrics["recall"]),
            oa=np.float64(scene_metrics["oa"]),
        )

        # -------------------------------------------------------------------
        # IMPORTANT: this intentionally reproduces the old ONE-LONG-ROW style.
        #
        # Old obsolete columns:
        #   Raw mask | Precision mask | Final mask
        #
        # are replaced by scientifically meaningful current columns:
        #   Ground truth | Official prediction | Error map
        #
        # because the canonical final protocol has NO semantic post-processing.
        # -------------------------------------------------------------------
        panel = make_panel(
            [
                ("T1 image", t1_full),
                ("T2 image", t2_full),
                (
                    "Model probability",
                    np.stack(
                        [float01_to_uint8_gray(prob)] * 3,
                        axis=-1,
                    ),
                ),
                (
                    "Temporal delta",
                    np.stack(
                        [float01_to_uint8_gray(temporal_delta)] * 3,
                        axis=-1,
                    ),
                ),
                (
                    "Ground truth",
                    np.stack(
                        [binary_mask_to_uint8(gt_full)] * 3,
                        axis=-1,
                    ),
                ),
                (
                    f"Prediction @ {threshold:.3f}",
                    np.stack(
                        [binary_mask_to_uint8(final_mask)] * 3,
                        axis=-1,
                    ),
                ),
                (
                    "Error map TP/FP/FN",
                    error_map,
                ),
                (
                    "Prediction overlay",
                    overlay_pred,
                ),
                (
                    "GT overlay",
                    overlay_gt,
                ),
            ],
            pad=8,
            title_bar_h=int(args.panel_title_bar_h),
            font_size=int(args.panel_font_size),
        )

        panel_path = (
            out_dir / "panels" / f"{patch_id}.png"
        )
        save_png_rgb(
            panel,
            panel_path,
        )

        rows.append(
            {
                "patch_id": patch_id,
                "source_tile_count": int(
                    row.get("tile_count", 0)
                ),
                "probability_key": selected_key,
                "threshold": threshold,
                "prob_mean": float(prob.mean()),
                "temporal_delta_mean": float(
                    temporal_delta.mean()
                ),
                "pred_ratio": float(final_mask.mean()),
                "gt_ratio": float(gt_full.mean()),
                "tp": int(scene_counts["tp"]),
                "fp": int(scene_counts["fp"]),
                "fn": int(scene_counts["fn"]),
                "tn": int(scene_counts["tn"]),
                "iou": float(scene_metrics["iou"]),
                "f1": float(scene_metrics["f1"]),
                "precision": float(
                    scene_metrics["precision"]
                ),
                "recall": float(
                    scene_metrics["recall"]
                ),
                "oa": float(scene_metrics["oa"]),
                "panel": str(panel_path),
            }
        )

    global_metrics = metrics_from_counts(
        global_counts
    )

    # Hard scientific gate AFTER all panels have been generated.
    verify_official_result(
        global_counts,
        global_metrics,
        report,
    )

    # Per-scene CSV in natural scene order.
    csv_path = out_dir / "export_summary.csv"

    fieldnames = [
        "patch_id",
        "source_tile_count",
        "probability_key",
        "threshold",
        "prob_mean",
        "temporal_delta_mean",
        "pred_ratio",
        "gt_ratio",
        "tp",
        "fp",
        "fn",
        "tn",
        "iou",
        "f1",
        "precision",
        "recall",
        "oa",
        "panel",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(
        rows,
        key=lambda row: float(row["iou"]),
        reverse=True,
    )

    top_k = max(
        0,
        min(int(args.top_k), len(ranked)),
    )

    for item in ranked[:top_k]:
        source = Path(item["panel"])
        shutil.copy2(
            source,
            out_dir / "best_iou_panels" / source.name,
        )

    for item in reversed(ranked[-top_k:]):
        source = Path(item["panel"])
        shutil.copy2(
            source,
            out_dir / "worst_iou_panels" / source.name,
        )

    ranked_csv = out_dir / "export_summary_ranked_by_iou.csv"

    with open(
        ranked_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(ranked)

    verification = {
        "purpose": (
            "canonical_full_scene_visualization_with_historical_horizontal_panel_style"
        ),
        "prediction_dir": str(prediction_dir),
        "prediction_manifest": str(
            prediction_dir / "prediction_manifest.json"
        ),
        "official_report": str(report_path),
        "output_dir": str(out_dir),
        "split": str(args.split),
        "scene_count": len(rows),
        "source_pixel_count": int(
            sum(global_counts.values())
        ),
        "selected_probability_key": selected_key,
        "threshold": threshold,
        "no_model_reinference": True,
        "no_threshold_search": True,
        "semantic_postprocessing": "none",
        "panel_layout": "historical_one_row_horizontal",
        "confusion": global_counts,
        "metrics": global_metrics,
        "official_confusion_exact_match": True,
        "official_metrics_exact_match": True,
        "error_map_legend": {
            "TN": "black",
            "TP": "green",
            "FP": "red",
            "FN": "blue",
        },
        "files": {
            "summary_csv": str(csv_path),
            "ranked_csv": str(ranked_csv),
            "probability_key_discovery": str(
                discovery_json
            ),
        },
    }

    verification_path = (
        out_dir / "visualization_verification.json"
    )
    verification_path.write_text(
        json.dumps(
            verification,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print(
        "EXPORT COMPLETE — HISTORICAL PANEL STYLE + "
        "EXACT OFFICIAL TEST MATCH VERIFIED"
    )
    print("=" * 96)
    print(f"Probability key : {selected_key}")
    print(f"Threshold       : {threshold:.3f}")
    print(f"Scenes          : {len(rows)}")
    print(f"TP              : {global_counts['tp']:,}")
    print(f"FP              : {global_counts['fp']:,}")
    print(f"FN              : {global_counts['fn']:,}")
    print(f"TN              : {global_counts['tn']:,}")
    print(f"IoU             : {global_metrics['iou'] * 100:.4f}%")
    print(f"F1              : {global_metrics['f1'] * 100:.4f}%")
    print(
        f"Precision       : "
        f"{global_metrics['precision'] * 100:.4f}%"
    )
    print(
        f"Recall          : "
        f"{global_metrics['recall'] * 100:.4f}%"
    )
    print(f"OA              : {global_metrics['oa'] * 100:.4f}%")
    print(f"Panels          : {out_dir / 'panels'}")
    print(f"Summary CSV     : {csv_path}")
    print(f"Verification    : {verification_path}")
    print("=" * 96)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
