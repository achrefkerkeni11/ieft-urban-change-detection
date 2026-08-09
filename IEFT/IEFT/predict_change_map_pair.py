import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from IEFT.modules.vilt_module import ViLTransformerSS


# =========================
# Utils
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def float01_to_uint8_gray(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.asarray(arr01, dtype=np.float32)
    arr01 = np.nan_to_num(arr01, nan=0.0, posinf=0.0, neginf=0.0)
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0).round().astype(np.uint8)


def binary_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return ((np.asarray(mask) > 0).astype(np.uint8) * 255)


def save_png_gray(arr01: np.ndarray, out_path: str):
    Image.fromarray(float01_to_uint8_gray(arr01), mode="L").save(out_path)


def save_png_binary(mask: np.ndarray, out_path: str):
    Image.fromarray(binary_mask_to_uint8(mask), mode="L").save(out_path)


def save_png_rgb(arr_uint8: np.ndarray, out_path: str):
    Image.fromarray(arr_uint8, mode="RGB").save(out_path)


def add_title_bar(img: np.ndarray, title: str, bar_h: int = 24) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w, c = img.shape
    canvas = np.full((h + bar_h, w, c), 255, dtype=np.uint8)
    canvas[bar_h:] = img
    pil = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.text((6, 4), title, fill=(0, 0, 0))
    return np.asarray(pil)


def make_panel(images_with_titles, pad: int = 6):
    prepared = [add_title_bar(img, title) for title, img in images_with_titles]
    max_h = max(p.shape[0] for p in prepared)
    total_w = sum(p.shape[1] for p in prepared) + pad * (len(prepared) + 1)
    panel = np.full((max_h + 2 * pad, total_w, 3), 255, dtype=np.uint8)
    x = pad
    for img in prepared:
        h, w, _ = img.shape
        y = pad + (max_h - h) // 2
        panel[y:y + h, x:x + w] = img
        x += w + pad
    return panel


def make_binary_overlay(base_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.float32)
    mask = (np.asarray(binary_mask) > 0).astype(np.float32)[..., None]
    red = np.zeros_like(base, dtype=np.float32)
    red[..., 0] = 255.0
    overlay = base * (1.0 - mask * alpha) + red * (mask * alpha)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(img, mode="RGB").resize((size, size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def resize_prob(prob: np.ndarray, size_hw) -> np.ndarray:
    h, w = size_hw
    prob_u8 = float01_to_uint8_gray(prob)
    return np.asarray(
        Image.fromarray(prob_u8, mode="L").resize((w, h), resample=Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def build_hann_window(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32)
    w = np.outer(w, w).astype(np.float32)
    w = w / max(w.max(), 1e-6)
    return (0.20 + 0.80 * w).astype(np.float32)


def tile_positions(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def _find_hparams_path_from_ckpt(ckpt_path: str) -> str:
    ckpt_path = os.path.abspath(ckpt_path)
    version_dir = os.path.dirname(os.path.dirname(ckpt_path))
    return os.path.join(version_dir, "hparams.yaml")


def _load_hparams_config(ckpt_path: str) -> dict:
    hparams_path = _find_hparams_path_from_ckpt(ckpt_path)
    if not os.path.isfile(hparams_path):
        return {}
    with open(hparams_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    if "config" in data and isinstance(data["config"], dict):
        return dict(data["config"])
    if "hyper_parameters" in data and isinstance(data["hyper_parameters"], dict):
        return dict(data["hyper_parameters"])
    return dict(data)


def build_cfg(args):
    defaults = {
        "exp_name": "predict_change_map_pair",
        "datasets": ["levir_cd"],
        "loss_names": {"itm": 0, "mlm": 0, "mpp": 0, "vqa": 0, "nlvr2": 0, "irtr": 0},
        "batch_size": 1,
        "per_gpu_batchsize": 1,
        "num_workers": 0,
        "num_gpus": 0,
        "num_nodes": 1,
        "precision": 16,
        "image_size": args.model_input_size,
        "model_input_size": args.model_input_size,
        "patch_size": 16,
        "hidden_size": 384,
        "num_heads": 6,
        "num_layers": 4,
        "drop_rate": 0.10,
        "vit": "vit_small_patch16_224",
        "tokenizer": "bert-base-uncased",
        "vocab_size": 30522,
        "max_text_len": 40,
        "load_path": args.ckpt_path,
        "levir_fixed_text": "building change detection",
        "change_global_gate_floor": 0.95,
        "dense_export_no_change_global_thr": 0.20,
        "dense_export_no_change_mean_thr": 0.030,
    }
    cfg = defaults.copy()
    cfg.update(_load_hparams_config(args.ckpt_path))
    cfg["load_path"] = args.ckpt_path
    cfg["image_size"] = args.model_input_size
    cfg["model_input_size"] = args.model_input_size
    return cfg


def make_batch_from_tiles(
    t1_tile_u8: np.ndarray,
    t2_tile_u8: np.ndarray,
    device,
    patch_id: str = "custom_pair",
    fallback_text: str = "building change detection",
):
    t1 = torch.from_numpy(t1_tile_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    t2 = torch.from_numpy(t2_tile_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()

    return {
        "image": [t2.to(device)],
        "image_t1": [t1.to(device)],
        "image_t2": [t2.to(device)],
        "text_ids": torch.zeros(1, 40, dtype=torch.long, device=device),
        "text_masks": torch.zeros(1, 40, dtype=torch.long, device=device),
        "text_labels": torch.full((1, 40), -100, dtype=torch.long, device=device),
        "text": [fallback_text],
        "main_text": [fallback_text],
        "file": [patch_id],
        "img_index": torch.tensor([0], dtype=torch.long, device=device),
        "image_index": torch.tensor([0], dtype=torch.long, device=device),
        "has_osm_text": torch.tensor([0], dtype=torch.long, device=device),
        "osm_struct": torch.zeros(1, 16, dtype=torch.float32, device=device),
        "osm_texts": [[]],
    }


# =========================
# Post-processing
# =========================

def _gradient_magnitude(gray01: np.ndarray) -> np.ndarray:
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = ndi.convolve(gray01.astype(np.float32), sx, mode="reflect")
    gy = ndi.convolve(gray01.astype(np.float32), sy, mode="reflect")
    return np.sqrt(gx * gx + gy * gy + 1e-6).astype(np.float32)


def robust_normalize(arr: np.ndarray, q_low: float = 0.02, q_high: float = 0.98, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    if abs(hi - lo) < eps:
        mn = float(arr.min())
        mx = float(arr.max())
        if abs(mx - mn) < eps:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - mn) / (mx - mn + eps), 0.0, 1.0)
    return np.clip((arr - lo) / (hi - lo + eps), 0.0, 1.0)


def compute_t2_building_support(t2_rgb_u8: np.ndarray) -> np.ndarray:
    rgb = t2_rgb_u8.astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    edge = _gradient_magnitude(gray)
    lap = np.abs(gray - ndi.uniform_filter(gray, size=5)).astype(np.float32)
    local_mean = ndi.uniform_filter(gray, size=11)
    local_var = ndi.uniform_filter((gray - local_mean) ** 2, size=11)
    local_std = np.sqrt(np.maximum(local_var, 1e-8)).astype(np.float32)

    bright = robust_normalize(gray)
    edge_n = robust_normalize(edge)
    lap_n = robust_normalize(lap)
    std_n = robust_normalize(local_std)
    diffuse_penalty = robust_normalize(np.maximum(0.0, std_n - 0.65 * edge_n))

    support = robust_normalize(
        0.42 * edge_n + 0.28 * lap_n + 0.18 * bright + 0.12 * std_n - 0.18 * diffuse_penalty
    )
    return support


def remove_small_components(mask: np.ndarray, min_region_size: int = 28) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask
    out = np.zeros_like(mask, dtype=np.uint8)
    sizes = ndi.sum(mask, labeled, index=np.arange(1, num + 1))
    for i, area in enumerate(sizes, start=1):
        if area >= min_region_size:
            out[labeled == i] = 1
    return out


def component_table(mask: np.ndarray, prob: np.ndarray, building_support: np.ndarray, border_band: int = 12):
    mask = (mask > 0).astype(np.uint8)
    labeled, num = ndi.label(mask)
    h, w = mask.shape
    rows = []
    for i in range(1, num + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area <= 0:
            continue
        ys, xs = np.where(comp)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bh = int(y1 - y0 + 1)
        bw = int(x1 - x0 + 1)
        bbox_area = float(bh * bw)
        fill_ratio = float(area) / max(1.0, bbox_area)
        elongation = float(max(bh, bw)) / max(1.0, float(min(bh, bw)))
        border_touch = 1.0 if (y0 < border_band or x0 < border_band or y1 >= h - border_band or x1 >= w - border_band) else 0.0
        pvals = prob[comp]
        bvals = building_support[comp]
        prob_mean = float(pvals.mean()) if pvals.size else 0.0
        prob_max = float(pvals.max()) if pvals.size else 0.0
        prob_q90 = float(np.quantile(pvals, 0.90)) if pvals.size else 0.0
        build_mean = float(bvals.mean()) if bvals.size else 0.0
        build_max = float(bvals.max()) if bvals.size else 0.0
        thin_penalty = max(0.0, min(1.0, (0.22 - fill_ratio) / 0.22)) * min(1.0, max(0.0, (elongation - 2.2) / 3.0))
        score = (
            0.28 * prob_mean
            + 0.24 * prob_max
            + 0.15 * prob_q90
            + 0.15 * build_mean
            + 0.08 * build_max
            + 0.08 * min(1.0, area / 320.0)
            + 0.05 * min(1.0, fill_ratio / 0.50)
            - 0.14 * thin_penalty
            - 0.06 * border_touch * max(0.0, 0.22 - build_mean) / 0.22
        )
        rows.append(
            {
                "id": i,
                "area": area,
                "prob_mean": prob_mean,
                "prob_max": prob_max,
                "prob_q90": prob_q90,
                "build_mean": build_mean,
                "build_max": build_max,
                "fill_ratio": fill_ratio,
                "elongation": elongation,
                "bbox_h": bh,
                "bbox_w": bw,
                "border_touch": border_touch,
                "score": float(score),
            }
        )
    return labeled, rows


def complete_buildings(prob: np.ndarray, t2_building_support: np.ndarray, fixed_threshold: float, keep_top_ratio: float, min_region_size: int) -> np.ndarray:
    prob = robust_normalize(prob)
    t2_building_support = robust_normalize(t2_building_support)

    q_seed = float(np.quantile(prob, 0.988))
    seed_thr = max(float(fixed_threshold) + 0.05, q_seed)
    seed = (prob >= seed_thr).astype(np.uint8)
    if seed.sum() == 0:
        seed_thr = float(np.quantile(prob, 0.994))
        seed = (prob >= seed_thr).astype(np.uint8)

    q_support = float(np.quantile(prob, 1.0 - float(np.clip(max(keep_top_ratio, 0.06), 1e-4, 0.95))))
    support_thr = max(0.18, min(0.38, q_support))
    support = (prob >= support_thr).astype(np.uint8)

    build_thr_hi = max(0.42, float(np.quantile(t2_building_support, 0.84)))
    build_thr_lo = max(0.28, float(np.quantile(t2_building_support, 0.72)))

    candidate = (t2_building_support >= build_thr_hi).astype(np.uint8)
    candidate = ndi.binary_opening(candidate.astype(bool), structure=np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.uint8)
    candidate = ndi.binary_closing(candidate.astype(bool), structure=np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.uint8)
    candidate = ndi.binary_fill_holes(candidate).astype(np.uint8)

    support_mask = ((support > 0) & (t2_building_support >= build_thr_lo)).astype(np.uint8)
    grown = ndi.binary_propagation(seed.astype(bool), mask=support_mask.astype(bool)).astype(np.uint8)
    grown = remove_small_components(grown, min_region_size=min_region_size)

    cand_labeled, _ = ndi.label(candidate > 0)
    labeled, num = ndi.label(grown > 0)
    if num == 0:
        return np.zeros_like(grown, dtype=np.uint8)

    out = np.zeros_like(grown, dtype=np.uint8)
    for i in range(1, num + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area < int(min_region_size):
            continue

        ys, xs = np.where(comp)
        if len(ys) == 0:
            continue
        y0 = max(0, int(ys.min()) - 3)
        y1 = min(comp.shape[0], int(ys.max()) + 4)
        x0 = max(0, int(xs.min()) - 3)
        x1 = min(comp.shape[1], int(xs.max()) + 4)

        comp_patch = comp[y0:y1, x0:x1]
        prob_patch = prob[y0:y1, x0:x1]
        build_patch = t2_building_support[y0:y1, x0:x1]

        final_comp = comp_patch.copy()
        overlap_ids = np.unique(cand_labeled[y0:y1, x0:x1][comp_patch])
        overlap_ids = overlap_ids[overlap_ids > 0]

        best_score = -1e9
        best_union = None
        for cid in overlap_ids:
            cand = cand_labeled[y0:y1, x0:x1] == int(cid)
            cand_area = float(cand.sum())
            if cand_area < float(min_region_size):
                continue
            overlap = float((cand & comp_patch).sum())
            if overlap <= 0:
                continue
            overlap_ratio = overlap / max(1.0, cand_area)
            union = cand | comp_patch
            union_area = float(union.sum())
            growth_ratio = union_area / max(1.0, float(comp_patch.sum()))
            extra = union & (~comp_patch)
            extra_area = float(extra.sum())
            if growth_ratio > 1.85:
                continue
            if overlap_ratio < 0.12 and overlap < 14:
                continue
            extra_prob_mean = float(prob_patch[extra].mean()) if extra_area > 0 else 0.0
            extra_build_mean = float(build_patch[extra].mean()) if extra_area > 0 else 0.0
            core_prob_mean = float(prob_patch[comp_patch].mean())
            if extra_area > 0 and extra_prob_mean < 0.10 and extra_build_mean < build_thr_hi:
                continue
            score = 2.0 * overlap_ratio + 0.9 * core_prob_mean + 0.8 * extra_prob_mean + 0.5 * extra_build_mean - 0.8 * max(0.0, growth_ratio - 1.0)
            if score > best_score:
                best_score = score
                best_union = union

        if best_union is not None:
            final_comp = best_union

        final_comp = ndi.binary_closing(final_comp, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
        final_comp = ndi.binary_fill_holes(final_comp)
        local_gate = (prob_patch >= max(0.13, support_thr - 0.03)) | (build_patch >= build_thr_lo)
        final_comp = final_comp & local_gate
        if int(final_comp.sum()) < int(min_region_size):
            continue
        out[y0:y1, x0:x1] |= final_comp.astype(np.uint8)

    out = remove_small_components(out, min_region_size=min_region_size)
    return out.astype(np.uint8)


def precision_refine_components(mask: np.ndarray, prob: np.ndarray, building_support: np.ndarray, min_region_size: int, gate_prob_floor: float, gate_build_floor: float) -> np.ndarray:
    labeled, rows = component_table(mask, prob, building_support)
    if not rows:
        return np.zeros_like(mask, dtype=np.uint8)

    out = np.zeros_like(mask, dtype=np.uint8)
    for row in rows:
        cid = int(row["id"])
        comp = labeled == cid
        ys, xs = np.where(comp)
        if len(ys) == 0:
            continue
        y0 = max(0, int(ys.min()) - 2)
        y1 = min(mask.shape[0], int(ys.max()) + 3)
        x0 = max(0, int(xs.min()) - 2)
        x1 = min(mask.shape[1], int(xs.max()) + 3)
        comp_patch = comp[y0:y1, x0:x1]
        p = prob[y0:y1, x0:x1]
        b = building_support[y0:y1, x0:x1]
        comp_p = p[comp_patch]
        comp_b = b[comp_patch]
        if comp_p.size == 0:
            continue

        adaptive_prob = max(gate_prob_floor, float(np.quantile(comp_p, 0.35)))
        adaptive_build = max(gate_build_floor, float(np.quantile(comp_b, 0.45)))

        keep_gate = (p >= adaptive_prob) | ((b >= adaptive_build) & (p >= max(0.10, adaptive_prob - 0.06)))
        refined = comp_patch & keep_gate
        refined = ndi.binary_opening(refined, structure=np.ones((2, 2), dtype=np.uint8), iterations=1)
        refined = ndi.binary_closing(refined, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
        refined = ndi.binary_fill_holes(refined)
        refined = refined & ((p >= max(0.09, adaptive_prob - 0.08)) | (b >= max(0.28, adaptive_build - 0.08)))

        if int(refined.sum()) < int(min_region_size):
            refined = comp_patch
        out[y0:y1, x0:x1] |= refined.astype(np.uint8)

    return remove_small_components(out.astype(np.uint8), min_region_size=min_region_size)


def precision_strict_filter(
    mask: np.ndarray,
    prob: np.ndarray,
    building_support: np.ndarray,
    min_region_size: int,
    strong_keep_area: int,
    component_score_thr: float,
    line_fill_ratio_thr: float,
    line_elongation_thr: float,
    line_build_max_thr: float,
    line_prob_mean_thr: float,
    max_components_keep: int,
) -> np.ndarray:
    labeled, rows = component_table(mask, prob, building_support)
    if not rows:
        return np.zeros_like(mask, dtype=np.uint8)

    rows_sorted = sorted(rows, key=lambda r: (float(r["score"]), float(r["prob_max"]), float(r["area"])), reverse=True)
    out = np.zeros_like(mask, dtype=np.uint8)
    kept = 0
    for row in rows_sorted:
        area = int(row["area"])
        score = float(row["score"])
        prob_mean = float(row["prob_mean"])
        prob_max = float(row["prob_max"])
        build_mean = float(row["build_mean"])
        build_max = float(row["build_max"])
        fill_ratio = float(row["fill_ratio"])
        elongation = float(row["elongation"])
        border_touch = float(row["border_touch"])

        line_like = (
            fill_ratio < line_fill_ratio_thr
            and elongation > line_elongation_thr
            and build_max < line_build_max_thr
            and prob_mean < line_prob_mean_thr
        )
        weak_border_line = (
            border_touch > 0.5
            and fill_ratio < max(0.12, line_fill_ratio_thr - 0.04)
            and elongation > max(2.8, line_elongation_thr - 0.8)
            and build_mean < 0.30
            and prob_mean < 0.20
        )
        tiny_weak = area < max(min_region_size * 3, 96) and score < max(component_score_thr + 0.01, 0.33)

        keep = False
        if area >= int(strong_keep_area) and (prob_mean >= 0.15 or build_mean >= 0.34):
            keep = True
        elif area >= max(112, min_region_size * 4) and score >= max(component_score_thr - 0.01, 0.30) and fill_ratio >= 0.15:
            keep = True
        elif score >= max(component_score_thr + 0.04, 0.36) and prob_max >= 0.36 and build_max >= 0.32:
            keep = True
        elif fill_ratio >= 0.26 and build_mean >= 0.36 and prob_mean >= 0.18 and area >= max(84, min_region_size * 3):
            keep = True
        elif prob_max >= 0.60 and prob_mean >= 0.20 and area >= max(56, min_region_size * 2):
            keep = True

        if line_like or weak_border_line or tiny_weak:
            keep = False

        if keep:
            out[labeled == int(row["id"])] = 1
            kept += 1
            if kept >= int(max_components_keep) and area < int(strong_keep_area):
                break

    return remove_small_components(out, min_region_size=min_region_size)


def veto_no_change_precision_strict(
    mask: np.ndarray,
    prob: np.ndarray,
    global_map: np.ndarray,
    building_support: np.ndarray,
    min_region_size: int,
    global_thr: float,
    prob_mean_thr: float,
    peak_thr: float,
    component_score_thr: float,
    strong_keep_area: int,
    max_components_nochange: int,
    dominant_component_ratio_thr: float,
) -> bool:
    pred_ratio = float(mask.mean())
    global_mean = float(global_map.mean())
    prob_mean = float(prob.mean())
    peak = float(np.quantile(prob, 0.997))

    labeled, rows = component_table(mask, prob, building_support)
    if not rows:
        return True

    areas = np.array([float(r["area"]) for r in rows], dtype=np.float32)
    scores = np.array([float(r["score"]) for r in rows], dtype=np.float32)
    fills = np.array([float(r["fill_ratio"]) for r in rows], dtype=np.float32)
    builds = np.array([float(r["build_mean"]) for r in rows], dtype=np.float32)
    probs = np.array([float(r["prob_mean"]) for r in rows], dtype=np.float32)
    borders = np.array([float(r["border_touch"]) for r in rows], dtype=np.float32)
    elongs = np.array([float(r["elongation"]) for r in rows], dtype=np.float32)

    comp_num = int(len(rows))
    total_area = float(areas.sum())
    best_area = float(areas.max())
    best_score = float(scores.max())
    dominant_ratio = best_area / max(1.0, total_area)
    mean_area = float(areas.mean())
    mean_fill = float(fills.mean())
    mean_build = float(builds.mean())
    mean_comp_prob = float(probs.mean())
    border_ratio = float(borders.mean())
    long_line_fraction = float(np.mean((elongs > 3.2) & (fills < 0.16))) if len(elongs) else 0.0

    strong_component_exists = any(
        (float(r["area"]) >= float(strong_keep_area) and (float(r["prob_mean"]) >= 0.15 or float(r["build_mean"]) >= 0.34))
        or (float(r["score"]) >= max(component_score_thr + 0.05, 0.37) and float(r["prob_max"]) >= 0.38 and float(r["build_max"]) >= 0.34)
        for r in rows
    )
    if strong_component_exists:
        return False

    if global_mean < global_thr and prob_mean < prob_mean_thr and peak < peak_thr:
        return True
    if pred_ratio < 0.018 and comp_num >= max_components_nochange and dominant_ratio < dominant_component_ratio_thr and mean_area < max(86.0, 2.9 * float(min_region_size)):
        return True
    if pred_ratio < 0.022 and best_area < max(150.0, 5.5 * float(min_region_size)) and best_score < max(component_score_thr + 0.02, 0.34):
        return True
    if border_ratio > 0.55 and mean_fill < 0.18 and mean_build < 0.30 and mean_comp_prob < 0.20:
        return True
    if long_line_fraction > 0.45 and best_area < max(240.0, 7.0 * float(min_region_size)):
        return True
    if dominant_ratio < dominant_component_ratio_thr and mean_fill < 0.17 and peak < max(peak_thr + 0.04, 0.28):
        return True

    return False


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--image_a", type=str, required=True, help="before image")
    parser.add_argument("--image_b", type=str, required=True, help="after image")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--patch_id", type=str, default="custom_pair")
    parser.add_argument("--model_input_size", type=int, default=256)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_stride", type=int, default=128)
    parser.add_argument("--fixed_threshold", type=float, default=0.50)
    parser.add_argument("--keep_top_ratio", type=float, default=0.05)
    parser.add_argument("--min_region_size", type=int, default=28)
    parser.add_argument("--border_band", type=int, default=24)
    parser.add_argument("--binary_overlay_alpha", type=float, default=0.55)
    parser.add_argument("--nochange_global_thr", type=float, default=0.21)
    parser.add_argument("--nochange_prob_mean_thr", type=float, default=0.028)
    parser.add_argument("--nochange_peak_thr", type=float, default=0.24)
    parser.add_argument("--strong_keep_area", type=int, default=220)
    parser.add_argument("--component_score_thr", type=float, default=0.31)
    parser.add_argument("--precision_gate_prob", type=float, default=0.20)
    parser.add_argument("--precision_gate_build", type=float, default=0.38)
    parser.add_argument("--max_components_nochange", type=int, default=4)
    parser.add_argument("--max_components_keep", type=int, default=5)
    parser.add_argument("--line_fill_ratio_thr", type=float, default=0.16)
    parser.add_argument("--line_elongation_thr", type=float, default=3.6)
    parser.add_argument("--line_build_max_thr", type=float, default=0.40)
    parser.add_argument("--line_prob_mean_thr", type=float, default=0.19)
    parser.add_argument("--dominant_component_ratio_thr", type=float, default=0.34)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(str(out_dir))

    cfg = build_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViLTransformerSS(cfg)
    model.eval().to(device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    t1_full = np.asarray(Image.open(args.image_a).convert("RGB"), dtype=np.uint8)
    t2_full = np.asarray(Image.open(args.image_b).convert("RGB"), dtype=np.uint8)

    if t1_full.shape[:2] != t2_full.shape[:2]:
        raise ValueError(f"A and B must have same size, got {t1_full.shape[:2]} vs {t2_full.shape[:2]}")

    h, w = t1_full.shape[:2]
    tile = int(args.tile_size)
    stride = int(args.tile_stride)

    ys = tile_positions(h, tile, stride)
    xs = tile_positions(w, tile, stride)
    hann = build_hann_window(tile)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    gp_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                t1_tile = t1_full[y:y + tile, x:x + tile]
                t2_tile = t2_full[y:y + tile, x:x + tile]

                if t1_tile.shape[0] != tile or t1_tile.shape[1] != tile:
                    pad_h = tile - t1_tile.shape[0]
                    pad_w = tile - t1_tile.shape[1]
                    t1_tile = np.pad(t1_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                    t2_tile = np.pad(t2_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

                in_t1 = resize_rgb(t1_tile, args.model_input_size) if tile != args.model_input_size else t1_tile
                in_t2 = resize_rgb(t2_tile, args.model_input_size) if tile != args.model_input_size else t2_tile

                batch = make_batch_from_tiles(
                    in_t1,
                    in_t2,
                    device,
                    patch_id=args.patch_id,
                    fallback_text=str(cfg.get("levir_fixed_text", "building change detection")),
                )

                infer = model.infer(batch, mask_text=False, mask_image=False)
                tile_prob = infer["change_refined_map_up"][0].detach().cpu().numpy().astype(np.float32)
                global_prob = float(infer["change_probs_global"][0, 0].detach().cpu().item())

                if tile_prob.shape != (tile, tile):
                    tile_prob = resize_prob(tile_prob, (tile, tile))

                yy = min(tile, h - y)
                xx = min(tile, w - x)
                prob_sum[y:y + yy, x:x + xx] += tile_prob[:yy, :xx] * hann[:yy, :xx]
                gp_sum[y:y + yy, x:x + xx] += global_prob * hann[:yy, :xx]
                weight_sum[y:y + yy, x:x + xx] += hann[:yy, :xx]

    prob = prob_sum / np.maximum(weight_sum, 1e-6)
    global_map = gp_sum / np.maximum(weight_sum, 1e-6)
    building_support = compute_t2_building_support(t2_full)

    raw_mask = complete_buildings(prob, building_support, args.fixed_threshold, args.keep_top_ratio, args.min_region_size)
    precision_mask = precision_refine_components(
        raw_mask,
        prob,
        building_support,
        args.min_region_size,
        args.precision_gate_prob,
        args.precision_gate_build,
    )
    final_mask = precision_strict_filter(
        precision_mask,
        prob,
        building_support,
        args.min_region_size,
        args.strong_keep_area,
        args.component_score_thr,
        args.line_fill_ratio_thr,
        args.line_elongation_thr,
        args.line_build_max_thr,
        args.line_prob_mean_thr,
        args.max_components_keep,
    )

    if veto_no_change_precision_strict(
        final_mask,
        prob,
        global_map,
        building_support,
        args.min_region_size,
        args.nochange_global_thr,
        args.nochange_prob_mean_thr,
        args.nochange_peak_thr,
        args.component_score_thr,
        args.strong_keep_area,
        args.max_components_nochange,
        args.dominant_component_ratio_thr,
    ):
        final_mask[:] = 0

    overlay_pred = make_binary_overlay(t2_full, final_mask, alpha=args.binary_overlay_alpha)
    overlay_raw = make_binary_overlay(t2_full, raw_mask, alpha=args.binary_overlay_alpha)
    overlay_precision = make_binary_overlay(t2_full, precision_mask, alpha=args.binary_overlay_alpha)

    panel = make_panel([
        ("A / Before", t1_full),
        ("B / After", t2_full),
        ("Prob", float01_to_uint8_gray(prob)),
        ("RawMask", binary_mask_to_uint8(raw_mask)),
        ("PrecisionMask", binary_mask_to_uint8(precision_mask)),
        ("FinalMask", binary_mask_to_uint8(final_mask)),
        ("OverlayFinal", overlay_pred),
    ])

    save_png_rgb(t1_full, str(out_dir / "A.png"))
    save_png_rgb(t2_full, str(out_dir / "B.png"))
    save_png_gray(prob, str(out_dir / "prob.png"))
    save_png_gray(global_map, str(out_dir / "global_prob_map.png"))
    save_png_gray(building_support, str(out_dir / "building_support.png"))
    save_png_binary(raw_mask, str(out_dir / "mask_raw.png"))
    save_png_binary(precision_mask, str(out_dir / "mask_precision.png"))
    save_png_binary(final_mask, str(out_dir / "mask_final.png"))
    save_png_rgb(overlay_raw, str(out_dir / "overlay_raw.png"))
    save_png_rgb(overlay_precision, str(out_dir / "overlay_precision.png"))
    save_png_rgb(overlay_pred, str(out_dir / "overlay_pred.png"))
    save_png_rgb(panel, str(out_dir / "panel.png"))

    np.savez_compressed(
        str(out_dir / "outputs.npz"),
        patch_id=np.array(args.patch_id),
        dense_prob=np.array(prob, dtype=np.float32),
        global_prob_map=np.array(global_map, dtype=np.float32),
        building_support=np.array(building_support, dtype=np.float32),
        pred_mask_raw=np.array(raw_mask, dtype=np.uint8),
        pred_mask_precision=np.array(precision_mask, dtype=np.uint8),
        pred_mask_final=np.array(final_mask, dtype=np.uint8),
    )

    print("[OK] Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()