
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from tqdm import tqdm

from IEFT.modules.vilt_module import ViLTransformerSS
from IEFT.datasets.levir_cd_dataset import LEVIROSMHelper


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


def make_binary_overlay(base_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.float32)
    mask = (np.asarray(binary_mask) > 0).astype(np.float32)[..., None]
    red = np.zeros_like(base, dtype=np.float32)
    red[..., 0] = 255.0
    overlay = base * (1.0 - mask * alpha) + red * (mask * alpha)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _load_font(font_size: int):
    font_candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for fp in font_candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def add_title_bar(img: np.ndarray, title: str, bar_h: int = 40, font_size: int = 18) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w, c = img.shape
    canvas = np.full((h + bar_h, w, c), 255, dtype=np.uint8)
    canvas[bar_h:] = img
    pil = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size)
    draw.text((8, max(4, (bar_h - font_size) // 2)), title, fill=(0, 0, 0), font=font)
    return np.asarray(pil)


def make_panel(images_with_titles, pad: int = 8, title_bar_h: int = 40, font_size: int = 18):
    prepared = [add_title_bar(img, title, bar_h=title_bar_h, font_size=font_size) for title, img in images_with_titles]
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


def _is_image_file(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def list_split_triplets(data_root: str, split: str, a_dir="A", b_dir="B", l_dir="label"):
    root = Path(data_root) / split
    dir_a = root / a_dir
    dir_b = root / b_dir
    dir_l = root / l_dir
    if not dir_a.is_dir() or not dir_b.is_dir() or not dir_l.is_dir():
        raise FileNotFoundError(f"Missing split folders under {root}")
    a_files = {Path(n).stem: dir_a / n for n in os.listdir(dir_a) if _is_image_file(n)}
    b_files = {Path(n).stem: dir_b / n for n in os.listdir(dir_b) if _is_image_file(n)}
    l_files = {Path(n).stem: dir_l / n for n in os.listdir(dir_l) if _is_image_file(n)}
    common = sorted(set(a_files) & set(b_files) & set(l_files))
    return [(stem, str(a_files[stem]), str(b_files[stem]), str(l_files[stem])) for stem in common]


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
        "exp_name": "export_change_maps_safe_rural_v3",
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
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 4,
        "drop_rate": 0.10,
        "vit": "vit_base_patch16_224",
        "vit_pretrained": False,
        "tokenizer": "bert-base-uncased",
        "vocab_size": 30522,
        "max_text_len": 40,
        "data_root": args.data_root,
        "load_path": args.ckpt_path,
        "levir_label_dirname": "label",
        "levir_image_a_dirname": "A",
        "levir_image_b_dirname": "B",
        "levir_fixed_text": "building change detection",
        "change_global_gate_floor": 0.95,
        "dense_export_no_change_global_thr": 0.20,
        "dense_export_no_change_mean_thr": 0.030,
        "levir_use_osm": False,
        "levir_osm_texts_json": "",
        "levir_osm_text_mode": "concat",
        "levir_osm_max_phrases": 3,
        "levir_osm_text_key": "text_v21",
        "levir_osm_fallback_text": "no_osm_context",
        "levir_osm_compose_mode": "signature_compact",
        "levir_osm_word_budget": 32,
        "levir_osm_joiner": " ; ",
        "levir_osm_include_source_text": False,
        "export_panel_title_bar_h": 40,
        "export_panel_font_size": 18,
        "use_semantic_multiscale": True,
        "ms_change_num_levels": 4,
        "ms_change_level_indices": [2, 5, 8, 11],
        "ms_change_decoder_dim": 128,
        "ms_change_dropout": 0.08,
    }
    cfg = defaults.copy()
    cfg.update(_load_hparams_config(args.ckpt_path))
    cfg["data_root"] = args.data_root
    cfg["load_path"] = args.ckpt_path
    cfg["image_size"] = args.model_input_size
    cfg["model_input_size"] = args.model_input_size
    if str(args.levir_osm_texts_json).strip():
        cfg["levir_osm_texts_json"] = str(args.levir_osm_texts_json).strip()
        cfg["levir_use_osm"] = True
    if bool(args.disable_osm):
        cfg["levir_use_osm"] = False
    return cfg


def resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img, mode="RGB").resize((size, size), resample=Image.BILINEAR), dtype=np.uint8)


def resize_prob(prob: np.ndarray, size_hw) -> np.ndarray:
    h, w = size_hw
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.squeeze(prob)
    if prob.ndim != 2:
        raise ValueError(f"resize_prob expected 2D array after squeeze, got shape={prob.shape}")
    prob_u8 = float01_to_uint8_gray(prob)
    return np.asarray(
        Image.fromarray(prob_u8, mode="L").resize((w, h), resample=Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def _resolve_osm(osm_helper: LEVIROSMHelper, patch_id: str, fallback_text: str):
    if osm_helper is None:
        return None
    return osm_helper.resolve([patch_id], fallback_text=fallback_text)


def make_batch_from_tiles(
    t1_tile_u8: np.ndarray,
    t2_tile_u8: np.ndarray,
    device,
    patch_id: str = "tile",
    osm_helper: LEVIROSMHelper = None,
    osm_resolved=None,
    fallback_text: str = "building change detection",
):
    t1 = torch.from_numpy(t1_tile_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    t2 = torch.from_numpy(t2_tile_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()

    if osm_resolved is None and osm_helper is not None:
        osm_resolved = _resolve_osm(osm_helper, patch_id, fallback_text)

    if osm_resolved is not None:
        main_text = str(osm_resolved["main_text"])
        has_osm_text = int(osm_resolved["has_osm_text"])
        osm_struct = osm_resolved["osm_struct"].clone().float()
        osm_texts = list(osm_resolved["osm_texts"])
    else:
        main_text = fallback_text
        has_osm_text = 0
        osm_struct = torch.zeros(16, dtype=torch.float32)
        osm_texts = []

    return {
        "image": [t2.to(device)],
        "image_t1": [t1.to(device)],
        "image_t2": [t2.to(device)],
        "text_ids": torch.zeros(1, 40, dtype=torch.long, device=device),
        "text_masks": torch.zeros(1, 40, dtype=torch.long, device=device),
        "text_labels": torch.full((1, 40), -100, dtype=torch.long, device=device),
        "text": [main_text],
        "main_text": [main_text],
        "file": [patch_id],
        "img_index": torch.tensor([0], dtype=torch.long, device=device),
        "image_index": torch.tensor([0], dtype=torch.long, device=device),
        "has_osm_text": torch.tensor([has_osm_text], dtype=torch.long, device=device),
        "osm_struct": osm_struct.unsqueeze(0).to(device),
        "osm_texts": [osm_texts],
    }


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


def compute_temporal_delta_support(t1_rgb_u8: np.ndarray, t2_rgb_u8: np.ndarray) -> np.ndarray:
    t1 = t1_rgb_u8.astype(np.float32) / 255.0
    t2 = t2_rgb_u8.astype(np.float32) / 255.0
    diff_rgb = np.abs(t2 - t1).mean(axis=2)
    g1 = t1.mean(axis=2)
    g2 = t2.mean(axis=2)
    e1 = _gradient_magnitude(g1)
    e2 = _gradient_magnitude(g2)
    edge_diff = np.abs(e2 - e1)
    delta = robust_normalize(0.68 * robust_normalize(diff_rgb) + 0.32 * robust_normalize(edge_diff))
    return delta


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


def infer_osm_group_boost(osm_resolved) -> float:
    if osm_resolved is None:
        return 0.0
    txt = " ".join([str(osm_resolved.get("main_text", ""))] + list(osm_resolved.get("osm_texts", []))).lower()
    boost = 0.0
    for kw, val in [
        ("residential", 0.18),
        ("industrial", 0.16),
        ("commercial", 0.12),
        ("built-up", 0.12),
        ("building", 0.14),
        ("transport", 0.10),
        ("road", 0.10),
        ("neighborhood", 0.10),
        ("urban", 0.12),
    ]:
        if kw in txt:
            boost += val
    if "water-related" in txt or "open area" in txt or "sparse undeveloped" in txt:
        boost -= 0.08
    return float(np.clip(boost, 0.0, 0.42))


def complete_buildings_osm_group(
    prob: np.ndarray,
    t2_building_support: np.ndarray,
    temporal_delta: np.ndarray,
    fixed_threshold: float,
    keep_top_ratio: float,
    min_region_size: int,
    osm_group_boost: float,
) -> np.ndarray:
    prob = robust_normalize(prob)
    t2_building_support = robust_normalize(t2_building_support)
    temporal_delta = robust_normalize(temporal_delta)

    q_seed = float(np.quantile(prob, 0.988))
    seed_thr = max(float(fixed_threshold) + 0.05 - 0.02 * osm_group_boost, q_seed - 0.02 * osm_group_boost)
    seed = (prob >= seed_thr).astype(np.uint8)
    if seed.sum() == 0:
        seed_thr = float(np.quantile(prob, 0.994))
        seed = (prob >= seed_thr).astype(np.uint8)

    q_support = float(np.quantile(prob, 1.0 - float(np.clip(max(keep_top_ratio, 0.06), 1e-4, 0.95))))
    support_thr = max(0.14, min(0.36, q_support - 0.05 * osm_group_boost))
    support = (prob >= support_thr).astype(np.uint8)

    build_thr_hi = max(0.34, float(np.quantile(t2_building_support, 0.84)) - 0.08 * osm_group_boost)
    build_thr_lo = max(0.22, float(np.quantile(t2_building_support, 0.72)) - 0.08 * osm_group_boost)
    delta_thr_lo = max(0.10, float(np.quantile(temporal_delta, 0.58)) - 0.02)

    candidate = (t2_building_support >= build_thr_hi).astype(np.uint8)
    candidate = ndi.binary_opening(candidate.astype(bool), structure=np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.uint8)
    candidate = ndi.binary_closing(candidate.astype(bool), structure=np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.uint8)
    candidate = ndi.binary_fill_holes(candidate).astype(np.uint8)

    support_mask = (
        ((support > 0) & (t2_building_support >= build_thr_lo) & (temporal_delta >= delta_thr_lo))
        | ((prob >= max(support_thr + 0.03, 0.18)) & (t2_building_support >= max(0.18, build_thr_lo - 0.05)))
    ).astype(np.uint8)

    grown = ndi.binary_propagation(seed.astype(bool), mask=support_mask.astype(bool)).astype(np.uint8)
    grown = remove_small_components(grown, min_region_size=min_region_size)

    cand_labeled, _ = ndi.label(candidate > 0)
    labeled, num = ndi.label(grown > 0)
    if num == 0:
        return np.zeros_like(grown, dtype=np.uint8)

    out = np.zeros_like(grown, dtype=np.uint8)
    max_growth_ratio = 1.75 + 0.75 * osm_group_boost
    min_overlap_ratio = max(0.07, 0.12 - 0.06 * osm_group_boost)

    for i in range(1, num + 1):
        comp = labeled == i
        area = int(comp.sum())
        if area < int(min_region_size):
            continue

        ys, xs = np.where(comp)
        if len(ys) == 0:
            continue
        y0 = max(0, int(ys.min()) - 4)
        y1 = min(comp.shape[0], int(ys.max()) + 5)
        x0 = max(0, int(xs.min()) - 4)
        x1 = min(comp.shape[1], int(xs.max()) + 5)

        comp_patch = comp[y0:y1, x0:x1]
        prob_patch = prob[y0:y1, x0:x1]
        build_patch = t2_building_support[y0:y1, x0:x1]
        delta_patch = temporal_delta[y0:y1, x0:x1]

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
            if growth_ratio > max_growth_ratio:
                continue
            if overlap_ratio < min_overlap_ratio and overlap < 10:
                continue
            extra_prob_mean = float(prob_patch[extra].mean()) if extra_area > 0 else 0.0
            extra_build_mean = float(build_patch[extra].mean()) if extra_area > 0 else 0.0
            extra_delta_mean = float(delta_patch[extra].mean()) if extra_area > 0 else 0.0
            core_prob_mean = float(prob_patch[comp_patch].mean())
            if extra_area > 0 and extra_prob_mean < (0.09 - 0.02 * osm_group_boost) and extra_build_mean < (build_thr_hi - 0.05) and extra_delta_mean < (delta_thr_lo - 0.02):
                continue
            score = (
                2.0 * overlap_ratio
                + 0.95 * core_prob_mean
                + 0.75 * extra_prob_mean
                + 0.55 * extra_build_mean
                + 0.45 * extra_delta_mean
                - 0.65 * max(0.0, growth_ratio - 1.0)
            )
            if score > best_score:
                best_score = score
                best_union = union

        if best_union is not None:
            final_comp = best_union

        final_comp = ndi.binary_closing(final_comp, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
        final_comp = ndi.binary_fill_holes(final_comp)
        local_gate = (
            (prob_patch >= max(0.11, support_thr - 0.05))
            | ((build_patch >= build_thr_lo) & (delta_patch >= max(0.08, delta_thr_lo - 0.03)))
        )
        final_comp = final_comp & local_gate
        if int(final_comp.sum()) < int(min_region_size):
            continue
        out[y0:y1, x0:x1] |= final_comp.astype(np.uint8)

    out = remove_small_components(out, min_region_size=min_region_size)
    return out.astype(np.uint8)


def connect_nearby_urban_components(mask: np.ndarray, prob: np.ndarray, building_support: np.ndarray, temporal_delta: np.ndarray, osm_group_boost: float, min_region_size: int) -> np.ndarray:
    if osm_group_boost <= 0.04:
        return mask.astype(np.uint8)

    labeled, rows = component_table(mask, prob, building_support)
    if len(rows) < 2:
        return mask.astype(np.uint8)

    max_gap = int(2 + round(3 * osm_group_boost))
    build_gate = max(0.24, 0.34 - 0.08 * osm_group_boost)
    prob_gate = max(0.10, 0.16 - 0.04 * osm_group_boost)
    delta_gate = max(0.08, float(np.quantile(temporal_delta, 0.58)) - 0.02)

    out = mask.astype(np.uint8).copy()
    rows_sorted = sorted(rows, key=lambda r: (r["area"], r["score"]), reverse=True)
    keep_ids = [int(r["id"]) for r in rows_sorted[: max(4, min(10, len(rows_sorted)))]]

    for i, cid_a in enumerate(keep_ids):
        comp_a = labeled == cid_a
        dil_a = ndi.binary_dilation(comp_a, iterations=max_gap)
        for cid_b in keep_ids[i + 1:]:
            comp_b = labeled == cid_b
            if not np.any(dil_a & comp_b):
                continue
            union = comp_a | comp_b
            bridge = ndi.binary_closing(union, structure=np.ones((3 + max_gap, 3 + max_gap), dtype=np.uint8), iterations=1)
            extra = bridge & (~union)
            if extra.sum() == 0:
                out[bridge] = 1
                continue
            extra_prob = float(prob[extra].mean()) if extra.sum() else 0.0
            extra_build = float(building_support[extra].mean()) if extra.sum() else 0.0
            extra_delta = float(temporal_delta[extra].mean()) if extra.sum() else 0.0
            if extra_delta >= delta_gate and (extra_build >= build_gate or (extra_build >= build_gate - 0.05 and extra_prob >= prob_gate)):
                out[bridge] = 1

    out = ndi.binary_closing(out.astype(bool), structure=np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.uint8)
    out = remove_small_components(out, min_region_size=min_region_size)
    return out


def precision_refine_components(mask: np.ndarray, prob: np.ndarray, building_support: np.ndarray, temporal_delta: np.ndarray, min_region_size: int, gate_prob_floor: float, gate_build_floor: float) -> np.ndarray:
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
        d = temporal_delta[y0:y1, x0:x1]
        comp_p = p[comp_patch]
        comp_b = b[comp_patch]
        if comp_p.size == 0:
            continue

        adaptive_prob = max(gate_prob_floor, float(np.quantile(comp_p, 0.35)))
        adaptive_build = max(gate_build_floor, float(np.quantile(comp_b, 0.45)))
        delta_floor = max(0.08, float(np.quantile(d[comp_patch], 0.30)) if int(comp_patch.sum()) > 0 else 0.08)

        keep_gate = (
            (p >= adaptive_prob)
            | ((b >= adaptive_build) & (d >= delta_floor))
            | ((p >= max(0.10, adaptive_prob - 0.06)) & (d >= max(0.06, delta_floor - 0.03)))
        )
        refined = comp_patch & keep_gate
        refined = ndi.binary_opening(refined, structure=np.ones((2, 2), dtype=np.uint8), iterations=1)
        refined = ndi.binary_closing(refined, structure=np.ones((3, 3), dtype=np.uint8), iterations=1)
        refined = ndi.binary_fill_holes(refined)
        refined = refined & (
            (p >= max(0.09, adaptive_prob - 0.08))
            | ((b >= max(0.28, adaptive_build - 0.08)) & (d >= max(0.06, delta_floor - 0.03)))
        )

        if int(refined.sum()) < int(min_region_size):
            refined = comp_patch
        out[y0:y1, x0:x1] |= refined.astype(np.uint8)

    return remove_small_components(out.astype(np.uint8), min_region_size=min_region_size)


def patch_level_false_positive_veto(mask: np.ndarray, prob: np.ndarray, global_map: np.ndarray, building_support: np.ndarray, temporal_delta: np.ndarray) -> bool:
    if int(mask.sum()) == 0:
        return False

    pred_ratio = float(mask.mean())
    prob_mean = float(prob.mean())
    prob_peak = float(np.quantile(prob, 0.997))
    global_mean = float(global_map.mean())
    pred_build_mean = float(building_support[mask > 0].mean()) if np.any(mask > 0) else 0.0
    pred_delta_mean = float(temporal_delta[mask > 0].mean()) if np.any(mask > 0) else 0.0

    labeled, num = ndi.label(mask > 0)
    if num == 0:
        return False

    sizes = np.asarray(ndi.sum((mask > 0).astype(np.uint8), labeled, index=np.arange(1, num + 1)), dtype=np.float32)
    max_area = float(sizes.max()) if sizes.size else 0.0
    mean_area = float(sizes.mean()) if sizes.size else 0.0

    border = np.zeros_like(mask, dtype=np.uint8)
    b = 12
    border[:b, :] = 1
    border[-b:, :] = 1
    border[:, :b] = 1
    border[:, -b:] = 1
    pred_border_ratio = float(((mask > 0) & (border > 0)).sum()) / max(1.0, float(border.sum()))

    compact_confident = (
        max_area >= 1800.0
        and pred_ratio >= 0.004
        and pred_build_mean >= 0.34
        and (pred_delta_mean >= 0.34 or prob_peak >= 0.80)
    )
    widespread_structured = (
        pred_ratio >= 0.020
        and mean_area >= 400.0
        and pred_build_mean >= 0.40
        and pred_delta_mean >= 0.40
    )

    suspicious_sparse_structured = (
        pred_ratio < 0.018
        and num >= 6
        and max_area < 3200.0
        and prob_mean < 0.010
        and global_mean < 0.55
        and pred_border_ratio < 0.04
        and pred_build_mean >= 0.62
        and pred_delta_mean >= 0.58
    )

    suspicious_compact_wrongloc = (
        pred_ratio < 0.012
        and num <= 12
        and max_area < 1600.0
        and global_mean >= 0.65
        and prob_mean < 0.020
        and prob_peak >= 0.70
        and pred_build_mean < 0.50
        and pred_delta_mean < 0.55
    )

    suspicious_weak_global = (
        pred_ratio < 0.010
        and global_mean < 0.20
        and prob_peak < 0.12
        and pred_delta_mean < 0.22
    )

    if compact_confident or widespread_structured:
        return False

    return bool(suspicious_sparse_structured or suspicious_compact_wrongloc or suspicious_weak_global)


def precision_strict_filter(
    mask: np.ndarray,
    prob: np.ndarray,
    building_support: np.ndarray,
    temporal_delta: np.ndarray,
    global_map: np.ndarray,
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
        cid = int(row["id"])
        area = int(row["area"])
        score = float(row["score"])
        prob_mean = float(row["prob_mean"])
        prob_max = float(row["prob_max"])
        prob_q90 = float(row["prob_q90"])
        build_mean = float(row["build_mean"])
        build_max = float(row["build_max"])
        fill_ratio = float(row["fill_ratio"])
        elongation = float(row["elongation"])
        border_touch = float(row["border_touch"])
        comp = labeled == cid
        delta_mean = float(temporal_delta[comp].mean()) if int(comp.sum()) > 0 else 0.0
        global_local_mean = float(global_map[comp].mean()) if int(comp.sum()) > 0 else 0.0

        line_like = (
            fill_ratio < line_fill_ratio_thr
            and elongation > line_elongation_thr
            and build_max < line_build_max_thr
            and prob_mean < line_prob_mean_thr
            and delta_mean < 0.14
        )
        weak_border_line = (
            border_touch > 0.5
            and fill_ratio < max(0.12, line_fill_ratio_thr - 0.04)
            and elongation > max(2.8, line_elongation_thr - 0.8)
            and build_mean < 0.30
            and prob_mean < 0.20
            and delta_mean < 0.14
        )
        tiny_weak = area < max(min_region_size * 3, 96) and score < max(component_score_thr + 0.01, 0.33) and delta_mean < 0.12

        sparse_structured_fp = (
            area < 2600
            and fill_ratio < 0.42
            and build_mean >= 0.62
            and delta_mean >= 0.56
            and prob_mean < 0.035
            and global_local_mean < 0.58
        )
        compact_wrongloc_fp = (
            area < 1700
            and fill_ratio >= 0.45
            and elongation < 2.2
            and prob_mean < 0.05
            and prob_q90 >= 0.50
            and build_mean < 0.50
            and delta_mean < 0.55
            and global_local_mean >= 0.60
        )
        fragmented_rural_fp = (
            area < 1200
            and fill_ratio < 0.36
            and build_mean >= 0.68
            and delta_mean >= 0.62
            and prob_mean < 0.018
        )

        # V2 minimale : terminer le nettoyage des petits résidus faux positifs
        small_residual_structured_fp = (
            area < 2800
            and prob_mean < 0.025
            and prob_q90 < 0.22
            and global_local_mean < 0.52
            and (
                (build_mean >= 0.60 and delta_mean >= 0.58)
                or (area < 1200 and prob_max < 0.18)
            )
        )

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

        if line_like or weak_border_line or tiny_weak or sparse_structured_fp or compact_wrongloc_fp or fragmented_rural_fp or small_residual_structured_fp:
            keep = False

        if keep:
            out[comp] = 1
            kept += 1
            if kept >= int(max_components_keep) and area < int(strong_keep_area):
                break

    out = remove_small_components(out, min_region_size=min_region_size)

    if patch_level_false_positive_veto(out, prob, global_map, building_support, temporal_delta):
        out[:] = 0

    return out


def soft_no_change_veto(mask: np.ndarray, prob: np.ndarray, global_map: np.ndarray, temporal_delta: np.ndarray, global_thr: float, prob_mean_thr: float, peak_thr: float) -> bool:
    if int(mask.sum()) == 0:
        return True
    pred_ratio = float(mask.mean())
    global_mean = float(global_map.mean())
    prob_mean = float(prob.mean())
    peak = float(np.quantile(prob, 0.997))
    delta_mean = float(temporal_delta[mask > 0].mean()) if np.any(mask > 0) else 0.0

    return (
        global_mean < max(0.15, global_thr - 0.03)
        and prob_mean < max(0.020, prob_mean_thr - 0.006)
        and peak < max(0.18, peak_thr - 0.04)
        and delta_mean < 0.14
        and pred_ratio < 0.040
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
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
    parser.add_argument("--levir_osm_texts_json", type=str, default="")
    parser.add_argument("--disable_osm", action="store_true")
    parser.add_argument("--panel_title_bar_h", type=int, default=40)
    parser.add_argument("--panel_font_size", type=int, default=18)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for sub in [
        "npz", "panels", "preview_t1", "preview_t2", "prob", "mask_raw", "mask_precision", "mask",
        "gt_mask", "overlay_pred_raw", "overlay_pred_precision", "overlay_pred", "overlay_gt",
        "global_prob_map", "building_support", "temporal_delta",
    ]:
        ensure_dir(str(out_dir / sub))

    cfg = build_cfg(args)
    osm_json = str(args.levir_osm_texts_json or cfg.get("levir_osm_texts_json", "")).strip()
    if bool(args.disable_osm) or not osm_json:
        osm_helper = None
    else:
        osm_helper = LEVIROSMHelper(
            osm_texts_json=osm_json,
            osm_text_mode=str(cfg.get("levir_osm_text_mode", "concat")),
            osm_max_phrases=int(cfg.get("levir_osm_max_phrases", 3)),
            osm_text_key=str(cfg.get("levir_osm_text_key", "text_v21")),
            osm_fallback_text=str(cfg.get("levir_osm_fallback_text", "no_osm_context")),
            osm_compose_mode=str(cfg.get("levir_osm_compose_mode", "signature_compact")),
            osm_word_budget=int(cfg.get("levir_osm_word_budget", 32)),
            osm_joiner=str(cfg.get("levir_osm_joiner", " ; ")),
            osm_include_source_text=bool(cfg.get("levir_osm_include_source_text", False)),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViLTransformerSS(cfg)
    model.eval().to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    triplets = list_split_triplets(
        args.data_root,
        args.split,
        cfg.get("levir_image_a_dirname", "A"),
        cfg.get("levir_image_b_dirname", "B"),
        cfg.get("levir_label_dirname", "label"),
    )
    tile = int(args.tile_size)
    stride = int(args.tile_stride)
    hann = build_hann_window(tile)
    rows = []

    with torch.no_grad():
        for patch_id, path_a, path_b, path_l in tqdm(triplets, desc=f"Safe export {args.split}"):
            t1_full = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.uint8)
            t2_full = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.uint8)
            gt_full = (np.asarray(Image.open(path_l).convert("L"), dtype=np.uint8) > 127).astype(np.uint8)

            osm_resolved = _resolve_osm(osm_helper, patch_id, fallback_text=str(cfg.get("levir_fixed_text", "building change detection")))
            osm_group_boost = infer_osm_group_boost(osm_resolved)

            h, w = gt_full.shape
            ys = tile_positions(h, tile, stride)
            xs = tile_positions(w, tile, stride)
            prob_sum = np.zeros((h, w), dtype=np.float32)
            gp_sum = np.zeros((h, w), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)

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
                        patch_id=patch_id,
                        osm_helper=osm_helper,
                        osm_resolved=osm_resolved,
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
            temporal_delta = compute_temporal_delta_support(t1_full, t2_full)

            raw_mask = complete_buildings_osm_group(
                prob, building_support, temporal_delta, args.fixed_threshold, args.keep_top_ratio, args.min_region_size, osm_group_boost
            )
            raw_mask = connect_nearby_urban_components(raw_mask, prob, building_support, temporal_delta, osm_group_boost, args.min_region_size)

            precision_mask = precision_refine_components(
                raw_mask, prob, building_support, temporal_delta, args.min_region_size,
                max(0.16, args.precision_gate_prob - 0.04 * osm_group_boost),
                max(0.30, args.precision_gate_build - 0.08 * osm_group_boost),
            )

            dynamic_max_keep = int(args.max_components_keep + round(6 * osm_group_boost))
            final_mask = precision_strict_filter(
                precision_mask, prob, building_support, temporal_delta, global_map, args.min_region_size,
                max(int(args.strong_keep_area * (1.0 - 0.16 * osm_group_boost)), args.min_region_size * 4),
                max(0.28, args.component_score_thr - 0.04 * osm_group_boost),
                args.line_fill_ratio_thr,
                args.line_elongation_thr,
                args.line_build_max_thr,
                args.line_prob_mean_thr,
                dynamic_max_keep,
            )

            if soft_no_change_veto(
                final_mask, prob, global_map, temporal_delta,
                args.nochange_global_thr, args.nochange_prob_mean_thr, args.nochange_peak_thr
            ):
                final_mask[:] = 0

            overlay_pred_raw = make_binary_overlay(t2_full, raw_mask, alpha=args.binary_overlay_alpha)
            overlay_pred_precision = make_binary_overlay(t2_full, precision_mask, alpha=args.binary_overlay_alpha)
            overlay_pred = make_binary_overlay(t2_full, final_mask, alpha=args.binary_overlay_alpha)
            overlay_gt = make_binary_overlay(t2_full, gt_full, alpha=args.binary_overlay_alpha)

            save_png_rgb(t1_full, str(out_dir / "preview_t1" / f"{patch_id}.png"))
            save_png_rgb(t2_full, str(out_dir / "preview_t2" / f"{patch_id}.png"))
            save_png_gray(prob, str(out_dir / "prob" / f"{patch_id}.png"))
            save_png_gray(global_map, str(out_dir / "global_prob_map" / f"{patch_id}.png"))
            save_png_gray(building_support, str(out_dir / "building_support" / f"{patch_id}.png"))
            save_png_gray(temporal_delta, str(out_dir / "temporal_delta" / f"{patch_id}.png"))
            save_png_binary(raw_mask, str(out_dir / "mask_raw" / f"{patch_id}.png"))
            save_png_binary(precision_mask, str(out_dir / "mask_precision" / f"{patch_id}.png"))
            save_png_binary(final_mask, str(out_dir / "mask" / f"{patch_id}.png"))
            save_png_binary(gt_full, str(out_dir / "gt_mask" / f"{patch_id}.png"))
            save_png_rgb(overlay_pred_raw, str(out_dir / "overlay_pred_raw" / f"{patch_id}.png"))
            save_png_rgb(overlay_pred_precision, str(out_dir / "overlay_pred_precision" / f"{patch_id}.png"))
            save_png_rgb(overlay_pred, str(out_dir / "overlay_pred" / f"{patch_id}.png"))
            save_png_rgb(overlay_gt, str(out_dir / "overlay_gt" / f"{patch_id}.png"))

            np.savez_compressed(
                str(out_dir / "npz" / f"{patch_id}.npz"),
                patch_id=patch_id,
                gt_mask=gt_full.astype(np.uint8),
                prob_map=prob.astype(np.float32),
                global_prob_map=global_map.astype(np.float32),
                building_support=building_support.astype(np.float32),
                temporal_delta=temporal_delta.astype(np.float32),
                pred_mask_fused_semantic_stable_raw=raw_mask.astype(np.uint8),
                pred_mask_fused_semantic_stable_precision=precision_mask.astype(np.uint8),
                pred_mask_fused_semantic_stable_precision_strict=final_mask.astype(np.uint8),
                osm_group_boost=np.float32(osm_group_boost),
            )

            panel = make_panel(
                [
                    ("T1 image", t1_full),
                    ("T2 image", t2_full),
                    ("Model probability", np.stack([float01_to_uint8_gray(prob)] * 3, axis=-1)),
                    ("Temporal delta", np.stack([float01_to_uint8_gray(temporal_delta)] * 3, axis=-1)),
                    ("Raw mask", np.stack([binary_mask_to_uint8(raw_mask)] * 3, axis=-1)),
                    ("Precision mask", np.stack([binary_mask_to_uint8(precision_mask)] * 3, axis=-1)),
                    ("Final mask", np.stack([binary_mask_to_uint8(final_mask)] * 3, axis=-1)),
                    ("Prediction overlay", overlay_pred),
                    ("GT overlay", overlay_gt),
                ],
                pad=8,
                title_bar_h=int(args.panel_title_bar_h),
                font_size=int(args.panel_font_size),
            )
            save_png_rgb(panel, str(out_dir / "panels" / f"{patch_id}.png"))

            rows.append(
                {
                    "patch_id": patch_id,
                    "osm_group_boost": float(osm_group_boost),
                    "prob_mean": float(prob.mean()),
                    "global_mean": float(global_map.mean()),
                    "temporal_delta_mean": float(temporal_delta.mean()),
                    "pred_ratio_raw": float(raw_mask.mean()),
                    "pred_ratio_precision": float(precision_mask.mean()),
                    "pred_ratio_final": float(final_mask.mean()),
                    "gt_ratio": float(gt_full.mean()),
                }
            )

    csv_path = out_dir / "export_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "patch_id",
            "osm_group_boost",
            "prob_mean",
            "global_mean",
            "temporal_delta_mean",
            "pred_ratio_raw",
            "pred_ratio_precision",
            "pred_ratio_final",
            "gt_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Export summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
