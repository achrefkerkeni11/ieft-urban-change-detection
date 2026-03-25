import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm
from scipy import ndimage as ndi

from IEFT.modules.vilt_module import ViLTransformerSS
from IEFT.datamodules.s2_npz_datamodule import S2NPZDataModule
from IEFT.modules.objectives import _compute_patchwise_change_pseudo_labels


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_rgb_to_uint8_image(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().float().clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def float01_to_uint8_gray(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.asarray(arr01, dtype=np.float32)
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0).round().astype(np.uint8)


def binary_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    return ((mask > 0).astype(np.uint8) * 255)


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


def add_title_bar(img: np.ndarray, title: str, bar_h: int = 28) -> np.ndarray:
    h, w, c = img.shape
    canvas = np.full((h + bar_h, w, c), 255, dtype=np.uint8)
    canvas[bar_h:, :, :] = img
    canvas[bar_h - 1:bar_h, :, :] = 180

    pil = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.text((6, 6), title, fill=(0, 0, 0))
    return np.array(pil)


def make_panel(images_with_titles, pad: int = 8, bg: int = 255) -> np.ndarray:
    prepared = []
    max_h = 0

    for title, img in images_with_titles:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = add_title_bar(img, title)
        prepared.append(img)
        max_h = max(max_h, img.shape[0])

    total_w = sum(img.shape[1] for img in prepared) + pad * (len(prepared) + 1)
    panel = np.full((max_h + 2 * pad, total_w, 3), bg, dtype=np.uint8)

    x = pad
    for img in prepared:
        h, w, _ = img.shape
        y = pad + (max_h - h) // 2
        panel[y:y + h, x:x + w] = img
        x += w + pad

    return panel


def resize_rgb(arr_uint8: np.ndarray, size: int) -> np.ndarray:
    return np.array(
        Image.fromarray(arr_uint8, mode="RGB").resize(
            (size, size), resample=Image.BILINEAR
        )
    )


def robust_normalize_map(arr: np.ndarray, q_low: float = 0.05, q_high: float = 0.95, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    out = (arr - lo) / (hi - lo + eps)
    return np.clip(out, 0.0, 1.0)


def adaptive_binary_mask(arr: np.ndarray, keep_top_ratio: float = 0.20) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    thr = float(np.quantile(arr, 1.0 - keep_top_ratio))
    return (arr >= thr).astype(np.uint8)


def clean_binary_mask(mask: np.ndarray, min_region_size: int = 80, closing_iters: int = 1) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8)

    if closing_iters > 0:
        mask = ndi.binary_closing(mask, iterations=closing_iters)

    mask = ndi.binary_fill_holes(mask)

    labeled, num = ndi.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    out = np.zeros_like(mask, dtype=np.uint8)
    sizes = ndi.sum(mask, labeled, index=np.arange(1, num + 1))

    for i, s in enumerate(sizes, start=1):
        if s >= min_region_size:
            out[labeled == i] = 1

    return out.astype(np.uint8)


def _find_hparams_path_from_ckpt(ckpt_path: str) -> str:
    ckpt_path = os.path.abspath(ckpt_path)
    version_dir = os.path.dirname(os.path.dirname(ckpt_path))
    hparams_path = os.path.join(version_dir, "hparams.yaml")
    if not os.path.isfile(hparams_path):
        raise FileNotFoundError(f"hparams.yaml introuvable: {hparams_path}")
    return hparams_path


def _load_hparams_config(ckpt_path: str) -> dict:
    hparams_path = _find_hparams_path_from_ckpt(ckpt_path)

    with open(hparams_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Contenu invalide dans {hparams_path}")

    if "config" in data and isinstance(data["config"], dict):
        cfg = dict(data["config"])
    elif "hyper_parameters" in data and isinstance(data["hyper_parameters"], dict):
        cfg = dict(data["hyper_parameters"])
    else:
        cfg = dict(data)

    print(f"[INFO] Loaded hparams from: {hparams_path}")
    return cfg


def build_cfg(args):
    defaults = {
        "data_root": "data_npz",
        "batch_size": 1,
        "per_gpu_batchsize": 1,
        "num_workers": 0,
        "image_size": 384,
        "max_text_len": 40,
        "draw_false_image": 1,
        "draw_false_text": 2,
        "tokenizer": "bert-base-uncased",
        "seed": 0,
        "num_gpus": 0,
        "num_nodes": 1,
        "precision": 16,
        "fast_dev_run": False,
        "resume_from": None,
        "test_only": False,
        "get_recall_metric": False,
        "vit": "vit_base_patch32_384",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "drop_rate": 0.1,
        "max_image_len": -1,
        "load_path": args.ckpt_path,
        "vqav2_label_size": 3129,
        "exp_name": "export_change_maps",
        "loss_names": {
            "mlm": 0,
            "itm": 0,
            "mpp": 0,
            "vqa": 0,
            "nlvr2": 0,
            "irtr": 1,
        },
        # ===== OSM defaults corrected for rich JSON =====
        "osm_texts_json": "",
        "osm_text_mode": "concat",
        "osm_max_phrases": 2,
        "osm_text_key": "text_v12",
        "osm_fallback_text": "no_osm_context",
        "osm_compose_mode": "signature_compact",
        "osm_word_budget": 28,
        "osm_joiner": " ; ",
        "osm_include_source_text": False,
        # ===== Change =====
        "change_loss_weight": 1.0,
        "change_global_loss_weight": 0.2,
        "change_pos_quantile": 0.80,
        "change_neg_quantile": 0.45,
        "change_smoothness_loss_weight": 0.05,
        # ===== Semantic gate =====
        "use_semantic_change_gate": True,
        "change_semantic_patch_weight": 0.20,
        "change_semantic_global_weight": 0.35,
        "change_semantic_patch_loss_weight": 0.05,
        "change_semantic_global_loss_weight": 0.10,
        "change_fusion_alpha": 0.65,
        "change_fusion_beta": 0.20,
        "change_fusion_gamma": 0.15,
        "s2_scale_div": 10000.0,
    }

    cfg = defaults.copy()
    ckpt_cfg = _load_hparams_config(args.ckpt_path)
    cfg.update(ckpt_cfg)

    cfg["data_root"] = args.data_root
    cfg["batch_size"] = args.batch_size
    cfg["per_gpu_batchsize"] = args.batch_size
    cfg["num_workers"] = args.num_workers
    cfg["image_size"] = args.image_size
    cfg["max_text_len"] = args.max_text_len
    cfg["draw_false_text"] = args.draw_false_text
    cfg["load_path"] = args.ckpt_path

    cfg["osm_texts_json"] = args.osm_texts_json
    cfg["osm_text_mode"] = args.osm_text_mode
    cfg["osm_max_phrases"] = args.osm_max_phrases
    cfg["osm_text_key"] = args.osm_text_key
    cfg["osm_fallback_text"] = args.osm_fallback_text
    cfg["osm_compose_mode"] = args.osm_compose_mode
    cfg["osm_word_budget"] = args.osm_word_budget
    cfg["osm_joiner"] = args.osm_joiner
    cfg["osm_include_source_text"] = bool(args.osm_include_source_text)

    cfg["change_pos_quantile"] = args.change_pos_quantile
    cfg["change_neg_quantile"] = args.change_neg_quantile

    cfg["seed"] = int(cfg.get("seed", 0))
    cfg["num_gpus"] = 0
    cfg["num_nodes"] = 1
    cfg["fast_dev_run"] = False
    cfg["resume_from"] = None
    cfg["test_only"] = False
    cfg["get_recall_metric"] = False
    cfg["exp_name"] = "export_change_maps"

    print("[INFO] Effective export config:")
    for k in [
        "load_path",
        "data_root",
        "batch_size",
        "num_workers",
        "image_size",
        "max_text_len",
        "draw_false_text",
        "tokenizer",
        "vit",
        "hidden_size",
        "num_layers",
        "num_heads",
        "mlp_ratio",
        "drop_rate",
        "max_image_len",
        "precision",
        "osm_texts_json",
        "osm_text_mode",
        "osm_max_phrases",
        "osm_text_key",
        "osm_fallback_text",
        "osm_compose_mode",
        "osm_word_budget",
        "osm_joiner",
        "osm_include_source_text",
        "use_semantic_change_gate",
        "change_semantic_patch_weight",
        "change_semantic_global_weight",
        "change_fusion_alpha",
        "change_fusion_beta",
        "change_fusion_gamma",
        "change_loss_weight",
        "change_global_loss_weight",
        "change_pos_quantile",
        "change_neg_quantile",
        "s2_scale_div",
    ]:
        print(f"  - {k}: {cfg.get(k)}")

    return cfg


def get_loader(dm, split):
    if split == "train":
        return dm.train_dataloader()
    if split == "val":
        return dm.val_dataloader()
    if split == "test":
        return dm.test_dataloader()
    raise ValueError(f"Unknown split: {split}")


def safe_patch_id(x) -> str:
    s = str(x)
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, "_")
    return s


def compute_pseudo_maps(batch, g: int, upsample_size: int, s2_scale_div: float,
                        pos_quantile: float = 0.80, neg_quantile: float = 0.45):
    pseudo_labels, pseudo_scores, confident_mask = _compute_patchwise_change_pseudo_labels(
        batch["x8"].float(),
        grid_size=g,
        s2_scale_div=s2_scale_div,
        pos_quantile=pos_quantile,
        neg_quantile=neg_quantile,
    )

    pseudo_grid = pseudo_labels.view(-1, g, g).unsqueeze(1).float()
    pseudo_score_grid = pseudo_scores.view(-1, g, g).unsqueeze(1).float()
    confident_grid = confident_mask.view(-1, g, g).unsqueeze(1).float()

    pseudo_up = F.interpolate(
        pseudo_grid,
        size=(upsample_size, upsample_size),
        mode="nearest",
    ).squeeze(1)

    pseudo_score_up = F.interpolate(
        pseudo_score_grid,
        size=(upsample_size, upsample_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    confident_up = F.interpolate(
        confident_grid,
        size=(upsample_size, upsample_size),
        mode="nearest",
    ).squeeze(1)

    return pseudo_labels, pseudo_scores, confident_mask, pseudo_up, pseudo_score_up, confident_up


def maybe_export_semantic_patch_map(out: dict, upsample_size: int):
    logits = out.get("change_semantic_patch_logits", None)
    if logits is None:
        return None, None

    if logits.ndim != 2:
        return None, None

    bsz, n = logits.shape
    g = int(np.sqrt(n))
    if g * g != n:
        return None, None

    semantic_prob = torch.sigmoid(logits).view(bsz, 1, g, g)
    semantic_up = F.interpolate(
        semantic_prob,
        size=(upsample_size, upsample_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    semantic_up = semantic_up.detach().cpu().numpy().astype(np.float32)
    semantic_up_norm = np.stack([robust_normalize_map(x) for x in semantic_up], axis=0)
    return semantic_up, semantic_up_norm


def maybe_export_semantic_global_prob(out: dict):
    logits = out.get("change_semantic_global_logits", None)
    if logits is None:
        return None
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data_npz")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--upsample_size", type=int, default=128)
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_text_len", type=int, default=40)
    parser.add_argument("--draw_false_text", type=int, default=2)

    parser.add_argument("--osm_texts_json", type=str, default="")
    parser.add_argument("--osm_text_mode", type=str, default="concat", choices=["first", "random", "concat"])
    parser.add_argument("--osm_max_phrases", type=int, default=2)
    parser.add_argument("--osm_text_key", type=str, default="text_v12")
    parser.add_argument("--osm_fallback_text", type=str, default="no_osm_context")
    parser.add_argument(
        "--osm_compose_mode",
        type=str,
        default="signature_compact",
        choices=["legacy", "main_only", "summary_plus_tags", "phrases_compact", "signature_compact", "auto"],
    )
    parser.add_argument("--osm_word_budget", type=int, default=28)
    parser.add_argument("--osm_joiner", type=str, default=" ; ")
    parser.add_argument("--osm_include_source_text", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--binary_overlay_alpha", type=float, default=0.55)

    parser.add_argument("--keep_top_ratio", type=float, default=0.20,
                        help="Adaptive mask keeps top X ratio of predicted scores.")
    parser.add_argument("--change_pos_quantile", type=float, default=0.80)
    parser.add_argument("--change_neg_quantile", type=float, default=0.45)

    parser.add_argument("--min_region_size", type=int, default=80,
                        help="Remove predicted regions smaller than this size.")
    parser.add_argument("--closing_iters", type=int, default=1,
                        help="Binary closing iterations for mask cleanup.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "npz"))
    ensure_dir(os.path.join(args.output_dir, "panels"))
    ensure_dir(os.path.join(args.output_dir, "preview_t1"))
    ensure_dir(os.path.join(args.output_dir, "preview_t2"))
    ensure_dir(os.path.join(args.output_dir, "pred_prob_raw"))
    ensure_dir(os.path.join(args.output_dir, "pred_prob_norm"))
    ensure_dir(os.path.join(args.output_dir, "mask_fixed"))
    ensure_dir(os.path.join(args.output_dir, "mask_adapt"))
    ensure_dir(os.path.join(args.output_dir, "mask_adapt_norm_raw"))
    ensure_dir(os.path.join(args.output_dir, "mask_adapt_norm_clean"))
    ensure_dir(os.path.join(args.output_dir, "pseudo_mask"))
    ensure_dir(os.path.join(args.output_dir, "confident_mask"))
    ensure_dir(os.path.join(args.output_dir, "overlay_adapt_norm_clean"))
    ensure_dir(os.path.join(args.output_dir, "pseudo_overlay"))
    ensure_dir(os.path.join(args.output_dir, "semantic_patch_prob"))
    ensure_dir(os.path.join(args.output_dir, "semantic_patch_prob_norm"))
    ensure_dir(os.path.join(args.output_dir, "coarse_prob_raw"))
    ensure_dir(os.path.join(args.output_dir, "coarse_prob_norm"))
    ensure_dir(os.path.join(args.output_dir, "refined_prob_raw"))
    ensure_dir(os.path.join(args.output_dir, "refined_prob_norm"))
    ensure_dir(os.path.join(args.output_dir, "boundary_prob"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    cfg = build_cfg(args)

    print("[1] Build datamodule")
    dm = S2NPZDataModule(cfg)
    dm.setup()
    cfg["vocab_size"] = dm.vocab_size

    print("[2] Build model")
    model = ViLTransformerSS(cfg)
    model = model.to(device)
    model.eval()

    print("[3] Build loader")
    loader = get_loader(dm, args.split)

    exported = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Exporting {args.split} cleaned comparison maps"):
            batch_device = move_to_device(batch, device)
            out = model.infer(batch_device)

            change_map = out["change_map"]
            if change_map is None:
                raise RuntimeError("change_map is None.")

            change_map_12 = change_map.unsqueeze(1)
            coarse_map_up = F.interpolate(
                change_map_12,
                size=(args.upsample_size, args.upsample_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            refined_map_up = out.get("change_refined_map_up", None)
            boundary_prob_up = out.get("change_boundary_probs_up", None)
            if refined_map_up is None:
                change_map_up = coarse_map_up
            else:
                change_map_up = refined_map_up

            semantic_patch_up, semantic_patch_up_norm = maybe_export_semantic_patch_map(
                out,
                upsample_size=args.upsample_size,
            )
            semantic_global_prob = maybe_export_semantic_global_prob(out)

            bsz, g, _ = change_map.shape

            (
                pseudo_labels,
                pseudo_scores,
                confident_mask,
                pseudo_up,
                pseudo_score_up,
                confident_up,
            ) = compute_pseudo_maps(
                batch=batch,
                g=g,
                upsample_size=args.upsample_size,
                s2_scale_div=float(cfg["s2_scale_div"]),
                pos_quantile=float(cfg["change_pos_quantile"]),
                neg_quantile=float(cfg["change_neg_quantile"]),
            )

            files = batch["file"]
            rgb_t1 = batch["image_t1"][0]
            rgb_t2 = batch["image_t2"][0]

            for i in range(bsz):
                patch_id = safe_patch_id(files[i])

                coarse_prob = np.clip(coarse_map_up[i].detach().cpu().numpy(), 0.0, 1.0)
                coarse_prob_norm = robust_normalize_map(coarse_prob, q_low=0.05, q_high=0.95)

                pred_prob = np.clip(change_map_up[i].detach().cpu().numpy(), 0.0, 1.0)
                pred_prob_norm = robust_normalize_map(pred_prob, q_low=0.05, q_high=0.95)

                boundary_prob = None
                if boundary_prob_up is not None:
                    boundary_prob = np.clip(boundary_prob_up[i].detach().cpu().numpy(), 0.0, 1.0)

                pred_bin_fixed = (pred_prob >= float(args.threshold)).astype(np.uint8)
                pred_bin_adapt = adaptive_binary_mask(pred_prob, keep_top_ratio=float(args.keep_top_ratio))
                pred_bin_adapt_norm_raw = adaptive_binary_mask(pred_prob_norm, keep_top_ratio=float(args.keep_top_ratio))
                pred_bin_adapt_norm_clean = clean_binary_mask(
                    pred_bin_adapt_norm_raw,
                    min_region_size=int(args.min_region_size),
                    closing_iters=int(args.closing_iters),
                )

                pseudo_bin = (pseudo_up[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
                conf_bin = (confident_up[i].detach().cpu().numpy() > 0.5).astype(np.uint8)

                semantic_patch_prob = None
                semantic_patch_prob_norm = None
                if semantic_patch_up is not None:
                    semantic_patch_prob = np.clip(semantic_patch_up[i], 0.0, 1.0)
                    semantic_patch_prob_norm = np.clip(semantic_patch_up_norm[i], 0.0, 1.0)

                semantic_global_value = None
                if semantic_global_prob is not None:
                    semantic_global_value = float(semantic_global_prob[i])

                rgb1 = resize_rgb(tensor_rgb_to_uint8_image(rgb_t1[i]), args.upsample_size)
                rgb2 = resize_rgb(tensor_rgb_to_uint8_image(rgb_t2[i]), args.upsample_size)

                overlay_adapt_norm_clean = make_binary_overlay(
                    rgb2, pred_bin_adapt_norm_clean, alpha=float(args.binary_overlay_alpha)
                )
                pseudo_overlay = make_binary_overlay(
                    rgb2, pseudo_bin, alpha=float(args.binary_overlay_alpha)
                )

                panel_items = [
                    ("T1", rgb1),
                    ("T2", rgb2),
                    ("PredProbRaw", np.stack([float01_to_uint8_gray(pred_prob)] * 3, axis=-1)),
                    ("PredProbNorm", np.stack([float01_to_uint8_gray(pred_prob_norm)] * 3, axis=-1)),
                    ("MaskFixed", np.stack([binary_mask_to_uint8(pred_bin_fixed)] * 3, axis=-1)),
                    ("MaskAdapt", np.stack([binary_mask_to_uint8(pred_bin_adapt)] * 3, axis=-1)),
                    ("MaskAdaptNormRaw", np.stack([binary_mask_to_uint8(pred_bin_adapt_norm_raw)] * 3, axis=-1)),
                    ("MaskAdaptNormClean", np.stack([binary_mask_to_uint8(pred_bin_adapt_norm_clean)] * 3, axis=-1)),
                    ("PseudoMask", np.stack([binary_mask_to_uint8(pseudo_bin)] * 3, axis=-1)),
                    ("ConfidentMask", np.stack([binary_mask_to_uint8(conf_bin)] * 3, axis=-1)),
                    ("OverlayAdaptNormClean", overlay_adapt_norm_clean),
                    ("PseudoOverlay", pseudo_overlay),
                ]
                if semantic_patch_prob is not None:
                    panel_items.append(("SemanticPatchProb", np.stack([float01_to_uint8_gray(semantic_patch_prob)] * 3, axis=-1)))
                    panel_items.append(("SemanticPatchProbNorm", np.stack([float01_to_uint8_gray(semantic_patch_prob_norm)] * 3, axis=-1)))
                panel = make_panel(panel_items)

                npz_payload = {
                    "patch_id": patch_id,
                    "split": args.split,
                    "pred_prob_128x128": pred_prob.astype(np.float32),
                    "pred_prob_norm_128x128": pred_prob_norm.astype(np.float32),
                    "pred_mask_fixed": pred_bin_fixed.astype(np.uint8),
                    "pred_mask_adapt": pred_bin_adapt.astype(np.uint8),
                    "pred_mask_adapt_norm_raw": pred_bin_adapt_norm_raw.astype(np.uint8),
                    "pred_mask_adapt_norm_clean": pred_bin_adapt_norm_clean.astype(np.uint8),
                    "pseudo_mask_128x128": pseudo_bin.astype(np.uint8),
                    "confident_mask_128x128": conf_bin.astype(np.uint8),
                    "pred_fixed_ratio": float(pred_bin_fixed.mean()),
                    "pred_adapt_ratio": float(pred_bin_adapt.mean()),
                    "pred_adapt_norm_raw_ratio": float(pred_bin_adapt_norm_raw.mean()),
                    "pred_adapt_norm_clean_ratio": float(pred_bin_adapt_norm_clean.mean()),
                    "pseudo_ratio": float(pseudo_bin.mean()),
                    "confident_ratio": float(conf_bin.mean()),
                }
                if semantic_patch_prob is not None:
                    npz_payload["semantic_patch_prob_128x128"] = semantic_patch_prob.astype(np.float32)
                    npz_payload["semantic_patch_prob_norm_128x128"] = semantic_patch_prob_norm.astype(np.float32)
                if semantic_global_value is not None:
                    npz_payload["semantic_global_prob"] = np.float32(semantic_global_value)

                np.savez_compressed(
                    os.path.join(args.output_dir, "npz", f"{patch_id}.npz"),
                    **npz_payload,
                )

                save_png_rgb(rgb1, os.path.join(args.output_dir, "preview_t1", f"{patch_id}_t1.png"))
                save_png_rgb(rgb2, os.path.join(args.output_dir, "preview_t2", f"{patch_id}_t2.png"))
                save_png_gray(pred_prob, os.path.join(args.output_dir, "pred_prob_raw", f"{patch_id}_pred_prob_raw.png"))
                save_png_gray(pred_prob_norm, os.path.join(args.output_dir, "pred_prob_norm", f"{patch_id}_pred_prob_norm.png"))
                save_png_binary(pred_bin_fixed, os.path.join(args.output_dir, "mask_fixed", f"{patch_id}_mask_fixed.png"))
                save_png_binary(pred_bin_adapt, os.path.join(args.output_dir, "mask_adapt", f"{patch_id}_mask_adapt.png"))
                save_png_binary(pred_bin_adapt_norm_raw, os.path.join(args.output_dir, "mask_adapt_norm_raw", f"{patch_id}_mask_adapt_norm_raw.png"))
                save_png_binary(pred_bin_adapt_norm_clean, os.path.join(args.output_dir, "mask_adapt_norm_clean", f"{patch_id}_mask_adapt_norm_clean.png"))
                save_png_binary(pseudo_bin, os.path.join(args.output_dir, "pseudo_mask", f"{patch_id}_pseudo_mask.png"))
                save_png_binary(conf_bin, os.path.join(args.output_dir, "confident_mask", f"{patch_id}_confident_mask.png"))
                save_png_rgb(overlay_adapt_norm_clean, os.path.join(args.output_dir, "overlay_adapt_norm_clean", f"{patch_id}_overlay_adapt_norm_clean.png"))
                save_png_rgb(pseudo_overlay, os.path.join(args.output_dir, "pseudo_overlay", f"{patch_id}_pseudo_overlay.png"))
                if semantic_patch_prob is not None:
                    save_png_gray(semantic_patch_prob, os.path.join(args.output_dir, "semantic_patch_prob", f"{patch_id}_semantic_patch_prob.png"))
                    save_png_gray(semantic_patch_prob_norm, os.path.join(args.output_dir, "semantic_patch_prob_norm", f"{patch_id}_semantic_patch_prob_norm.png"))
                save_png_rgb(panel, os.path.join(args.output_dir, "panels", f"{patch_id}_panel.png"))

                exported += 1
                if args.max_items > 0 and exported >= args.max_items:
                    print(f"\n[INFO] Reached max_items={args.max_items}")
                    print(f"[INFO] Exported {exported} samples.")
                    print(f"[INFO] Output dir: {args.output_dir}")
                    return

    print(f"\n[INFO] Exported {exported} samples.")
    print(f"[INFO] Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
