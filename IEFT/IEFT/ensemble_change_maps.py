import os
import argparse
import shutil
import numpy as np
from scipy import ndimage as ndi


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def clean_binary_mask(mask: np.ndarray, min_region_size: int = 40, closing_iters: int = 0) -> np.ndarray:
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


def parse_weights(weights_str: str, n_dirs: int):
    if not weights_str:
        return np.ones(n_dirs, dtype=np.float32) / float(n_dirs)

    vals = [float(x.strip()) for x in weights_str.split(",") if x.strip()]
    if len(vals) != n_dirs:
        raise ValueError(f"Nombre de poids ({len(vals)}) != nombre de dossiers ({n_dirs})")

    w = np.asarray(vals, dtype=np.float32)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("La somme des poids doit être > 0")
    return w / s


def list_patch_ids(npz_dir: str):
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"Dossier introuvable: {npz_dir}")
    files = sorted([f for f in os.listdir(npz_dir) if f.lower().endswith(".npz")])
    return [os.path.splitext(f)[0] for f in files]


def load_npz(npz_path: str):
    return np.load(npz_path, allow_pickle=True)


def maybe_copy_tree(src_root: str, dst_root: str, subdir: str):
    src = os.path.join(src_root, subdir)
    dst = os.path.join(dst_root, subdir)
    if os.path.isdir(src):
        ensure_dir(dst)
        for name in os.listdir(src):
            src_file = os.path.join(src, name)
            dst_file = os.path.join(dst, name)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dirs",
        type=str,
        required=True,
        help='Liste de dossiers séparés par ";" contenant chacun un sous-dossier npz'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Dossier de sortie ensemble"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help='Poids optionnels séparés par "," dans le même ordre que input_dirs, ex: "0.5,0.3,0.2"'
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--keep_top_ratio", type=float, default=0.20)
    parser.add_argument("--min_region_size", type=int, default=40)
    parser.add_argument("--closing_iters", type=int, default=0)
    parser.add_argument(
        "--copy_previews_from_first",
        action="store_true",
        help="Copier preview_t1 / preview_t2 / panels / overlays depuis le premier dossier si présents"
    )
    args = parser.parse_args()

    input_dirs = [x.strip() for x in args.input_dirs.split(";") if x.strip()]
    if len(input_dirs) < 2:
        raise ValueError("Il faut au moins 2 dossiers dans --input_dirs")

    npz_dirs = [os.path.join(d, "npz") for d in input_dirs]
    for d in npz_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Sous-dossier npz introuvable: {d}")

    patch_ids_ref = list_patch_ids(npz_dirs[0])
    ref_set = set(patch_ids_ref)

    for npz_dir in npz_dirs[1:]:
        s = set(list_patch_ids(npz_dir))
        if s != ref_set:
            missing = sorted(list(ref_set - s))[:10]
            extra = sorted(list(s - ref_set))[:10]
            raise ValueError(
                f"Les patch_ids ne correspondent pas entre dossiers.\n"
                f"Manquants sample: {missing}\n"
                f"Extras sample: {extra}"
            )

    weights = parse_weights(args.weights, len(input_dirs))

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "npz"))

    # Créer la même structure de base
    for sub in [
        "preview_t1", "preview_t2", "pred_prob_raw", "pred_prob_norm",
        "mask_fixed", "mask_adapt", "mask_adapt_norm_raw", "mask_adapt_norm_clean",
        "pseudo_mask", "confident_mask", "overlay_adapt_norm_clean", "pseudo_overlay", "panels"
    ]:
        ensure_dir(os.path.join(args.output_dir, sub))

    first_root = input_dirs[0]
    if args.copy_previews_from_first:
        for sub in [
            "preview_t1", "preview_t2", "pseudo_mask", "confident_mask",
            "overlay_adapt_norm_clean", "pseudo_overlay", "panels"
        ]:
            maybe_copy_tree(first_root, args.output_dir, sub)

    print("[INFO] Input dirs:")
    for w, d in zip(weights, input_dirs):
        print(f"  - weight={float(w):.6f} | {d}")

    exported = 0
    n = len(patch_ids_ref)

    for idx, patch_id in enumerate(patch_ids_ref, start=1):
        entries = []
        for root in input_dirs:
            z = load_npz(os.path.join(root, "npz", f"{patch_id}.npz"))
            entries.append(z)

        pred_probs = [np.asarray(z["pred_prob_128x128"], dtype=np.float32) for z in entries]
        avg_prob = np.zeros_like(pred_probs[0], dtype=np.float32)
        for w, p in zip(weights, pred_probs):
            avg_prob += float(w) * p
        avg_prob = np.clip(avg_prob, 0.0, 1.0)

        pred_prob_norm = robust_normalize_map(avg_prob, q_low=0.05, q_high=0.95)
        pred_bin_fixed = (avg_prob >= float(args.threshold)).astype(np.uint8)
        pred_bin_adapt = adaptive_binary_mask(avg_prob, keep_top_ratio=float(args.keep_top_ratio))
        pred_bin_adapt_norm_raw = adaptive_binary_mask(pred_prob_norm, keep_top_ratio=float(args.keep_top_ratio))
        pred_bin_adapt_norm_clean = clean_binary_mask(
            pred_bin_adapt_norm_raw,
            min_region_size=int(args.min_region_size),
            closing_iters=int(args.closing_iters),
        )

        # pseudo/confident: on reprend le premier dossier (normalement identiques pour tous les modèles)
        z0 = entries[0]
        pseudo_bin = np.asarray(z0["pseudo_mask_128x128"], dtype=np.uint8)
        conf_bin = np.asarray(z0["confident_mask_128x128"], dtype=np.uint8)
        split = str(z0["split"])

        np.savez_compressed(
            os.path.join(args.output_dir, "npz", f"{patch_id}.npz"),
            patch_id=patch_id,
            split=split,
            pred_prob_128x128=avg_prob.astype(np.float32),
            pred_prob_norm_128x128=pred_prob_norm.astype(np.float32),
            pred_mask_fixed=pred_bin_fixed.astype(np.uint8),
            pred_mask_adapt=pred_bin_adapt.astype(np.uint8),
            pred_mask_adapt_norm_raw=pred_bin_adapt_norm_raw.astype(np.uint8),
            pred_mask_adapt_norm_clean=pred_bin_adapt_norm_clean.astype(np.uint8),
            pseudo_mask_128x128=pseudo_bin.astype(np.uint8),
            confident_mask_128x128=conf_bin.astype(np.uint8),
            pred_fixed_ratio=float(pred_bin_fixed.mean()),
            pred_adapt_ratio=float(pred_bin_adapt.mean()),
            pred_adapt_norm_raw_ratio=float(pred_bin_adapt_norm_raw.mean()),
            pred_adapt_norm_clean_ratio=float(pred_bin_adapt_norm_clean.mean()),
            pseudo_ratio=float(pseudo_bin.mean()),
            confident_ratio=float(conf_bin.mean()),
        )

        exported += 1
        if idx % 25 == 0 or idx == n:
            print(f"[INFO] Processed {idx}/{n}")

    print(f"[INFO] Exported ensemble npz for {exported} patches.")
    print(f"[INFO] Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
