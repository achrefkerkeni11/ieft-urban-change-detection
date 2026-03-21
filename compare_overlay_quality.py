import os
import csv
import argparse
import numpy as np


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def compute_binary_metrics(pred, ref):
    pred = (pred > 0).astype(np.uint8)
    ref = (ref > 0).astype(np.uint8)

    tp = int(((pred == 1) & (ref == 1)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    tn = int(((pred == 0) & (ref == 0)).sum())

    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def compute_binary_metrics_in_region(pred, ref, region_mask):
    region_mask = (region_mask > 0)

    if region_mask.sum() == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "num_pixels": 0,
        }

    pred_r = pred[region_mask]
    ref_r = ref[region_mask]

    out = compute_binary_metrics(pred_r, ref_r)
    out["num_pixels"] = int(region_mask.sum())
    return out


def list_npz_files(folder):
    npz_dir = os.path.join(folder, "npz")
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"NPZ folder not found: {npz_dir}")

    files = []
    for name in os.listdir(npz_dir):
        if name.lower().endswith(".npz"):
            files.append(os.path.join(npz_dir, name))
    files.sort()
    return files


def summarize_rows(rows, key):
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    if len(vals) == 0:
        return {}
    return {
        f"{key}_mean": float(vals.mean()),
        f"{key}_std": float(vals.std()),
        f"{key}_min": float(vals.min()),
        f"{key}_max": float(vals.max()),
        f"{key}_median": float(np.median(vals)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder that contains npz/ produced by export_change_maps.py",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path for per-patch metrics",
    )
    args = parser.parse_args()

    npz_files = list_npz_files(args.input_dir)
    rows = []

    for path in npz_files:
        data = np.load(path)

        patch_id = str(data["patch_id"])

        pred = data["pred_mask_adapt_norm_clean"].astype(np.uint8)
        pseudo = data["pseudo_mask_128x128"].astype(np.uint8)
        confident = data["confident_mask_128x128"].astype(np.uint8)

        global_metrics = compute_binary_metrics(pred, pseudo)
        confident_metrics = compute_binary_metrics_in_region(pred, pseudo, confident)

        pred_ratio = float(pred.mean())
        pseudo_ratio = float(pseudo.mean())
        confident_ratio = float(confident.mean())

        abs_ratio_gap = abs(pred_ratio - pseudo_ratio)

        row = {
            "patch_id": patch_id,

            "pred_ratio": pred_ratio,
            "pseudo_ratio": pseudo_ratio,
            "confident_ratio": confident_ratio,
            "abs_ratio_gap": abs_ratio_gap,

            "iou": global_metrics["iou"],
            "dice": global_metrics["dice"],
            "precision": global_metrics["precision"],
            "recall": global_metrics["recall"],
            "accuracy": global_metrics["accuracy"],

            "tp": global_metrics["tp"],
            "fp": global_metrics["fp"],
            "fn": global_metrics["fn"],
            "tn": global_metrics["tn"],

            "iou_confident": confident_metrics["iou"],
            "dice_confident": confident_metrics["dice"],
            "precision_confident": confident_metrics["precision"],
            "recall_confident": confident_metrics["recall"],
            "accuracy_confident": confident_metrics["accuracy"],
            "confident_pixels": confident_metrics["num_pixels"],
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    fieldnames = [
        "patch_id",
        "pred_ratio",
        "pseudo_ratio",
        "confident_ratio",
        "abs_ratio_gap",
        "iou",
        "dice",
        "precision",
        "recall",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "iou_confident",
        "dice_confident",
        "precision_confident",
        "recall_confident",
        "accuracy_confident",
        "confident_pixels",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[INFO] Per-patch comparison saved to:")
    print(args.output_csv)
    print()
    print(f"[INFO] Number of patches: {len(rows)}")

    keys_to_summarize = [
        "abs_ratio_gap",
        "iou",
        "dice",
        "precision",
        "recall",
        "accuracy",
        "iou_confident",
        "dice_confident",
        "precision_confident",
        "recall_confident",
        "accuracy_confident",
    ]

    print("\n[INFO] Global summary:")
    for key in keys_to_summarize:
        s = summarize_rows(rows, key)
        print(
            f"{key:20s} "
            f"mean={s[f'{key}_mean']:.6f}  "
            f"std={s[f'{key}_std']:.6f}  "
            f"min={s[f'{key}_min']:.6f}  "
            f"median={s[f'{key}_median']:.6f}  "
            f"max={s[f'{key}_max']:.6f}"
        )

    rows_sorted_good = sorted(rows, key=lambda x: x["dice_confident"], reverse=True)
    rows_sorted_bad = sorted(rows, key=lambda x: x["dice_confident"])

    print("\n[INFO] Top 10 patches with BEST dice_confident:")
    for r in rows_sorted_good[:10]:
        print(
            f"{r['patch_id']} | "
            f"dice_confident={r['dice_confident']:.6f} | "
            f"iou_confident={r['iou_confident']:.6f} | "
            f"dice={r['dice']:.6f}"
        )

    print("\n[INFO] Top 10 patches with WORST dice_confident:")
    for r in rows_sorted_bad[:10]:
        print(
            f"{r['patch_id']} | "
            f"dice_confident={r['dice_confident']:.6f} | "
            f"iou_confident={r['iou_confident']:.6f} | "
            f"dice={r['dice']:.6f}"
        )


if __name__ == "__main__":
    main()