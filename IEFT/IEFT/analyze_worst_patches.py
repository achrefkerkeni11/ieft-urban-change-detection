
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def safe_div(a, b):
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def binary_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def component_stats(mask):
    mask = (mask > 0).astype(np.uint8)
    labeled, num = ndi.label(mask)
    if num == 0:
        return {
            "component_count": 0,
            "mean_component_area": 0.0,
            "max_component_area": 0.0,
            "mean_fill_ratio": 0.0,
            "mean_elongation": 0.0,
            "border_touch_count": 0,
            "line_like_count": 0,
        }

    sizes = []
    fill_ratios = []
    elongations = []
    border_touch_count = 0
    line_like_count = 0
    h, w = mask.shape
    border_band = 12

    for cid in range(1, num + 1):
        comp = labeled == cid
        area = int(comp.sum())
        if area <= 0:
            continue
        ys, xs = np.where(comp)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bh = int(y1 - y0 + 1)
        bw = int(x1 - x0 + 1)
        bbox_area = float(max(1, bh * bw))
        fill_ratio = float(area) / bbox_area
        elongation = float(max(bh, bw)) / float(max(1, min(bh, bw)))
        border_touch = int(y0 < border_band or x0 < border_band or y1 >= h - border_band or x1 >= w - border_band)
        line_like = int(fill_ratio < 0.16 and elongation > 3.6)

        sizes.append(float(area))
        fill_ratios.append(fill_ratio)
        elongations.append(elongation)
        border_touch_count += border_touch
        line_like_count += line_like

    return {
        "component_count": int(num),
        "mean_component_area": float(np.mean(sizes)) if sizes else 0.0,
        "max_component_area": float(np.max(sizes)) if sizes else 0.0,
        "mean_fill_ratio": float(np.mean(fill_ratios)) if fill_ratios else 0.0,
        "mean_elongation": float(np.mean(elongations)) if elongations else 0.0,
        "border_touch_count": int(border_touch_count),
        "line_like_count": int(line_like_count),
    }


def border_fp_ratio(pred, gt, border_band=8):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    h, w = pred.shape
    border = np.zeros((h, w), dtype=np.uint8)
    b = int(max(1, border_band))
    border[:b, :] = 1
    border[-b:, :] = 1
    border[:, :b] = 1
    border[:, -b:] = 1
    fp_border = int(((pred == 1) & (gt == 0) & (border == 1)).sum())
    return safe_div(fp_border, int(border.sum()))


def infer_error_type(row):
    gt_ratio = float(row["gt_ratio"])
    pred_ratio = float(row["pred_ratio"])
    dice = float(row["dice"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    component_count = float(row["component_count"])
    line_like_count = float(row["line_like_count"])
    mean_elongation = float(row["mean_elongation"])
    building_support_mean = float(row["building_support_mean"])
    temporal_delta_mean = float(row["temporal_delta_mean"])
    fp_border = float(row["fp_border_ratio"])

    if gt_ratio == 0.0 and pred_ratio > 0.0:
        if line_like_count >= 1 or mean_elongation > 3.0 or fp_border > 0.01:
            return "rural_linear_false_positive"
        if building_support_mean < 0.25 and temporal_delta_mean < 0.20:
            return "diffuse_rural_false_positive"
        return "nochange_false_positive"

    if gt_ratio > 0.0 and pred_ratio == 0.0:
        return "missed_change"

    if dice == 0.0 and pred_ratio > 0.0 and gt_ratio > 0.0:
        return "wrong_location_or_heavy_false_positive"

    if recall < 0.45 and precision >= 0.60:
        return "under_detection"

    if precision < 0.45 and recall >= 0.60:
        if component_count > 20:
            return "over_detection_fragmented"
        return "over_detection"

    if component_count > 40 and line_like_count >= 2:
        return "fragmented_linear_noise"

    return "mixed_or_ok"


def load_npz_records(input_dir, pred_key):
    npz_dir = Path(input_dir) / "npz"
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"NPZ folder not found: {npz_dir}")

    records = []
    for npz_path in sorted(npz_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        patch_id = str(data["patch_id"]) if "patch_id" in data else npz_path.stem
        gt = data["gt_mask"].astype(np.uint8)
        pred = data[pred_key].astype(np.uint8)
        prob = data["prob_map"].astype(np.float32)
        global_map = data["global_prob_map"].astype(np.float32)
        building_support = data["building_support"].astype(np.float32)
        temporal_delta = data["temporal_delta"].astype(np.float32)

        m = binary_metrics(pred, gt)
        c = component_stats(pred)

        row = {
            "patch_id": patch_id,
            "gt_ratio": float(gt.mean()),
            "pred_ratio": float(pred.mean()),
            "abs_ratio_gap": abs(float(pred.mean()) - float(gt.mean())),
            "prob_mean": float(prob.mean()),
            "prob_peak": float(np.quantile(prob, 0.997)),
            "global_mean": float(global_map.mean()),
            "building_support_mean": float(building_support[pred > 0].mean()) if np.any(pred > 0) else float(building_support.mean()),
            "temporal_delta_mean": float(temporal_delta[pred > 0].mean()) if np.any(pred > 0) else float(temporal_delta.mean()),
            "fp_border_ratio": float(border_fp_ratio(pred, gt, border_band=8)),
            **m,
            **c,
        }
        row["error_type"] = infer_error_type(row)
        records.append(row)

    return records


def sort_worst(rows):
    return sorted(
        rows,
        key=lambda r: (
            float(r["dice"]),
            float(r["iou"]),
            -float(r["abs_ratio_gap"]),
            -float(r["component_count"]),
            -float(r["line_like_count"]),
            str(r["patch_id"]),
        )
    )


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_txt(path, rows, pred_key, top_k):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    worst = sort_worst(rows)[:top_k]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Analysis for pred_key={pred_key}\n")
        f.write(f"Total patches: {len(rows)}\n\n")

        if rows:
            etypes = {}
            for r in rows:
                etypes[r["error_type"]] = etypes.get(r["error_type"], 0) + 1
            f.write("Error type counts:\n")
            for k, v in sorted(etypes.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write(f"- {k}: {v}\n")

            f.write("\nWorst patches:\n")
            for i, r in enumerate(worst, start=1):
                f.write(
                    f"{i:2d}. {r['patch_id']} | dice={r['dice']:.6f} | iou={r['iou']:.6f} | "
                    f"precision={r['precision']:.6f} | recall={r['recall']:.6f} | "
                    f"pred_ratio={r['pred_ratio']:.6f} | gt_ratio={r['gt_ratio']:.6f} | "
                    f"components={r['component_count']} | line_like={r['line_like_count']} | "
                    f"fp_border={r['fp_border_ratio']:.6f} | build_mean={r['building_support_mean']:.6f} | "
                    f"delta_mean={r['temporal_delta_mean']:.6f} | type={r['error_type']}\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--pred_key", type=str, default="pred_mask_fused_semantic_stable_precision_strict")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--summary_txt", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    rows = load_npz_records(args.input_dir, args.pred_key)
    rows_sorted = sort_worst(rows)
    write_csv(args.output_csv, rows_sorted)
    write_summary_txt(args.summary_txt, rows_sorted, args.pred_key, int(args.top_k))

    print(f"[INFO] Analysis CSV saved to: {args.output_csv}")
    print(f"[INFO] Summary TXT saved to: {args.summary_txt}")


if __name__ == "__main__":
    main()
