import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def compute_binary_metrics(pred, ref):
    pred = (pred > 0).astype(np.uint8)
    ref = (ref > 0).astype(np.uint8)
    tp = int(((pred == 1) & (ref == 1)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    tn = int(((pred == 0) & (ref == 0)).sum())
    empty_empty = (tp == 0 and fp == 0 and fn == 0)
    if empty_empty:
        iou = dice = f1 = precision = recall = accuracy = 1.0
    else:
        iou = safe_div(tp, tp + fp + fn)
        dice = safe_div(2 * tp, 2 * tp + fp + fn)
        f1 = dice
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "iou": iou,
        "dice": dice,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "empty_empty_match": 1 if empty_empty else 0,
    }


def list_npz_files(folder):
    npz_dir = os.path.join(folder, "npz")
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"NPZ folder not found: {npz_dir}")
    return sorted([os.path.join(npz_dir, n) for n in os.listdir(npz_dir) if n.lower().endswith(".npz")])


def summarize(values):
    vals = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "std": float(vals.std()) if len(vals) else 0.0,
        "min": float(vals.min()) if len(vals) else 0.0,
        "median": float(np.median(vals)) if len(vals) else 0.0,
        "max": float(vals.max()) if len(vals) else 0.0,
    }


def component_stats(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    labeled, num = ndi.label(mask)
    if num == 0:
        return 0, 0.0, 0.0
    sizes = np.asarray(ndi.sum(mask, labeled, index=np.arange(1, num + 1)), dtype=np.float64)
    return int(num), float(sizes.mean()), float(sizes.max())


def save_summary_plot(summary_dict, out_path):
    metrics = ["iou", "dice", "f1", "precision", "recall", "accuracy"]
    means = [summary_dict[m]["mean"] for m in metrics]
    mins = [summary_dict[m]["min"] for m in metrics]
    maxs = [summary_dict[m]["max"] for m in metrics]
    x = np.arange(len(metrics))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means)
    plt.scatter(x, mins, marker="_")
    plt.scatter(x, maxs, marker="_")
    plt.xticks(x, [m.upper() for m in metrics])
    plt.ylim(0, 1.02)
    plt.title("Final metrics summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--sort_metric", type=str, default="f1", choices=["iou", "dice", "f1", "precision", "recall", "accuracy"])
    return p.parse_args()


def main():
    args = parse_args()
    files = list_npz_files(args.input_dir)
    rows = []
    for fp in files:
        data = np.load(fp, allow_pickle=True)
        patch_id = str(data.get("patch_id", Path(fp).stem))
        ref = (data["ref_mask"] > 0).astype(np.uint8)
        pred = (data["pred_mask_final"] > 0).astype(np.uint8)
        metrics = compute_binary_metrics(pred, ref)
        comp_count, comp_mean, comp_max = component_stats(pred)
        row = {
            "patch_id": patch_id,
            **metrics,
            "pred_component_count": comp_count,
            "pred_mean_component_area": comp_mean,
            "pred_max_component_area": comp_max,
        }
        rows.append(row)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["patch_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {m: summarize([r[m] for r in rows]) for m in ["iou", "dice", "f1", "precision", "recall", "accuracy"]}
    summary_path = os.path.splitext(args.output_csv)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FINAL METRICS SUMMARY\n")
        f.write("=====================\n")
        for m, s in summary.items():
            f.write(f"{m.upper():10s} mean={s['mean']:.6f}  std={s['std']:.6f}  min={s['min']:.6f}  median={s['median']:.6f}  max={s['max']:.6f}\n")
        empty_rate = np.mean([r["empty_empty_match"] for r in rows]) if rows else 0.0
        f.write(f"\nEMPTY_EMPTY_MATCH_RATE={empty_rate:.6f}\n")

    rank_sorted = sorted(
        rows,
        key=lambda r: (
            -float(r.get(args.sort_metric, 0.0)),
            -float(r.get("iou", 0.0)),
            -float(r.get("dice", 0.0)),
            str(r.get("patch_id", "")),
        ),
    )
    best = rank_sorted[:args.top_k]
    worst = list(reversed(rank_sorted[-args.top_k:])) if rank_sorted else []
    rank_csv = os.path.splitext(args.output_csv)[0] + "_top_bottom.csv"
    with open(rank_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["section", "rank"] + list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(best, 1):
            writer.writerow({"section": "best", "rank": i, **r})
        for i, r in enumerate(worst, 1):
            writer.writerow({"section": "worst", "rank": i, **r})

    plot_path = os.path.splitext(args.output_csv)[0] + "_summary.png"
    save_summary_plot(summary, plot_path)

    print(f"[INFO] Per-patch final metrics saved to: {args.output_csv}")
    print(f"[INFO] Final summary TXT saved to: {summary_path}")
    print(f"[INFO] Final summary plot saved to: {plot_path}")
    print("[INFO] Final metrics:")
    for m, s in summary.items():
        print(f"  {m.upper():10s} mean={s['mean']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}")
    if best:
        print(f"[INFO] Best patch by {args.sort_metric}: {best[0]['patch_id']} {args.sort_metric}={best[0][args.sort_metric]:.6f}")
    if worst:
        print(f"[INFO] Worst patch by {args.sort_metric}: {worst[0]['patch_id']} {args.sort_metric}={worst[0][args.sort_metric]:.6f}")


if __name__ == "__main__":
    main()
