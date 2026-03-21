import os
import argparse
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"outputs\change_maps_cleaned_v10_full\npz",
        help="Directory containing exported .npz files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"outputs\change_maps_cleaned_v10_full\change_map_summary.csv",
        help="Where to save the CSV summary",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Directory not found: {args.input_dir}")

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".npz")]
    files.sort()

    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {args.input_dir}")

    rows = []

    for fname in files:
        path = os.path.join(args.input_dir, fname)
        d = np.load(path, allow_pickle=True)

        patch_id = str(d["patch_id"]) if "patch_id" in d else fname

        row = {
            "patch_id": patch_id,
            "pred_fixed_ratio": float(d["pred_fixed_ratio"]) if "pred_fixed_ratio" in d else np.nan,
            "pred_adapt_ratio": float(d["pred_adapt_ratio"]) if "pred_adapt_ratio" in d else np.nan,
            "pred_adapt_norm_raw_ratio": float(d["pred_adapt_norm_raw_ratio"]) if "pred_adapt_norm_raw_ratio" in d else np.nan,
            "pred_adapt_norm_clean_ratio": float(d["pred_adapt_norm_clean_ratio"]) if "pred_adapt_norm_clean_ratio" in d else np.nan,
            "pseudo_ratio": float(d["pseudo_ratio"]) if "pseudo_ratio" in d else np.nan,
            "confident_ratio": float(d["confident_ratio"]) if "confident_ratio" in d else np.nan,
        }

        row["abs_diff_clean_vs_pseudo"] = abs(row["pred_adapt_norm_clean_ratio"] - row["pseudo_ratio"])
        row["abs_diff_clean_vs_confident"] = abs(row["pred_adapt_norm_clean_ratio"] - row["confident_ratio"])

        rows.append(row)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n[INFO] Summary saved to:")
    print(args.output_csv)

    print("\n[INFO] Number of samples:")
    print(len(df))

    print("\n[INFO] Global stats:")
    print(df.describe(include="all"))

    print("\n[INFO] Top 10 highest predicted clean ratios:")
    print(df.sort_values("pred_adapt_norm_clean_ratio", ascending=False).head(10)[
        ["patch_id", "pred_adapt_norm_clean_ratio", "pseudo_ratio", "confident_ratio"]
    ].to_string(index=False))

    print("\n[INFO] Top 10 lowest predicted clean ratios:")
    print(df.sort_values("pred_adapt_norm_clean_ratio", ascending=True).head(10)[
        ["patch_id", "pred_adapt_norm_clean_ratio", "pseudo_ratio", "confident_ratio"]
    ].to_string(index=False))

    print("\n[INFO] Top 10 closest clean-vs-pseudo:")
    print(df.sort_values("abs_diff_clean_vs_pseudo", ascending=True).head(10)[
        ["patch_id", "pred_adapt_norm_clean_ratio", "pseudo_ratio", "abs_diff_clean_vs_pseudo"]
    ].to_string(index=False))

    print("\n[INFO] Top 10 farthest clean-vs-pseudo:")
    print(df.sort_values("abs_diff_clean_vs_pseudo", ascending=False).head(10)[
        ["patch_id", "pred_adapt_norm_clean_ratio", "pseudo_ratio", "abs_diff_clean_vs_pseudo"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()