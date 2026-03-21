import os
import argparse
import numpy as np
import pandas as pd


def load_npz_summary(npz_dir: str, prefix: str):
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f"Directory not found: {npz_dir}")

    files = [f for f in os.listdir(npz_dir) if f.lower().endswith(".npz")]
    files.sort()

    rows = []
    for fname in files:
        path = os.path.join(npz_dir, fname)
        d = np.load(path, allow_pickle=True)

        patch_id = str(d["patch_id"]) if "patch_id" in d else os.path.splitext(fname)[0]

        row = {
            "patch_id": patch_id,
            f"{prefix}_pred_fixed_ratio": float(d["pred_fixed_ratio"]) if "pred_fixed_ratio" in d else np.nan,
            f"{prefix}_pred_adapt_ratio": float(d["pred_adapt_ratio"]) if "pred_adapt_ratio" in d else np.nan,
            f"{prefix}_pred_adapt_norm_raw_ratio": float(d["pred_adapt_norm_raw_ratio"]) if "pred_adapt_norm_raw_ratio" in d else np.nan,
            f"{prefix}_pred_adapt_norm_clean_ratio": float(d["pred_adapt_norm_clean_ratio"]) if "pred_adapt_norm_clean_ratio" in d else np.nan,
            f"{prefix}_pseudo_ratio": float(d["pseudo_ratio"]) if "pseudo_ratio" in d else np.nan,
            f"{prefix}_confident_ratio": float(d["confident_ratio"]) if "confident_ratio" in d else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10_dir", type=str, required=True, help="NPZ dir for baseline v10")
    parser.add_argument("--v11_clip_dir", type=str, required=True, help="NPZ dir for new model")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    df10 = load_npz_summary(args.v10_dir, "v10")
    df11 = load_npz_summary(args.v11_clip_dir, "v11_clip")

    df = pd.merge(df10, df11, on="patch_id", how="inner")

    if len(df) == 0:
        raise RuntimeError("No common patch_id found between the two exports.")

    df["delta_clip_vs_v10"] = (
        df["v11_clip_pred_adapt_norm_clean_ratio"]
        - df["v10_pred_adapt_norm_clean_ratio"]
    )
    df["abs_delta_clip_vs_v10"] = df["delta_clip_vs_v10"].abs()

    df["v10_abs_diff_to_pseudo"] = (
        df["v10_pred_adapt_norm_clean_ratio"] - df["v10_pseudo_ratio"]
    ).abs()

    df["v11_clip_abs_diff_to_pseudo"] = (
        df["v11_clip_pred_adapt_norm_clean_ratio"] - df["v11_clip_pseudo_ratio"]
    ).abs()

    df["improve_clip_vs_v10_to_pseudo"] = (
        df["v10_abs_diff_to_pseudo"] - df["v11_clip_abs_diff_to_pseudo"]
    )

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print("\n[INFO] Comparison CSV saved to:")
    print(args.output_csv)

    print("\n[INFO] Number of common samples:")
    print(len(df))

    print("\n[INFO] Global comparison stats:")
    print(df[[
        "v10_pred_adapt_norm_clean_ratio",
        "v11_clip_pred_adapt_norm_clean_ratio",
        "delta_clip_vs_v10",
        "v10_abs_diff_to_pseudo",
        "v11_clip_abs_diff_to_pseudo",
        "improve_clip_vs_v10_to_pseudo",
    ]].describe())

    print("\n[INFO] Top 10 patches where v11_clip predicts MORE change than v10:")
    print(
        df.sort_values("delta_clip_vs_v10", ascending=False).head(10)[
            [
                "patch_id",
                "v10_pred_adapt_norm_clean_ratio",
                "v11_clip_pred_adapt_norm_clean_ratio",
                "delta_clip_vs_v10",
            ]
        ].to_string(index=False)
    )

    print("\n[INFO] Top 10 patches where v11_clip predicts LESS change than v10:")
    print(
        df.sort_values("delta_clip_vs_v10", ascending=True).head(10)[
            [
                "patch_id",
                "v10_pred_adapt_norm_clean_ratio",
                "v11_clip_pred_adapt_norm_clean_ratio",
                "delta_clip_vs_v10",
            ]
        ].to_string(index=False)
    )

    print("\n[INFO] Top 10 patches where v11_clip moved CLOSER to pseudo than v10:")
    print(
        df.sort_values("improve_clip_vs_v10_to_pseudo", ascending=False).head(10)[
            [
                "patch_id",
                "v10_abs_diff_to_pseudo",
                "v11_clip_abs_diff_to_pseudo",
                "improve_clip_vs_v10_to_pseudo",
            ]
        ].to_string(index=False)
    )

    print("\n[INFO] Top 10 patches where v11_clip moved FARTHER from pseudo than v10:")
    print(
        df.sort_values("improve_clip_vs_v10_to_pseudo", ascending=True).head(10)[
            [
                "patch_id",
                "v10_abs_diff_to_pseudo",
                "v11_clip_abs_diff_to_pseudo",
                "improve_clip_vs_v10_to_pseudo",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()