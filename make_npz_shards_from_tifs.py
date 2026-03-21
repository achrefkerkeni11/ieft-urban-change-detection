import os
import json
import argparse
import random
from typing import List, Dict, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


BANDS = [
    "T1_B2","T1_B3","T1_B4","T1_B8",
    "T2_B2","T2_B3","T2_B4","T2_B8",
]


def list_tifs(input_dir: str) -> List[str]:
    files = []
    for f in os.listdir(input_dir):
        if f.lower().endswith((".tif", ".tiff")):
            files.append(os.path.join(input_dir, f))
    files.sort()
    return files


def ensure_dirs(output_root: str):
    for split in ["train", "val", "test", "meta"]:
        os.makedirs(os.path.join(output_root, split), exist_ok=True)


def split_tifs(tifs: List[str], seed: int, train: float, val: float) -> Dict[str, List[str]]:
    """
    Split au niveau des TIFS (important) :
    - toutes les patches d’une même tuile restent dans le même split
    -> évite fuite train/test.
    """
    rng = random.Random(seed)
    tifs = tifs[:]
    rng.shuffle(tifs)

    n = len(tifs)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    n_train = max(1, n_train)
    n_val = max(0, n_val)
    n_test = max(0, n - n_train - n_val)

    train_list = tifs[:n_train]
    val_list = tifs[n_train:n_train + n_val]
    test_list = tifs[n_train + n_val:]

    # fallback si val/test vides (petit nb de tifs)
    if len(val_list) == 0:
        val_list = train_list
    if len(test_list) == 0:
        test_list = train_list

    return {"train": train_list, "val": val_list, "test": test_list}


def compute_zero_fraction(patch: np.ndarray) -> float:
    """
    patch: [8, ps, ps]
    calcule fraction de pixels où toutes les bandes == 0
    """
    all_zero = np.all(patch == 0, axis=0)  # [ps,ps]
    return float(all_zero.mean())


def flush_shard(out_dir: str, split: str, shard_id: int,
                X_list: List[np.ndarray], files_list: List[str],
                store_dtype: str):
    if len(X_list) == 0:
        return shard_id

    X = np.stack(X_list, axis=0)  # [N,8,ps,ps]
    if store_dtype == "int16":
        X = X.astype(np.int16)
    else:
        X = X.astype(np.float32)

    out_name = f"s2_{split}_{shard_id:04d}.npz"
    out_path = os.path.join(out_dir, split, out_name)

    np.savez_compressed(
        out_path,
        X=X,
        files=np.array(files_list, dtype=object),
        bands=np.array(BANDS, dtype=object),
    )

    X_list.clear()
    files_list.clear()
    return shard_id + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=r"data_raw\tifs", help="dossier contenant les .tif exportés GEE")
    ap.add_argument("--output_root", default=r"data_npz", help="dossier où écrire train/val/test")
    ap.add_argument("--patch_size", type=int, default=128, help="taille patch (px)")
    ap.add_argument("--stride", type=int, default=128, help="stride (px). 128=non overlap, 64=overlap 50%")
    ap.add_argument("--shard_size", type=int, default=512, help="nb patches par .npz (évite fichiers énormes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--store_dtype", choices=["int16", "float32"], default="int16",
                    help="int16 = plus petit sur disque (recommandé)")
    ap.add_argument("--drop_zero_frac_gt", type=float, default=0.60,
                    help="drop patch si > X% des pixels sont 0 sur toutes les bandes")

    args = ap.parse_args()

    ensure_dirs(args.output_root)

    tifs = list_tifs(args.input_dir)
    if len(tifs) == 0:
        raise FileNotFoundError(f"Aucun .tif trouvé dans: {args.input_dir}")

    splits = split_tifs(tifs, args.seed, args.train_ratio, args.val_ratio)

    # Buffers par split
    buffers = {
        "train": {"X": [], "files": [], "shard_id": 0},
        "val":   {"X": [], "files": [], "shard_id": 0},
        "test":  {"X": [], "files": [], "shard_id": 0},
    }

    kept = 0
    dropped = 0

    for split in ["train", "val", "test"]:
        tif_list = splits[split]
        print(f"\n[{split}] nb tifs = {len(tif_list)}")

        for tif_path in tqdm(tif_list, desc=f"Reading {split} tifs"):
            base = os.path.splitext(os.path.basename(tif_path))[0]

            with rasterio.open(tif_path) as ds:
                if ds.count != 8:
                    raise ValueError(f"{tif_path}: attendu 8 bandes, trouvé {ds.count}")

                H, W = ds.height, ds.width
                ps = args.patch_size
                st = args.stride

                if H < ps or W < ps:
                    print(f"[WARN] {tif_path} trop petit ({H}x{W}), skip")
                    continue

                max_r = H - ps
                max_c = W - ps

                for r in range(0, max_r + 1, st):
                    for c in range(0, max_c + 1, st):
                        window = Window(c, r, ps, ps)
                        patch = ds.read(window=window)  # [8,ps,ps]
                        # on garde le type natif (souvent int16) ici
                        patch = patch.astype(np.int32)

                        zf = compute_zero_fraction(patch)
                        if zf > args.drop_zero_frac_gt:
                            dropped += 1
                            continue

                        patch_id = f"{base}_r{r}_c{c}"
                        buffers[split]["X"].append(patch)
                        buffers[split]["files"].append(patch_id)
                        kept += 1

                        if len(buffers[split]["X"]) >= args.shard_size:
                            buffers[split]["shard_id"] = flush_shard(
                                args.output_root, split, buffers[split]["shard_id"],
                                buffers[split]["X"], buffers[split]["files"],
                                store_dtype=args.store_dtype
                            )

        # flush fin split
        buffers[split]["shard_id"] = flush_shard(
            args.output_root, split, buffers[split]["shard_id"],
            buffers[split]["X"], buffers[split]["files"],
            store_dtype=args.store_dtype
        )

    meta = {
        "input_dir": args.input_dir,
        "output_root": args.output_root,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "shard_size": args.shard_size,
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": 1.0 - args.train_ratio - args.val_ratio},
        "bands": BANDS,
        "store_dtype": args.store_dtype,
        "drop_zero_frac_gt": args.drop_zero_frac_gt,
        "kept_patches": kept,
        "dropped_patches": dropped,
        "tifs_per_split": {k: len(v) for k, v in splits.items()},
    }

    meta_path = os.path.join(args.output_root, "meta", "dataset.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDONE")
    print("Kept:", kept, "Dropped:", dropped)
    print("Meta:", meta_path)
    print("Example NPZ:", os.path.join(args.output_root, "train", "s2_train_0000.npz"))


if __name__ == "__main__":
    main()
