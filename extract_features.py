import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from IEFT.modules import ViLTransformerSS
from IEFT.datamodules.s2_npz_datamodule import S2NPZDataModule


def move_to_device(obj, device):
    """Recursively move tensors/lists/dicts to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def pick_feature_from_infer_output(out_dict):
    """
    Try several common keys used by ViLT-like models.
    Returns a [B, D] tensor.
    """
    if not isinstance(out_dict, dict):
        raise TypeError(f"infer() must return a dict, got {type(out_dict)}")

    # Best case: global CLS-like embedding already exposed
    for key in ["cls_feats", "cls_feat", "raw_cls_feats"]:
        if key in out_dict:
            feat = out_dict[key]
            if feat.ndim == 2:
                return feat
            if feat.ndim == 3:
                return feat[:, 0, :]

    # Fallback: token-level image features
    if "image_feats" in out_dict:
        feat = out_dict["image_feats"]
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            # usually token 0 is CLS-like/global token
            return feat[:, 0, :]

    # Fallback: text feats if image feats not available
    if "text_feats" in out_dict:
        feat = out_dict["text_feats"]
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            return feat[:, 0, :]

    raise KeyError(
        f"Could not find a usable feature key in infer() output. Keys: {list(out_dict.keys())}"
    )


def run_infer(model, batch):
    """
    Tries a few infer() signatures commonly seen in ViLT-style repos.
    """
    try:
        return model.infer(batch, mask_text=False, mask_image=False)
    except TypeError:
        pass

    try:
        return model.infer(batch)
    except TypeError:
        pass

    # last resort: some repos use forward() directly
    out = model(batch)
    if isinstance(out, dict):
        return out

    raise RuntimeError("Could not call model.infer(batch) or model(batch) successfully.")


def build_config_from_checkpoint(ckpt, args):
    """
    Recover training config from Lightning checkpoint, then override a few fields
    for feature extraction.
    """
    cfg = {}

    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, dict):
        if "config" in hp and isinstance(hp["config"], dict):
            cfg = dict(hp["config"])
        else:
            cfg = dict(hp)

    # Required overrides / safe defaults
    cfg["data_root"] = args.data_root
    cfg["batch_size"] = args.batch_size
    cfg["per_gpu_batchsize"] = args.batch_size
    cfg["num_workers"] = args.num_workers
    cfg["image_size"] = args.image_size
    cfg["max_text_len"] = int(cfg.get("max_text_len", 40))
    cfg["tokenizer"] = cfg.get("tokenizer", "bert-base-uncased")

    # Disable negatives for extraction
    cfg["draw_false_image"] = 0
    cfg["draw_false_text"] = 0

    # Safe defaults used by module init
    cfg["num_gpus"] = 1 if torch.cuda.is_available() else 0
    cfg["num_nodes"] = 1
    cfg["precision"] = 16 if torch.cuda.is_available() else 32
    cfg["fast_dev_run"] = False
    cfg["resume_from"] = None
    cfg["test_only"] = False
    cfg["get_recall_metric"] = False

    return cfg


def get_loader(dm, split):
    if split == "train":
        return dm.train_dataloader()
    if split == "val":
        return dm.val_dataloader()
    if split == "test":
        return dm.test_dataloader()
    raise ValueError(f"Unknown split: {split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to last.ckpt",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_npz",
        help="Root folder containing train/ val/ test shards",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to extract features from",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Feature extraction batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=384,
        help="Must match your stable training setup",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="Limit number of samples for quick tests (-1 = all)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=r"features\test_features.npz",
        help="Output .npz file",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # 1) Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    cfg = build_config_from_checkpoint(ckpt, args)

    # 2) Build datamodule
    dm = S2NPZDataModule(cfg)
    dm.setup("fit")
    loader = get_loader(dm, args.split)

    # 3) Build model and load checkpoint weights
    model = ViLTransformerSS(cfg)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] Missing keys: {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    all_feats = []
    all_patch_ids = []

    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {args.split} features"):
            batch_device = move_to_device(batch, device)

            out = run_infer(model, batch_device)
            feats = pick_feature_from_infer_output(out)  # [B, D]

            feats = feats.detach().cpu().float().numpy()
            files = batch["file"]  # keep original CPU-side strings

            all_feats.append(feats)
            all_patch_ids.extend([str(f) for f in files])

            count += len(files)
            if args.max_items > 0 and count >= args.max_items:
                break

    if len(all_feats) == 0:
        raise RuntimeError("No features were extracted.")

    features = np.concatenate(all_feats, axis=0)

    # If max_items cut the loop after overshooting by a batch
    if args.max_items > 0:
        features = features[:args.max_items]
        all_patch_ids = all_patch_ids[:args.max_items]

    np.savez_compressed(
        args.out_path,
        features=features.astype(np.float32),
        patch_ids=np.array(all_patch_ids, dtype=object),
        split=args.split,
        ckpt_path=args.ckpt_path,
        feature_dim=features.shape[1],
    )

    print("\n[INFO] Done.")
    print(f"[INFO] features shape = {features.shape}")
    print(f"[INFO] saved to      = {args.out_path}")


if __name__ == "__main__":
    main()
