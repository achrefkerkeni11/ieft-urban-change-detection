import os
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def list_npz_files(npz_root: str) -> List[str]:
    files = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(npz_root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            if name.lower().endswith(".npz"):
                files.append(os.path.join(split_dir, name))
    files.sort()
    return files


def load_osm_json(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"OSM JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("OSM JSON must be a dict: patch_id -> list[str]")

    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        patch_id = str(k)

        if isinstance(v, str):
            texts = [v.strip()] if v.strip() else []
        elif isinstance(v, list):
            texts = [str(x).strip() for x in v if str(x).strip()]
        else:
            texts = []

        out[patch_id] = texts

    return out


def x8_to_t2_rgb_uint8(x8: np.ndarray, s2_scale_div: float = 10000.0) -> np.ndarray:
    """
    x8 shape: [8, H, W]
    Order:
      T1_B2,T1_B3,T1_B4,T1_B8,T2_B2,T2_B3,T2_B4,T2_B8
    Build RGB from T2: [B4, B3, B2]
    """
    if x8.ndim != 3 or x8.shape[0] != 8:
        raise ValueError(f"Expected x8 shape [8,H,W], got {x8.shape}")

    b2 = x8[4].astype(np.float32)
    b3 = x8[5].astype(np.float32)
    b4 = x8[6].astype(np.float32)

    rgb = np.stack([b4, b3, b2], axis=-1) / float(s2_scale_div)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).round().astype(np.uint8)
    return rgb


def collect_patch_images_from_npz(npz_root: str, s2_scale_div: float) -> Dict[str, Image.Image]:
    """
    Reads all NPZ shards and builds:
      patch_id -> PIL.Image(RGB from T2)
    """
    npz_files = list_npz_files(npz_root)
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No NPZ files found under: {npz_root}")

    patch_to_image: Dict[str, Image.Image] = {}

    for npz_path in tqdm(npz_files, desc="Reading NPZ shards"):
        d = np.load(npz_path, allow_pickle=True)

        if "X" not in d or "files" not in d:
            d.close()
            continue

        X = d["X"]
        files = d["files"]

        if isinstance(files, np.ndarray):
            files = files.tolist()

        for i, patch_id in enumerate(files):
            pid = str(patch_id)
            x8 = X[i]
            rgb = x8_to_t2_rgb_uint8(x8, s2_scale_div=s2_scale_div)
            patch_to_image[pid] = Image.fromarray(rgb, mode="RGB")

        d.close()

    return patch_to_image


def has_required_clip_files(local_model_dir: str) -> bool:
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "pytorch_model.bin",
    ]

    for name in required_files:
        if not os.path.exists(os.path.join(local_model_dir, name)):
            return False

    return True


def prepare_local_clip_repo(
    repo_id: str,
    local_model_dir: str,
    force_download: bool = False,
) -> str:
    """
    Download only the files really needed by CLIPProcessor + CLIPModel (PyTorch).
    This avoids downloading TensorFlow / Flax weights and reduces disk usage.
    """
    ensure_dir(local_model_dir)

    if has_required_clip_files(local_model_dir) and not force_download:
        print(f"[INFO] Reusing local CLIP repo: {local_model_dir}")
        return local_model_dir

    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "pytorch_model.bin",
    ]

    print("[INFO] Downloading only required CLIP files...")
    print(f"[INFO] repo_id         = {repo_id}")
    print(f"[INFO] local_model_dir = {local_model_dir}")
    print(f"[INFO] allow_patterns  = {allow_patterns}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_model_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
    )

    if not has_required_clip_files(local_model_dir):
        raise RuntimeError(
            "Required CLIP files are still missing after download.\n"
            f"Check folder: {local_model_dir}"
        )

    return local_model_dir


def load_clip_from_local(local_model_dir: str, device: torch.device):
    if not has_required_clip_files(local_model_dir):
        raise RuntimeError(
            f"Local CLIP directory is incomplete: {local_model_dir}"
        )

    print("[INFO] Loading CLIP from local directory")
    print(f"[INFO] local_model_dir = {local_model_dir}")

    processor = CLIPProcessor.from_pretrained(
        local_model_dir,
        local_files_only=True,
    )

    model = CLIPModel.from_pretrained(
        local_model_dir,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def score_patch_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    texts: List[str],
    device: torch.device,
) -> List[float]:
    if len(texts) == 0:
        return []

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    scores = logits_per_image.squeeze(0).detach().cpu().float().tolist()

    if isinstance(scores, float):
        scores = [scores]

    return scores


def topk_texts_with_scores(texts: List[str], scores: List[float], top_k: int):
    pairs = list(zip(texts, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    keep_n = max(1, min(int(top_k), len(pairs)))
    pairs = pairs[:keep_n]

    top_texts = [p[0] for p in pairs]
    top_scores = [float(p[1]) for p in pairs]
    return top_texts, top_scores, pairs


def save_json(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npz_root",
        type=str,
        default=r"data_npz",
        help="Root folder with train/val/test NPZ shards",
    )
    parser.add_argument(
        "--input_osm_json",
        type=str,
        default=r"data_osm\osm_texts_by_patch.json",
        help="Patch -> candidate OSM phrases JSON",
    )
    parser.add_argument(
        "--output_osm_json",
        type=str,
        default=r"data_osm\osm_texts_by_patch_clip_topk.json",
        help="Patch -> CLIP-filtered top-k OSM phrases JSON",
    )
    parser.add_argument(
        "--output_debug_json",
        type=str,
        default=r"data_osm\osm_texts_by_patch_clip_topk_debug.json",
        help="Debug JSON with CLIP scores",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face repo id for CLIP",
    )
    parser.add_argument(
        "--local_model_dir",
        type=str,
        default=r"models\clip-vit-base-patch32",
        help="Local folder where the minimal CLIP repo will be stored",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="How many OSM phrases to keep per patch",
    )
    parser.add_argument(
        "--max_patches",
        type=int,
        default=-1,
        help="Limit number of patches for testing; -1 = all",
    )
    parser.add_argument(
        "--s2_scale_div",
        type=float,
        default=10000.0,
        help="Sentinel-2 scale divisor",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download of local CLIP files",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare local CLIP files and stop",
    )

    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    print("[0] Preparing minimal local CLIP repo")
    local_model_dir = prepare_local_clip_repo(
        repo_id=args.hf_repo_id,
        local_model_dir=args.local_model_dir,
        force_download=args.force_download,
    )
    print(f"[INFO] CLIP local dir ready: {local_model_dir}")

    if args.prepare_only:
        print("[INFO] prepare_only=True -> stopping here.")
        return

    print("[1] Loading candidate OSM texts")
    osm_by_patch = load_osm_json(args.input_osm_json)
    print(f"[INFO] patches in OSM JSON = {len(osm_by_patch)}")

    print("[2] Loading patch images from NPZ")
    patch_to_image = collect_patch_images_from_npz(
        npz_root=args.npz_root,
        s2_scale_div=args.s2_scale_div,
    )
    print(f"[INFO] patches with images = {len(patch_to_image)}")

    common_patch_ids = sorted(set(osm_by_patch.keys()) & set(patch_to_image.keys()))
    if len(common_patch_ids) == 0:
        raise RuntimeError("No common patch_id between NPZ data and OSM JSON.")

    if args.max_patches > 0:
        common_patch_ids = common_patch_ids[:args.max_patches]

    print(f"[INFO] common patches to process = {len(common_patch_ids)}")

    print("[3] Loading CLIP locally")
    processor, model = load_clip_from_local(
        local_model_dir=local_model_dir,
        device=device,
    )

    filtered_by_patch: Dict[str, List[str]] = {}
    debug_by_patch: Dict[str, Any] = {}

    print("[4] Scoring OSM texts with CLIP")
    for patch_id in tqdm(common_patch_ids, desc="CLIP filtering"):
        texts = osm_by_patch.get(patch_id, [])
        image = patch_to_image[patch_id]

        if len(texts) == 0:
            filtered_by_patch[patch_id] = []
            debug_by_patch[patch_id] = {
                "num_candidates": 0,
                "num_kept": 0,
                "all_texts": [],
                "all_scores": [],
                "ranked_texts_with_scores": [],
                "kept_texts": [],
                "kept_scores": [],
            }
            continue

        scores = score_patch_texts(
            model=model,
            processor=processor,
            image=image,
            texts=texts,
            device=device,
        )

        kept_texts, kept_scores, ranked_pairs = topk_texts_with_scores(
            texts=texts,
            scores=scores,
            top_k=args.top_k,
        )

        filtered_by_patch[patch_id] = kept_texts
        debug_by_patch[patch_id] = {
            "num_candidates": len(texts),
            "num_kept": len(kept_texts),
            "all_texts": texts,
            "all_scores": [float(s) for s in scores],
            "ranked_texts_with_scores": [
                {"text": t, "score": float(s)} for t, s in ranked_pairs
            ],
            "kept_texts": kept_texts,
            "kept_scores": kept_scores,
        }

    print("[5] Saving outputs")
    save_json(args.output_osm_json, filtered_by_patch)
    save_json(args.output_debug_json, debug_by_patch)

    print("\n[INFO] Done.")
    print(f"[INFO] Filtered JSON: {args.output_osm_json}")
    print(f"[INFO] Debug JSON   : {args.output_debug_json}")


if __name__ == "__main__":
    main()