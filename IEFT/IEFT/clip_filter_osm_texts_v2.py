import os
import json
import argparse
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def clean_text(x: Any) -> str:
    s = str(x).strip()
    s = " ".join(s.split())
    return s


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


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


def load_osm_json_raw(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"OSM JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("OSM JSON must be a dict: patch_id -> record")

    return {str(k): v for k, v in data.items()}


def extract_candidate_texts(record: Any) -> List[str]:
    """
    Retourne les phrases candidates à scorer avec CLIP.
    - si record est str -> [str]
    - si record est list -> list[str]
    - si record est dict -> priorité à record['phrases']
      sinon fallback sur text_v12 / summary / source_text
    """
    if isinstance(record, str):
        texts = [clean_text(record)]
        return [t for t in texts if t]

    if isinstance(record, list):
        texts = [clean_text(x) for x in record if clean_text(x)]
        return dedup_keep_order(texts)

    if isinstance(record, dict):
        phrases = record.get("phrases", [])
        if isinstance(phrases, str):
            phrases = [phrases]
        elif not isinstance(phrases, list):
            phrases = []

        texts = [clean_text(x) for x in phrases if clean_text(x)]
        texts = dedup_keep_order(texts)

        if len(texts) > 0:
            return texts

        fallback = []
        for key in ["text_v12", "summary", "source_text"]:
            v = clean_text(record.get(key, ""))
            if v:
                fallback.append(v)

        return dedup_keep_order(fallback)

    return []


def build_filtered_record(original_record: Any, kept_texts: List[str]) -> Any:
    """
    On garde le format le plus compatible possible avec le dataset.
    - str  -> str
    - list -> list[str]
    - dict -> on conserve tout et on remplace 'phrases'
    """
    if isinstance(original_record, str):
        return kept_texts[0] if len(kept_texts) > 0 else ""

    if isinstance(original_record, list):
        return kept_texts

    if isinstance(original_record, dict):
        out = dict(original_record)
        out["phrases"] = kept_texts
        out["clip_filtered_phrases"] = kept_texts
        out["clip_num_kept"] = len(kept_texts)
        return out

    return kept_texts


def x8_to_rgb_uint8(x8: np.ndarray, which: str = "T2", s2_scale_div: float = 10000.0) -> np.ndarray:
    """
    x8 shape: [8, H, W]
    Order:
      T1_B2,T1_B3,T1_B4,T1_B8,T2_B2,T2_B3,T2_B4,T2_B8
    RGB = [B4, B3, B2]
    """
    if x8.ndim != 3 or x8.shape[0] != 8:
        raise ValueError(f"Expected x8 shape [8,H,W], got {x8.shape}")

    which = which.upper()
    if which == "T1":
        b2, b3, b4 = x8[0], x8[1], x8[2]
    else:
        b2, b3, b4 = x8[4], x8[5], x8[6]

    rgb = np.stack([b4, b3, b2], axis=-1).astype(np.float32) / float(s2_scale_div)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).round().astype(np.uint8)
    return rgb


def x8_to_absdiff_rgb_uint8(x8: np.ndarray, s2_scale_div: float = 10000.0) -> np.ndarray:
    t1 = x8_to_rgb_uint8(x8, which="T1", s2_scale_div=s2_scale_div).astype(np.float32) / 255.0
    t2 = x8_to_rgb_uint8(x8, which="T2", s2_scale_div=s2_scale_div).astype(np.float32) / 255.0
    d = np.abs(t2 - t1)
    d = np.clip(d, 0.0, 1.0)
    d = (d * 255.0).round().astype(np.uint8)
    return d


def build_clip_image_from_x8(
    x8: np.ndarray,
    s2_scale_div: float = 10000.0,
    mode: str = "triplet",
    out_size: int = 224,
) -> Image.Image:
    """
    mode:
      - t2      : CLIP voit seulement T2
      - t1      : CLIP voit seulement T1
      - triplet : [T1 | |T2-T1| | T2] -> mieux pour le changement
    """
    mode = str(mode).strip().lower()

    if mode == "t1":
        arr = x8_to_rgb_uint8(x8, which="T1", s2_scale_div=s2_scale_div)
        return Image.fromarray(arr, mode="RGB").resize((out_size, out_size), Image.BICUBIC)

    if mode == "t2":
        arr = x8_to_rgb_uint8(x8, which="T2", s2_scale_div=s2_scale_div)
        return Image.fromarray(arr, mode="RGB").resize((out_size, out_size), Image.BICUBIC)

    t1 = x8_to_rgb_uint8(x8, which="T1", s2_scale_div=s2_scale_div)
    diff = x8_to_absdiff_rgb_uint8(x8, s2_scale_div=s2_scale_div)
    t2 = x8_to_rgb_uint8(x8, which="T2", s2_scale_div=s2_scale_div)

    canvas = np.concatenate([t1, diff, t2], axis=1)
    return Image.fromarray(canvas, mode="RGB").resize((out_size, out_size), Image.BICUBIC)


def collect_patch_images_from_npz(
    npz_root: str,
    s2_scale_div: float,
    clip_image_mode: str,
    clip_image_size: int,
) -> Dict[str, Image.Image]:
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
            image = build_clip_image_from_x8(
                x8=x8,
                s2_scale_div=s2_scale_div,
                mode=clip_image_mode,
                out_size=clip_image_size,
            )
            patch_to_image[pid] = image

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


def prepare_local_clip_repo(repo_id: str, local_model_dir: str, force_download: bool = False) -> str:
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

    print("[INFO] Downloading only required CLIP files.")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_model_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
    )

    if not has_required_clip_files(local_model_dir):
        raise RuntimeError(f"Local CLIP directory incomplete: {local_model_dir}")

    return local_model_dir


def load_clip_from_local(local_model_dir: str, device: torch.device):
    processor = CLIPProcessor.from_pretrained(local_model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(local_model_dir, local_files_only=True)
    model = model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def score_patch_texts_fast(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    texts: List[str],
    device: torch.device,
    text_batch_size: int = 32,
) -> List[float]:
    if len(texts) == 0:
        return []

    img_inputs = processor(images=image, return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(device)

    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logit_scale = model.logit_scale.exp().detach()

    scores_all: List[float] = []

    for start in range(0, len(texts), text_batch_size):
        batch_texts = texts[start:start + text_batch_size]

        txt_inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

        text_features = model.get_text_features(**txt_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * (image_features @ text_features.T)  # [1, Bt]
        scores = logits.squeeze(0).detach().cpu().float().tolist()

        if isinstance(scores, float):
            scores = [scores]

        scores_all.extend([float(s) for s in scores])

    return scores_all


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

    parser.add_argument("--npz_root", type=str, default=r"data_npz")
    parser.add_argument("--input_osm_json", type=str, required=True)
    parser.add_argument("--output_osm_json", type=str, required=True)
    parser.add_argument("--output_debug_json", type=str, required=True)

    parser.add_argument("--hf_repo_id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--local_model_dir", type=str, default=r"models\clip-vit-base-patch32")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_patches", type=int, default=-1)
    parser.add_argument("--s2_scale_div", type=float, default=10000.0)

    parser.add_argument("--clip_image_mode", type=str, default="triplet", choices=["t1", "t2", "triplet"])
    parser.add_argument("--clip_image_size", type=int, default=224)
    parser.add_argument("--text_batch_size", type=int, default=32)

    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")

    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    local_model_dir = prepare_local_clip_repo(
        repo_id=args.hf_repo_id,
        local_model_dir=args.local_model_dir,
        force_download=args.force_download,
    )
    print(f"[INFO] CLIP local dir ready: {local_model_dir}")

    if args.prepare_only:
        print("[INFO] prepare_only=True -> stopping here.")
        return

    print("[1] Loading raw OSM JSON")
    osm_raw = load_osm_json_raw(args.input_osm_json)
    print(f"[INFO] patches in OSM JSON = {len(osm_raw)}")

    print("[2] Building CLIP images from NPZ")
    patch_to_image = collect_patch_images_from_npz(
        npz_root=args.npz_root,
        s2_scale_div=args.s2_scale_div,
        clip_image_mode=args.clip_image_mode,
        clip_image_size=args.clip_image_size,
    )
    print(f"[INFO] patches with images = {len(patch_to_image)}")

    common_patch_ids = sorted(set(osm_raw.keys()) & set(patch_to_image.keys()))
    if len(common_patch_ids) == 0:
        raise RuntimeError("No common patch_id between NPZ data and OSM JSON.")

    if args.max_patches > 0:
        common_patch_ids = common_patch_ids[:args.max_patches]

    print(f"[INFO] common patches to process = {len(common_patch_ids)}")

    print("[3] Loading CLIP")
    processor, model = load_clip_from_local(local_model_dir=local_model_dir, device=device)

    filtered_by_patch: Dict[str, Any] = {}
    debug_by_patch: Dict[str, Any] = {}

    print("[4] CLIP filtering")
    for patch_id in tqdm(common_patch_ids, desc="CLIP filtering"):
        raw_record = osm_raw.get(patch_id)
        image = patch_to_image[patch_id]

        candidate_texts = extract_candidate_texts(raw_record)

        if len(candidate_texts) == 0:
            filtered_by_patch[patch_id] = build_filtered_record(raw_record, [])
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

        scores = score_patch_texts_fast(
            model=model,
            processor=processor,
            image=image,
            texts=candidate_texts,
            device=device,
            text_batch_size=args.text_batch_size,
        )

        kept_texts, kept_scores, ranked_pairs = topk_texts_with_scores(
            texts=candidate_texts,
            scores=scores,
            top_k=args.top_k,
        )

        filtered_by_patch[patch_id] = build_filtered_record(raw_record, kept_texts)
        debug_by_patch[patch_id] = {
            "num_candidates": len(candidate_texts),
            "num_kept": len(kept_texts),
            "all_texts": candidate_texts,
            "all_scores": [float(s) for s in scores],
            "ranked_texts_with_scores": [{"text": t, "score": float(s)} for t, s in ranked_pairs],
            "kept_texts": kept_texts,
            "kept_scores": kept_scores,
        }

    print("[5] Saving JSON")
    save_json(args.output_osm_json, filtered_by_patch)
    save_json(args.output_debug_json, debug_by_patch)

    print("[INFO] Done")
    print(f"[INFO] output_osm_json   = {args.output_osm_json}")
    print(f"[INFO] output_debug_json = {args.output_debug_json}")


if __name__ == "__main__":
    main()