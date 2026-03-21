import os
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class S2NPZDataset(Dataset):
    DEFAULT_BANDS = [
        "T1_B2", "T1_B3", "T1_B4", "T1_B8",
        "T2_B2", "T2_B3", "T2_B4", "T2_B8",
    ]

    def __init__(
        self,
        npz_paths: List[str],
        image_size: int = 224,
        tokenizer=None,
        max_text_len: int = 40,
        draw_false_image: int = 1,
        draw_false_text: int = 15,
        rgb_from: str = "T2",
        s2_scale_div: float = 10000.0,
        osm_texts_json: str = "",
        osm_text_mode: str = "concat",
        osm_max_phrases: int = 3,
        osm_text_key: str = "text_v12",
        osm_fallback_text: str = "no_osm_context",
    ):
        super().__init__()
        if not npz_paths:
            raise ValueError("npz_paths est vide")

        self.npz_paths = [p for p in npz_paths if os.path.exists(p)]
        if len(self.npz_paths) == 0:
            raise FileNotFoundError(f"Aucun NPZ valide: {npz_paths}")

        self.image_size = int(image_size)
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)
        self.draw_false_image = int(draw_false_image)
        self.draw_false_text = int(draw_false_text)
        self.rgb_from = str(rgb_from)
        self.s2_scale_div = float(s2_scale_div)

        self.osm_texts_json = str(osm_texts_json).strip()
        self.osm_text_mode = str(osm_text_mode).strip().lower()
        self.osm_max_phrases = max(1, int(osm_max_phrases))
        self.osm_text_key = str(osm_text_key).strip() or "text_v12"
        self.osm_fallback_text = str(osm_fallback_text).strip() or "no_osm_context"

        self._lengths = []
        self._cum = []
        self.bands = self.DEFAULT_BANDS

        self.osm_texts_by_patch: Dict[str, List[str]] = {}
        self.osm_main_text_by_patch: Dict[str, str] = {}
        self.osm_negative_text_pool: List[str] = []

        total = 0
        for i, p in enumerate(self.npz_paths):
            d = np.load(p, allow_pickle=True)
            if "X" not in d:
                raise KeyError(f"{p} doit contenir X. Keys={list(d.keys())}")
            X = d["X"]
            if X.ndim != 4 or X.shape[1] != 8:
                raise ValueError(f"{p}: X doit être (N,8,H,W). Got {X.shape}")

            n = int(X.shape[0])
            self._lengths.append(n)
            total += n

            if i == 0 and "bands" in d:
                bands = d["bands"]
                if isinstance(bands, np.ndarray):
                    bands = bands.tolist()
                self.bands = [str(b) for b in bands]
            d.close()

        self._cum = np.cumsum(self._lengths).tolist()
        self._total = total

        self._load_osm_texts()

    def _clean_text(self, x: Any) -> str:
        s = str(x).strip()
        s = " ".join(s.split())
        return s

    def _clean_tag(self, x: Any) -> str:
        s = str(x).strip().lower()
        s = s.replace("-", "_")
        s = s.replace(" ", "_")
        s = "_".join([p for p in s.split("_") if p])
        return s

    def _dedup_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _build_text_from_phrases(self, phrases: List[str]) -> str:
        phrases = [self._clean_text(p) for p in phrases if self._clean_text(p)]
        if len(phrases) == 0:
            return ""

        phrases = phrases[:self.osm_max_phrases]

        if self.osm_text_mode == "first":
            return phrases[0]

        if self.osm_text_mode == "random":
            return random.choice(phrases)

        return " ".join(phrases)

    def _load_osm_texts(self):
        if not self.osm_texts_json:
            self.osm_texts_by_patch = {}
            self.osm_main_text_by_patch = {}
            self.osm_negative_text_pool = []
            print("[S2NPZDataset] INFO: No OSM JSON provided. Using fallback text.")
            return

        if not os.path.exists(self.osm_texts_json):
            print(f"[S2NPZDataset] WARN: osm_texts_json not found: {self.osm_texts_json}")
            self.osm_texts_by_patch = {}
            self.osm_main_text_by_patch = {}
            self.osm_negative_text_pool = []
            return

        with open(self.osm_texts_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"osm_texts_json must contain a dict patch_id -> list[str] or dict, got {type(data)}"
            )

        texts_by_patch: Dict[str, List[str]] = {}
        main_text_by_patch: Dict[str, str] = {}
        negative_pool: List[str] = []

        for k, v in data.items():
            patch_id = str(k)
            phrases: List[str] = []
            main_text = ""

            if isinstance(v, str):
                phrases = [self._clean_text(v)]
                main_text = self._build_text_from_phrases(phrases)

            elif isinstance(v, list):
                phrases = [self._clean_text(x) for x in v if self._clean_text(x)]
                main_text = self._build_text_from_phrases(phrases)

            elif isinstance(v, dict):
                raw_main = self._clean_text(v.get(self.osm_text_key, ""))
                raw_summary = self._clean_text(v.get("summary", ""))
                raw_source = self._clean_text(v.get("source_text", ""))

                raw_tags = v.get("tags", [])
                if isinstance(raw_tags, list):
                    tags = [self._clean_tag(x) for x in raw_tags if self._clean_tag(x)]
                    tags = self._dedup_keep_order(tags)
                else:
                    tags = []

                raw_phrases = v.get("phrases", [])
                if isinstance(raw_phrases, list):
                    phrases = [self._clean_text(x) for x in raw_phrases if self._clean_text(x)]
                elif isinstance(raw_phrases, str):
                    phrases = [self._clean_text(raw_phrases)]
                else:
                    phrases = []

                if raw_main:
                    main_text = raw_main
                elif raw_summary:
                    main_text = raw_summary
                elif len(tags) > 0:
                    main_text = " | ".join(tags)
                elif len(phrases) > 0:
                    main_text = self._build_text_from_phrases(phrases)
                elif raw_source:
                    main_text = raw_source
                else:
                    main_text = ""

                if len(phrases) == 0 and raw_source:
                    phrases = [raw_source]
                if len(phrases) == 0 and main_text:
                    phrases = [main_text]

            else:
                phrases = []
                main_text = ""

            phrases = [p for p in phrases if p]
            main_text = self._clean_text(main_text)

            if not main_text and len(phrases) > 0:
                main_text = self._build_text_from_phrases(phrases)

            if not main_text:
                continue

            texts_by_patch[patch_id] = phrases
            main_text_by_patch[patch_id] = main_text
            negative_pool.append(main_text)

        self.osm_texts_by_patch = texts_by_patch
        self.osm_main_text_by_patch = main_text_by_patch
        self.osm_negative_text_pool = self._dedup_keep_order(negative_pool)

        print(
            f"[S2NPZDataset] Loaded OSM texts: "
            f"{len(self.osm_main_text_by_patch)} patch entries, "
            f"{len(self.osm_negative_text_pool)} unique main texts."
        )

    def __len__(self) -> int:
        return int(self._total)

    def _find_shard(self, idx: int) -> Tuple[int, int]:
        prev = 0
        for sid, c in enumerate(self._cum):
            if idx < c:
                return sid, idx - prev
            prev = c
        raise IndexError(idx)

    def _get_files(self, d, n: int, sid: int) -> List[str]:
        if "files" in d:
            files = d["files"]
            if isinstance(files, np.ndarray):
                files = files.tolist()
            return [str(f) for f in files]
        return [f"sh{sid}_patch_{i:06d}" for i in range(n)]

    def _make_rgb(self, x8: torch.Tensor, which: str) -> torch.Tensor:
        which = which.upper()

        if which == "T1":
            b2, b3, b4 = x8[0], x8[1], x8[2]
        else:
            b2, b3, b4 = x8[4], x8[5], x8[6]

        rgb = torch.stack([b4, b3, b2], dim=0)
        rgb = rgb / self.s2_scale_div
        rgb = torch.clamp(rgb, 0.0, 1.0)

        rgb = rgb.unsqueeze(0)
        rgb = F.interpolate(
            rgb,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return rgb.squeeze(0)

    def _tokenize(self, texts: List[str]):
        if self.tokenizer is None:
            B = len(texts)
            L = self.max_text_len
            return (
                torch.zeros((B, L), dtype=torch.long),
                torch.ones((B, L), dtype=torch.long),
            )

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def _get_osm_phrases_for_patch(self, patch_id: str) -> List[str]:
        return self.osm_texts_by_patch.get(str(patch_id), [])

    def _build_main_text(self, patch_id: str) -> str:
        patch_id = str(patch_id)

        if patch_id in self.osm_main_text_by_patch:
            return self.osm_main_text_by_patch[patch_id]

        phrases = self._get_osm_phrases_for_patch(patch_id)
        if len(phrases) > 0:
            built = self._build_text_from_phrases(phrases)
            if built:
                return built

        return self.osm_fallback_text

    def _build_negative_text(self, current_main_text: str, neg_idx: int) -> str:
        pool = self.osm_negative_text_pool

        if len(pool) == 0:
            return self.osm_fallback_text

        if len(pool) == 1:
            return pool[0]

        candidates = [t for t in pool if t != current_main_text]
        if len(candidates) == 0:
            return random.choice(pool)

        return random.choice(candidates)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, lid = self._find_shard(int(idx))
        d = np.load(self.npz_paths[sid], allow_pickle=True)

        X = d["X"]
        n = int(X.shape[0])
        files = self._get_files(d, n, sid)

        x = X[lid]
        x8 = torch.from_numpy(x).float()

        rgb_t1 = self._make_rgb(x8, "T1")
        rgb_t2 = self._make_rgb(x8, "T2")
        rgb = rgb_t2 if self.rgb_from.upper() != "T1" else rgb_t1

        patch_id = files[lid]
        osm_phrases = self._get_osm_phrases_for_patch(patch_id)
        main_text = self._build_main_text(patch_id)
        has_osm_text = int(str(patch_id) in self.osm_main_text_by_patch)

        sample = {
            "idx": int(idx),
            "file": patch_id,
            "rgb": rgb,
            "rgb_t1": rgb_t1,
            "rgb_t2": rgb_t2,
            "x8": x8,
            "osm_texts": osm_phrases,
            "main_text": main_text,
            "has_osm_text": has_osm_text,
        }

        d.close()
        return sample

    def collate(self, batch: List[Dict[str, Any]], mlm_collator=None) -> Dict[str, Any]:
        B = len(batch)

        imgs = torch.stack([b["rgb"] for b in batch], dim=0)
        imgs_t1 = torch.stack([b["rgb_t1"] for b in batch], dim=0)
        imgs_t2 = torch.stack([b["rgb_t2"] for b in batch], dim=0)

        out: Dict[str, Any] = {}

        out["image"] = [imgs]
        out["image_t1"] = [imgs_t1]
        out["image_t2"] = [imgs_t2]

        out["image_index"] = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
        out["file"] = [b["file"] for b in batch]
        out["x8"] = torch.stack([b["x8"] for b in batch], dim=0)

        out["osm_texts"] = [b["osm_texts"] for b in batch]
        out["main_text"] = [b["main_text"] for b in batch]
        out["has_osm_text"] = torch.tensor([b["has_osm_text"] for b in batch], dtype=torch.long)

        texts = out["main_text"]
        text_ids, text_masks = self._tokenize(texts)
        out["text_ids"] = text_ids
        out["text_masks"] = text_masks
        out["text_labels"] = torch.full_like(text_ids, fill_value=-100)

        if self.draw_false_image > 0:
            neg_imgs = imgs.clone() if B == 1 else imgs[torch.randperm(B)]
            for i in range(self.draw_false_image):
                out[f"false_image_{i}"] = [neg_imgs]

        for i in range(max(0, self.draw_false_text)):
            neg_texts = [self._build_negative_text(b["main_text"], i) for b in batch]
            neg_ids, neg_masks = self._tokenize(neg_texts)
            out[f"false_text_{i}_ids"] = neg_ids
            out[f"false_text_{i}_masks"] = neg_masks
            out[f"false_text_{i}_labels"] = torch.full_like(neg_ids, fill_value=-100)

        return out