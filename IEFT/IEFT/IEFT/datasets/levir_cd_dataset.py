import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _is_image_file(name: str) -> bool:
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def _tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length < tile:
        return [0]
    xs = list(range(0, length - tile + 1, stride))
    if len(xs) == 0:
        xs = [0]
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy + 1e-6).astype(np.float32)


def _safe_text(x) -> str:
    return str(x).strip() if x is not None else ""


def _norm_count(x, cap=100.0):
    return float(min(float(x), cap) / cap)


def _robust01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.quantile(x, 0.02))
    hi = float(np.quantile(x, 0.98))
    if hi - lo < eps:
        mn = float(x.min())
        mx = float(x.max())
        if mx - mn < eps:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - mn) / (mx - mn + eps), 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def _negative_hardness_score(t1: np.ndarray, t2: np.ndarray) -> float:
    g1 = t1.mean(axis=2).astype(np.float32)
    g2 = t2.mean(axis=2).astype(np.float32)
    d_rgb = np.abs(t2 - t1).mean(axis=2).astype(np.float32)
    e1 = _grad_mag(g1)
    e2 = _grad_mag(g2)
    e_diff = np.abs(e2 - e1).astype(np.float32)
    tex = np.maximum(e1, e2).astype(np.float32)
    std2 = np.sqrt(np.maximum((g2 - g2.mean()) ** 2, 1e-8)).astype(np.float32)

    d_n = _robust01(d_rgb)
    ed_n = _robust01(e_diff)
    tex_n = _robust01(tex)
    std_n = _robust01(std2)

    score_map = 0.44 * d_n + 0.28 * ed_n + 0.18 * tex_n + 0.10 * std_n
    return float(np.clip(score_map.mean(), 0.0, 1.0))


class LEVIROSMHelper:
    def __init__(
        self,
        osm_texts_json: str,
        osm_text_mode: str = "concat",
        osm_max_phrases: int = 3,
        osm_text_key: str = "text_v21",
        osm_fallback_text: str = "no_osm_context",
        osm_compose_mode: str = "signature_compact",
        osm_word_budget: int = 32,
        osm_joiner: str = " ; ",
        osm_include_source_text: bool = False,
    ):
        self.osm_texts_json = _safe_text(osm_texts_json)
        self.osm_text_mode = str(osm_text_mode)
        self.osm_max_phrases = int(osm_max_phrases)
        self.osm_text_key = str(osm_text_key)
        self.osm_fallback_text = str(osm_fallback_text)
        self.osm_compose_mode = str(osm_compose_mode)
        self.osm_word_budget = int(osm_word_budget)
        self.osm_joiner = str(osm_joiner)
        self.osm_include_source_text = bool(osm_include_source_text)

        self.db: Dict[str, Dict] = {}
        if self.osm_texts_json and os.path.isfile(self.osm_texts_json):
            with open(self.osm_texts_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.db = raw

    def _stem_candidates(self, patch_id: str) -> List[str]:
        patch_id = _safe_text(patch_id)
        out = []
        if patch_id:
            out.append(patch_id)
        m = re.match(r"^(.*?)(_y\d+_x\d+)$", patch_id)
        if m:
            out.append(m.group(1))
        if patch_id.endswith(".png"):
            out.append(patch_id[:-4])
        seen = []
        for k in out:
            if k and k not in seen:
                seen.append(k)
        return seen

    def _get_entry(self, patch_id: str) -> Dict:
        for k in self._stem_candidates(patch_id):
            if k in self.db:
                v = self.db[k]
                if isinstance(v, list):
                    return {"phrases": v, self.osm_text_key: self.osm_joiner.join([_safe_text(x) for x in v if _safe_text(x)])}
                if isinstance(v, dict):
                    return v
                if isinstance(v, str):
                    return {self.osm_text_key: v}
        return {}

    def _text_from_entry(self, entry: Dict, fallback_text: str) -> Dict:
        text = _safe_text(entry.get(self.osm_text_key, ""))
        phrases = entry.get("phrases", [])
        if not isinstance(phrases, list):
            phrases = []
        phrases = [_safe_text(x) for x in phrases if _safe_text(x)]
        phrases = phrases[: self.osm_max_phrases]
        summary = _safe_text(entry.get("summary", ""))
        source_text = _safe_text(entry.get("source_text", ""))

        if self.osm_text_mode == "phrases" and phrases:
            main_text = self.osm_joiner.join(phrases)
        elif self.osm_text_mode == "summary" and summary:
            main_text = summary
        elif self.osm_text_mode == "concat":
            parts = []
            if text:
                parts.append(text)
            if summary and summary.lower() not in text.lower():
                parts.append(summary)
            if phrases:
                parts.extend(phrases)
            if self.osm_include_source_text and source_text:
                parts.append(source_text)
            uniq = []
            for p in parts:
                if p and p not in uniq:
                    uniq.append(p)
            main_text = self.osm_joiner.join(uniq)
        else:
            main_text = text or summary or (self.osm_joiner.join(phrases) if phrases else "")

        main_text = _safe_text(main_text)
        if not main_text:
            main_text = fallback_text

        words = main_text.split()
        if len(words) > self.osm_word_budget:
            main_text = " ".join(words[: self.osm_word_budget])

        return {
            "main_text": main_text,
            "phrases": phrases,
            "summary": summary,
            "source_text": source_text,
        }

    def _struct_from_entry(self, entry: Dict) -> torch.Tensor:
        hint = entry.get("osm_struct_hint", {})
        if not isinstance(hint, dict):
            hint = {}
        tags = entry.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = set([_safe_text(x).lower() for x in tags if _safe_text(x)])
        text_blob = " ".join([
            _safe_text(entry.get("text_v21", "")),
            _safe_text(entry.get("summary", "")),
            _safe_text(entry.get("source_text", "")),
            " ".join([_safe_text(x) for x in entry.get("phrases", []) if _safe_text(x)]),
            " ".join(sorted(tags)),
        ]).lower()

        def has_any(words):
            return 1.0 if any(w in text_blob for w in words) else 0.0

        building_count = float(hint.get("building_count", 0))
        road_count = float(hint.get("road_count", 0))
        railway_count = float(hint.get("railway_count", 0))
        water_count = float(hint.get("water_count", 0))
        vegetation_count = float(hint.get("vegetation_count", 0))
        construction_count = float(hint.get("construction_count", 0))
        industrial_count = float(hint.get("industrial_count", 0))
        residential_count = float(hint.get("residential_count", 0))
        amenity_count = float(hint.get("amenity_count", 0))

        vec = np.zeros(16, dtype=np.float32)
        vec[0] = max(has_any(["building", "built-up", "built up", "residential", "industrial"]), 1.0 if building_count > 0 else 0.0)
        vec[1] = max(has_any(["road", "transport", "highway", "street"]), 1.0 if road_count > 0 else 0.0)
        vec[2] = max(has_any(["railway", "rail"]), 1.0 if railway_count > 0 else 0.0)
        vec[3] = max(has_any(["water", "river", "wetland"]), 1.0 if water_count > 0 else 0.0)
        vec[4] = max(has_any(["vegetation", "forest", "wood", "park", "green"]), 1.0 if vegetation_count > 0 else 0.0)
        vec[5] = max(has_any(["residential", "neighborhood"]), 1.0 if residential_count > 0 else 0.0)
        vec[6] = max(has_any(["industrial", "commercial"]), 1.0 if industrial_count > 0 else 0.0)
        vec[7] = max(has_any(["construction"]), 1.0 if construction_count > 0 else 0.0)
        vec[8] = max(has_any(["amenity", "parking"]), 1.0 if amenity_count > 0 else 0.0)
        vec[9] = _norm_count(building_count, 120.0)
        vec[10] = _norm_count(road_count + railway_count, 180.0)
        vec[11] = _norm_count(water_count, 30.0)
        vec[12] = _norm_count(vegetation_count, 60.0)
        vec[13] = min(1.0, 0.55 * vec[0] + 0.25 * vec[5] + 0.20 * vec[6] + 0.35 * vec[9])
        vec[14] = min(1.0, 0.55 * vec[1] + 0.20 * vec[2] + 0.25 * vec[10])
        vec[15] = min(1.0, 0.50 * vec[13] + 0.30 * vec[14] - 0.20 * vec[3] - 0.10 * vec[4] + 0.25 * vec[7])
        return torch.from_numpy(vec)

    def resolve(self, patch_ids: List[str], fallback_text: str = "no_osm_context") -> Dict:
        if not patch_ids:
            return {
                "main_text": fallback_text,
                "osm_texts": [],
                "has_osm_text": 0,
                "osm_struct": torch.zeros(16, dtype=torch.float32),
            }
        entry = self._get_entry(patch_ids[0])
        txt = self._text_from_entry(entry, fallback_text)
        struct = self._struct_from_entry(entry) if entry else torch.zeros(16, dtype=torch.float32)
        has_text = 1 if entry else 0
        osm_texts = txt["phrases"] if txt["phrases"] else ([txt["main_text"]] if has_text else [])
        return {
            "main_text": txt["main_text"],
            "osm_texts": osm_texts,
            "has_osm_text": has_text,
            "osm_struct": struct,
        }


class LEVIRCDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 256,
        crop_size: int = 256,
        tile_stride: int = 256,
        train_repeat: int = 1,
        train_focus_positive: bool = True,
        positive_focus_prob: float = 0.58,
        hard_negative_prob: float = 0.20,
        positive_thr: float = 0.003,
        negative_mining_mode: str = "rural_structural",
        hard_negative_gamma: float = 2.0,
        hard_negative_min_weight: float = 0.02,
        extreme_negative_prob: float = 0.10,
        extreme_negative_quantile: float = 0.88,
        random_aug: bool = False,
        label_smoothing: float = 0.0,
        tokenizer=None,
        max_text_len: int = 40,
        fixed_text: str = "building change detection",
        mask_threshold: int = 127,
        a_dirname: str = "A",
        b_dirname: str = "B",
        label_dirname: str = "label",
        use_osm: bool = False,
        osm_texts_json: str = "",
        osm_text_mode: str = "concat",
        osm_max_phrases: int = 3,
        osm_text_key: str = "text_v21",
        osm_fallback_text: str = "no_osm_context",
        osm_compose_mode: str = "signature_compact",
        osm_word_budget: int = 32,
        osm_joiner: str = " ; ",
        osm_include_source_text: bool = False,
    ):
        super().__init__()
        self.root = str(root)
        self.split = str(split)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size)
        self.tile_stride = int(tile_stride)
        self.train_repeat = max(1, int(train_repeat))
        self.train_focus_positive = bool(train_focus_positive) and self.split == "train"
        self.positive_focus_prob = float(positive_focus_prob)
        self.hard_negative_prob = float(hard_negative_prob)
        self.positive_thr = float(positive_thr)
        self.negative_mining_mode = str(negative_mining_mode)
        self.hard_negative_gamma = float(max(0.5, hard_negative_gamma))
        self.hard_negative_min_weight = float(max(1e-4, hard_negative_min_weight))
        self.extreme_negative_prob = float(max(0.0, extreme_negative_prob))
        self.extreme_negative_quantile = float(min(max(extreme_negative_quantile, 0.50), 0.99))
        self.random_aug = bool(random_aug) and self.split == "train"
        self.label_smoothing = float(label_smoothing)
        self.fixed_text = str(fixed_text)
        self.mask_threshold = int(mask_threshold)
        self.max_text_len = int(max_text_len)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("bert-base-uncased")
        self.a_dir = Path(root) / split / a_dirname
        self.b_dir = Path(root) / split / b_dirname
        self.l_dir = Path(root) / split / label_dirname

        self.use_osm = bool(use_osm) and bool(osm_texts_json)
        self.osm_helper = LEVIROSMHelper(
            osm_texts_json=osm_texts_json,
            osm_text_mode=osm_text_mode,
            osm_max_phrases=osm_max_phrases,
            osm_text_key=osm_text_key,
            osm_fallback_text=osm_fallback_text,
            osm_compose_mode=osm_compose_mode,
            osm_word_budget=osm_word_budget,
            osm_joiner=osm_joiner,
            osm_include_source_text=osm_include_source_text,
        ) if self.use_osm else None

        if not self.a_dir.is_dir() or not self.b_dir.is_dir() or not self.l_dir.is_dir():
            raise FileNotFoundError(f"Missing LEVIR split folders under {Path(root) / split}")

        files_a = {Path(n).stem: self.a_dir / n for n in os.listdir(self.a_dir) if _is_image_file(n)}
        files_b = {Path(n).stem: self.b_dir / n for n in os.listdir(self.b_dir) if _is_image_file(n)}
        files_l = {Path(n).stem: self.l_dir / n for n in os.listdir(self.l_dir) if _is_image_file(n)}
        stems = sorted(set(files_a) & set(files_b) & set(files_l))
        if len(stems) == 0:
            raise RuntimeError(f"No aligned A/B/label files found in {Path(root) / split}")

        self.samples: List[Dict] = []
        self.positive_indices: List[int] = []
        self.negative_indices: List[int] = []
        self.negative_weights: List[float] = []
        self.extreme_negative_indices: List[int] = []
        self.extreme_negative_weights: List[float] = []

        # Build tile metadata and stronger hard-negative weights without changing model parameters.
        for stem in stems:
            label_img = np.asarray(Image.open(files_l[stem]).convert("L"), dtype=np.uint8)
            need_negative_scores = self.split == "train" and self.negative_mining_mode != "none"
            img_a = None
            img_b = None
            if need_negative_scores:
                img_a = np.asarray(Image.open(files_a[stem]).convert("RGB"), dtype=np.float32) / 255.0
                img_b = np.asarray(Image.open(files_b[stem]).convert("RGB"), dtype=np.float32) / 255.0
            h, w = label_img.shape
            ys = _tile_starts(h, self.crop_size, self.tile_stride)
            xs = _tile_starts(w, self.crop_size, self.tile_stride)
            for y in ys:
                for x in xs:
                    crop_l = label_img[y:y + self.crop_size, x:x + self.crop_size]
                    if crop_l.shape[0] != self.crop_size or crop_l.shape[1] != self.crop_size:
                        continue
                    pos_ratio = float((crop_l > self.mask_threshold).mean())
                    if pos_ratio > self.positive_thr:
                        hardness = float(max(1e-4, pos_ratio + 0.01))
                        neg_score = 0.0
                    else:
                        neg_score = 0.0
                        if need_negative_scores and img_a is not None and img_b is not None:
                            crop_a = img_a[y:y + self.crop_size, x:x + self.crop_size]
                            crop_b = img_b[y:y + self.crop_size, x:x + self.crop_size]
                            if crop_a.shape[0] == self.crop_size and crop_a.shape[1] == self.crop_size:
                                neg_score = _negative_hardness_score(crop_a, crop_b)
                        hardness = float(max(self.hard_negative_min_weight, 0.02 + neg_score))
                    meta = {
                        "stem": stem,
                        "patch_id": f"{stem}_y{int(y)}_x{int(x)}",
                        "path_a": str(files_a[stem]),
                        "path_b": str(files_b[stem]),
                        "path_l": str(files_l[stem]),
                        "x": int(x),
                        "y": int(y),
                        "pos_ratio": pos_ratio,
                        "hardness": hardness,
                        "negative_score": float(neg_score),
                    }
                    idx = len(self.samples)
                    self.samples.append(meta)
                    if pos_ratio > self.positive_thr:
                        self.positive_indices.append(idx)
                    else:
                        self.negative_indices.append(idx)
                        self.negative_weights.append(max(self.hard_negative_min_weight, hardness))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid tiles built for split={split}")

        if len(self.negative_weights) > 0:
            w = np.asarray(self.negative_weights, dtype=np.float64)
            w = np.power(w / max(w.max(), 1e-6), self.hard_negative_gamma)
            w = np.clip(w, self.hard_negative_min_weight, None)
            w = w / max(w.sum(), 1e-12)
            self.negative_weights = w.tolist()

            neg_scores = np.asarray([float(self.samples[i].get("negative_score", 0.0)) for i in self.negative_indices], dtype=np.float64)
            if neg_scores.size > 0:
                q = float(np.quantile(neg_scores, self.extreme_negative_quantile))
                for neg_idx, neg_score, neg_w in zip(self.negative_indices, neg_scores.tolist(), self.negative_weights):
                    if neg_score >= q and neg_score > 0:
                        self.extreme_negative_indices.append(int(neg_idx))
                        self.extreme_negative_weights.append(float(max(neg_w, self.hard_negative_min_weight)))
                if len(self.extreme_negative_weights) > 0:
                    ew = np.asarray(self.extreme_negative_weights, dtype=np.float64)
                    ew = ew / max(ew.sum(), 1e-12)
                    self.extreme_negative_weights = ew.tolist()

    def __len__(self):
        return len(self.samples) * self.train_repeat if self.split == "train" else len(self.samples)

    def _choose_index(self, idx: int) -> int:
        if self.split != "train":
            return idx
        r = random.random()
        p_pos = self.positive_focus_prob if (self.train_focus_positive and len(self.positive_indices) > 0) else 0.0
        p_hard = self.hard_negative_prob if len(self.negative_indices) > 0 else 0.0
        p_extreme = self.extreme_negative_prob if len(self.extreme_negative_indices) > 0 else 0.0

        if r < p_pos:
            return random.choice(self.positive_indices)
        r -= p_pos
        if r < p_hard:
            return random.choices(self.negative_indices, weights=self.negative_weights, k=1)[0]
        r -= p_hard
        if r < p_extreme:
            return random.choices(self.extreme_negative_indices, weights=self.extreme_negative_weights, k=1)[0]
        return random.randrange(len(self.samples))

    def _load_tile(self, meta: Dict):
        t1 = np.asarray(Image.open(meta["path_a"]).convert("RGB"), dtype=np.float32) / 255.0
        t2 = np.asarray(Image.open(meta["path_b"]).convert("RGB"), dtype=np.float32) / 255.0
        lb = np.asarray(Image.open(meta["path_l"]).convert("L"), dtype=np.float32)
        y, x = meta["y"], meta["x"]
        t1 = t1[y:y + self.crop_size, x:x + self.crop_size]
        t2 = t2[y:y + self.crop_size, x:x + self.crop_size]
        lb = lb[y:y + self.crop_size, x:x + self.crop_size]
        return t1, t2, lb

    def _augment(self, t1, t2, lb):
        if not self.random_aug:
            return t1, t2, lb
        if random.random() < 0.5:
            t1 = np.flip(t1, axis=1).copy()
            t2 = np.flip(t2, axis=1).copy()
            lb = np.flip(lb, axis=1).copy()
        if random.random() < 0.5:
            t1 = np.flip(t1, axis=0).copy()
            t2 = np.flip(t2, axis=0).copy()
            lb = np.flip(lb, axis=0).copy()
        k = random.randint(0, 3)
        if k > 0:
            t1 = np.rot90(t1, k=k, axes=(0, 1)).copy()
            t2 = np.rot90(t2, k=k, axes=(0, 1)).copy()
            lb = np.rot90(lb, k=k, axes=(0, 1)).copy()
        return t1, t2, lb

    def __getitem__(self, idx: int):
        ridx = self._choose_index(idx)
        meta = self.samples[ridx]
        t1, t2, lb = self._load_tile(meta)
        t1, t2, lb = self._augment(t1, t2, lb)

        gt = (lb > self.mask_threshold).astype(np.float32)
        if self.label_smoothing > 0:
            gt = gt * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        patch_id = meta["patch_id"]
        if self.osm_helper is not None:
            osm = self.osm_helper.resolve([patch_id], fallback_text=self.fixed_text)
            main_text = str(osm["main_text"])
            has_osm_text = int(osm["has_osm_text"])
            osm_struct = osm["osm_struct"].clone().float()
            osm_texts = list(osm["osm_texts"])
        else:
            main_text = self.fixed_text
            has_osm_text = 0
            osm_struct = torch.zeros(16, dtype=torch.float32)
            osm_texts = []

        enc = self.tokenizer(
            main_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        return {
            "image": [torch.from_numpy(t2.transpose(2, 0, 1).astype(np.float32))],
            "image_t1": [torch.from_numpy(t1.transpose(2, 0, 1).astype(np.float32))],
            "image_t2": [torch.from_numpy(t2.transpose(2, 0, 1).astype(np.float32))],
            "gt_mask": torch.from_numpy(gt.astype(np.float32)).unsqueeze(0),
            "text_ids": enc["input_ids"][0],
            "text_masks": enc["attention_mask"][0],
            "text_labels": torch.full((self.max_text_len,), -100, dtype=torch.long),
            "text": [main_text],
            "main_text": [main_text],
            "file": [patch_id],
            "img_index": torch.tensor(0, dtype=torch.long),
            "image_index": torch.tensor(0, dtype=torch.long),
            "has_osm_text": torch.tensor(has_osm_text, dtype=torch.long),
            "osm_struct": osm_struct,
            "osm_texts": [osm_texts],
        }
