import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, default_collate

from IEFT.datasets.levir_auxiliary import (
    ManifestIndex,
    OSMGeoJSONCache,
    SPECTRAL_CHANNEL_ORDER,
    SPECTRAL_INDEX_ORDER,
    SURFACE_REFLECTANCE_ORDER,
    SpectralCache,
    assess_temporal_source,
    build_levir_tokenizer,
    load_index_normalization,
    load_spectral_date,
    natural_sample_sort_key,
    normalize_sample_key,
    osm_struct_from_entry,
    osm_text_from_entry,
)

from IEFT.osm_geometry import (
    OSM_SPATIAL_CHANNEL_ORDER,
    rasterize_feature_collection,
    summarize_tile,
)

try:
    from IEFT.levir_metadata import month_timestamp, region_for_sample, tile_bbox
except (ImportError, AttributeError):  # pragma: no cover - old installation compatibility.
    month_timestamp = None
    region_for_sample = None
    tile_bbox = None


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


OSM_CLIP_CONCEPT_ORDER = (
    "building",
    "transport",
    "water",
    "vegetation",
    "land_use",
)
OSM_CLIP_FILTER_SCHEMA = "levir-osm-clip-filter-v1"


def _load_osm_clip_filter(
    path: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load the optional offline CLIP-to-OSM category-gate cache."""
    if path is None or not path.is_file():
        return {}, {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"OSM CLIP filter root must be an object: {path}")
    schema = _safe_text(document.get("schema_version", ""))
    if schema != OSM_CLIP_FILTER_SCHEMA:
        raise ValueError(
            f"Unsupported OSM CLIP filter schema {schema!r} in {path}; "
            f"expected {OSM_CLIP_FILTER_SCHEMA!r}"
        )
    concept_order = tuple(str(x) for x in document.get("concept_order", []))
    if concept_order != OSM_CLIP_CONCEPT_ORDER:
        raise ValueError(
            "OSM CLIP concept order mismatch: "
            f"{concept_order!r} != {OSM_CLIP_CONCEPT_ORDER!r}"
        )
    raw_samples = document.get("samples", {})
    if not isinstance(raw_samples, Mapping):
        raise ValueError(f"OSM CLIP filter samples must be an object: {path}")

    records: Dict[str, Dict[str, Any]] = {}
    for key, raw in raw_samples.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid OSM CLIP record for {key!r}")
        stem = normalize_sample_key(key)
        record = dict(raw)
        gate = np.asarray(record.get("category_gate", []), dtype=np.float32)
        similarities = np.asarray(record.get("similarities", []), dtype=np.float32)
        if gate.shape != (5,) or similarities.shape != (5,):
            raise ValueError(
                f"{stem}: category_gate/similarities must both be length 5"
            )
        if not np.all(np.isfinite(gate)) or np.any(gate < 0.0) or np.any(gate > 1.0):
            raise ValueError(f"{stem}: invalid category_gate values")
        if not np.all(np.isfinite(similarities)):
            raise ValueError(f"{stem}: non-finite CLIP similarities")
        confidence = float(record.get("confidence", 0.0))
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{stem}: invalid CLIP confidence {confidence}")
        records[stem] = record
    return dict(document), records


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
        return osm_struct_from_entry(entry, dim=16)

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
        osm_manifest_json: str = "",
        spectral_manifest_json: str = "",
        auxiliary_policy: str = "mask",
        require_paired_osm: Optional[bool] = None,
        require_spectral: Optional[bool] = None,
        require_osm_t1: Optional[bool] = None,
        require_osm_t2: Optional[bool] = None,
        require_indices_t1: Optional[bool] = None,
        require_indices_t2: Optional[bool] = None,
        accept_partial_indices: bool = True,
        require_nonempty_osm: bool = False,
        filter_failed_osm: bool = True,
        osm_timestamp_policy: str = "",
        temporal_osm_mode: str = "paired",
        use_temporal_osm: Optional[bool] = None,
        use_spectral: Optional[bool] = None,
        allow_legacy_spectral_indices: bool = False,
        spectral_min_valid_fraction: float = 0.0,
        spectral_cache_size: int = 2,
        osm_cache_size: int = 1,
        index_normalization: str = "natural",
        index_normalization_stats: str = "",
        tokenizer_mode: str = "simple",
        tokenizer_name: str = "bert-base-uncased",
        tokenizer_local_only: bool = True,
        tokenizer_fallback_to_simple: bool = False,
        tokenizer_vocab_size: int = 30522,
        # Constructor aliases retained for external callers while config uses
        # the canonical *_manifest_json names above.
        osm_manifest: str = "",
        spectral_manifest: str = "",
        strict_auxiliary: Optional[bool] = None,
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
        self.tokenizer = build_levir_tokenizer(
            tokenizer=tokenizer,
            mode=tokenizer_mode,
            name=tokenizer_name,
            local_files_only=tokenizer_local_only,
            fallback_to_simple=tokenizer_fallback_to_simple,
            vocab_size=tokenizer_vocab_size,
        )
        self.a_dir = Path(root) / split / a_dirname
        self.b_dir = Path(root) / split / b_dirname
        self.l_dir = Path(root) / split / label_dirname

        osm_manifest_path = _safe_text(osm_manifest_json) or _safe_text(osm_manifest)
        spectral_manifest_path = _safe_text(spectral_manifest_json) or _safe_text(spectral_manifest)
        self.osm_manifest = ManifestIndex(osm_manifest_path, kind="temporal OSM")
        self.spectral_manifest = ManifestIndex(spectral_manifest_path, kind="spectral")
        self.osm_clip_filter_path = (
            self.osm_manifest.base_dir / "clip_filter.json"
            if self.osm_manifest.enabled and self.osm_manifest.base_dir is not None
            else None
        )
        (
            self.osm_clip_filter_document,
            self.osm_clip_filter_records,
        ) = _load_osm_clip_filter(self.osm_clip_filter_path)
        osm_mode_aliases = {
            "temporal": "paired",
            "paired_temporal": "paired",
            "t2_only": "t2",
            "single_t2": "t2",
        }
        requested_osm_mode = _safe_text(temporal_osm_mode).lower() or "paired"
        self.temporal_osm_mode = osm_mode_aliases.get(requested_osm_mode, requested_osm_mode)
        if self.temporal_osm_mode not in {"paired", "t2"}:
            raise ValueError(
                "temporal_osm_mode must be 'paired' or 't2', got "
                f"{temporal_osm_mode!r}"
            )
        self.osm_temporal_keys = (
            ("t2",) if self.temporal_osm_mode == "t2" else ("t1", "t2")
        )
        self.allow_legacy_spectral_indices = bool(allow_legacy_spectral_indices)
        policy = _safe_text(auxiliary_policy).lower().replace("-", "_")
        policy_aliases = {
            "mask_missing": "mask",
            "masked_missing": "mask",
            "keep": "mask",
            "strict_filter": "strict",
            "filter": "strict",
        }
        policy = policy_aliases.get(policy, policy or "mask")
        if strict_auxiliary is not None:
            policy = "strict" if bool(strict_auxiliary) else "mask"
        if policy not in {"mask", "strict"}:
            raise ValueError(f"auxiliary_policy must be 'mask' or 'strict', got {auxiliary_policy!r}")
        self.auxiliary_policy = policy
        self._explicit_osm_requirements = require_osm_t1 is not None or require_osm_t2 is not None
        paired_osm_default = (
            self.osm_manifest.enabled if require_paired_osm is None else bool(require_paired_osm)
        )
        spectral_default = (
            self.spectral_manifest.enabled if require_spectral is None else bool(require_spectral)
        )
        self.require_osm_t1 = paired_osm_default if require_osm_t1 is None else bool(require_osm_t1)
        self.require_osm_t2 = paired_osm_default if require_osm_t2 is None else bool(require_osm_t2)
        if self.temporal_osm_mode == "t2":
            if require_osm_t1 is not None and bool(require_osm_t1):
                raise ValueError(
                    "T2-only OSM mode forbids require_osm_t1=True; no T1 OSM may be loaded"
                )
            self.require_osm_t1 = False
        self.require_indices_t1 = (
            spectral_default if require_indices_t1 is None else bool(require_indices_t1)
        )
        self.require_indices_t2 = (
            spectral_default if require_indices_t2 is None else bool(require_indices_t2)
        )
        self.require_paired_osm = bool(self.require_osm_t1 and self.require_osm_t2)
        self.require_spectral = bool(self.require_indices_t1 and self.require_indices_t2)
        if (self.require_osm_t1 or self.require_osm_t2) and not self.osm_manifest.enabled:
            raise ValueError("required temporal OSM dates need an OSM temporal manifest")
        if (self.require_indices_t1 or self.require_indices_t2) and not self.spectral_manifest.enabled:
            raise ValueError("required spectral dates need a spectral manifest")
        self.accept_partial_indices = bool(accept_partial_indices)
        self.require_nonempty_osm = bool(require_nonempty_osm)
        self.filter_failed_osm = bool(filter_failed_osm)
        self.osm_timestamp_policy = _safe_text(osm_timestamp_policy)
        self.use_temporal_osm = self.osm_manifest.enabled if use_temporal_osm is None else bool(use_temporal_osm)
        self.use_spectral = self.spectral_manifest.enabled if use_spectral is None else bool(use_spectral)
        self.spectral_min_valid_fraction = float(spectral_min_valid_fraction)
        if not 0.0 <= self.spectral_min_valid_fraction <= 1.0:
            raise ValueError("spectral_min_valid_fraction must lie in [0, 1]")
        self.index_normalization = load_index_normalization(
            index_normalization,
            stats_path=index_normalization_stats,
            spectral_manifest_sha256=self.spectral_manifest.sha256,
        )
        self._validate_auxiliary_timestamps()
        self._spectral_cache = SpectralCache(max_sources=spectral_cache_size)
        self._osm_geojson_cache = OSMGeoJSONCache(max_sources=osm_cache_size)
        self.source_auxiliary: Dict[str, Dict[str, Any]] = {}
        self.excluded_sources: List[Dict[str, Any]] = []
        self.retained_sources: List[str] = []
        self.filtering_stats: Dict[str, Any] = {}

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
        stems = sorted(set(files_a) & set(files_b) & set(files_l), key=natural_sample_sort_key)
        if len(stems) == 0:
            raise RuntimeError(f"No aligned A/B/label files found in {Path(root) / split}")

        status_counts = {
            "osm_t1": Counter(),
            "osm_t2": Counter(),
            "spectral_t1": Counter(),
            "spectral_t2": Counter(),
        }
        reason_counts: Counter = Counter()
        retained_stems: List[str] = []
        for stem in stems:
            osm_assessment = assess_temporal_source(
                stem,
                self.osm_manifest,
                "osm",
                require_nonempty=self.require_nonempty_osm,
                require_osm_cache=(
                    self.use_temporal_osm or self.require_osm_t1 or self.require_osm_t2
                ),
                temporal_keys=self.osm_temporal_keys,
            )
            spectral_assessment = assess_temporal_source(
                stem,
                self.spectral_manifest,
                "spectral",
                min_valid_fraction=self.spectral_min_valid_fraction,
                accept_partial=self.accept_partial_indices,
                allow_legacy_spectral_indices=self.allow_legacy_spectral_indices,
            )
            for temporal_key in ("t1", "t2"):
                status_counts[f"osm_{temporal_key}"][osm_assessment["status"][temporal_key]] += 1
                status_counts[f"spectral_{temporal_key}"][spectral_assessment["status"][temporal_key]] += 1

            osm_requirement_reasons: List[str] = []
            spectral_requirement_reasons: List[str] = []
            for temporal_key, required in (
                ("t1", self.require_osm_t1),
                ("t2", self.require_osm_t2),
            ):
                if required:
                    osm_requirement_reasons.extend(
                        osm_assessment["reasons_by_date"][temporal_key]
                    )
            for temporal_key, required in (
                ("t1", self.require_indices_t1),
                ("t2", self.require_indices_t2),
            ):
                if required:
                    spectral_requirement_reasons.extend(
                        spectral_assessment["reasons_by_date"][temporal_key]
                    )
            failed_osm_reasons = []
            if self.osm_manifest.enabled and self.filter_failed_osm:
                for temporal_key in self.osm_temporal_keys:
                    status = osm_assessment["status"][temporal_key]
                    if status in {"failed_retryable", "failed_permanent"}:
                        failed_osm_reasons.append(f"osm_{temporal_key}:{status}")
            exclusion_reasons = (
                osm_requirement_reasons
                + spectral_requirement_reasons
                + failed_osm_reasons
            )
            # Deduplicate while retaining a stable date/modality order.
            exclusion_reasons = list(dict.fromkeys(exclusion_reasons))
            self.source_auxiliary[stem] = {
                "osm": osm_assessment,
                "spectral": spectral_assessment,
            }
            # Canonical per-date OSM requirements are unconditional source
            # eligibility constraints.  The legacy pair-wide flag continues
            # to respect the historical general mask/strict policy.  Spectral
            # requirements follow levir_spectral_missing_policy.
            should_exclude = (
                bool(failed_osm_reasons)
                or (self._explicit_osm_requirements and bool(osm_requirement_reasons))
                or (
                    self.auxiliary_policy == "strict"
                    and bool(osm_requirement_reasons + spectral_requirement_reasons)
                )
            )
            if should_exclude:
                for reason in exclusion_reasons:
                    reason_counts[reason] += 1
                self.excluded_sources.append(
                    {
                        "source_pair_id": stem,
                        "filename": stem + ".png",
                        "split": self.split,
                        "reasons": exclusion_reasons,
                        "osm_status_t1": osm_assessment["status"]["t1"],
                        "osm_status_t2": osm_assessment["status"]["t2"],
                        "spectral_status_t1": spectral_assessment["status"]["t1"],
                        "spectral_status_t2": spectral_assessment["status"]["t2"],
                        "osm_missing_paths": list(osm_assessment.get("missing_paths", [])),
                        "osm_cache_errors": dict(osm_assessment.get("cache_errors", {})),
                        "spectral_missing_paths": list(
                            spectral_assessment.get("missing_paths", [])
                        ),
                    }
                )
            else:
                retained_stems.append(stem)

        self.retained_sources = list(retained_stems)
        self.filtering_stats = {
            "split": self.split,
            "policy": self.auxiliary_policy,
            "required_modalities": {
                "paired_osm": bool(self.require_paired_osm),
                "spectral": bool(self.require_spectral),
                "osm_t1": bool(self.require_osm_t1),
                "osm_t2": bool(self.require_osm_t2),
                "indices_t1": bool(self.require_indices_t1),
                "indices_t2": bool(self.require_indices_t2),
            },
            "source_counts": {
                "discovered": len(stems),
                "retained": len(retained_stems),
                "excluded": len(self.excluded_sources),
            },
            "status_counts": {
                name: dict(sorted(counter.items())) for name, counter in status_counts.items()
            },
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "retained_source_ids": list(retained_stems),
            "excluded_sources": list(self.excluded_sources),
            "osm_manifest": {
                "path": str(self.osm_manifest.path) if self.osm_manifest.path else "",
                "schema_version": self.osm_manifest.schema_version,
                "sha256": self.osm_manifest.sha256,
            },
            "spectral_manifest": {
                "path": str(self.spectral_manifest.path) if self.spectral_manifest.path else "",
                "schema_version": self.spectral_manifest.schema_version,
                "sha256": self.spectral_manifest.sha256,
            },
            "spectral_min_valid_fraction": self.spectral_min_valid_fraction,
            "accept_partial_indices": self.accept_partial_indices,
            "require_nonempty_osm": self.require_nonempty_osm,
            "filter_failed_osm": self.filter_failed_osm,
            "osm_cache_size_sources": self._osm_geojson_cache.max_sources,
            "index_normalization": {
                "mode": self.index_normalization["mode"],
                "path": self.index_normalization["path"],
            },
        }
        stems = retained_stems
        if len(stems) == 0:
            reasons = ", ".join(
                f"{name}={count}" for name, count in sorted(reason_counts.items())
            )
            raise RuntimeError(
                f"No LEVIR sources remain for split={split} after {self.auxiliary_policy} "
                f"auxiliary filtering ({reasons or 'no eligible records'})"
            )

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
            osm_manifest_sample = self.osm_manifest.get(stem)
            spectral_manifest_sample = self.spectral_manifest.get(stem)
            manifest_sample = osm_manifest_sample or spectral_manifest_sample or {}
            if isinstance(osm_manifest_sample, Mapping) and isinstance(spectral_manifest_sample, Mapping):
                for field in ("split", "region_id", "bbox"):
                    left = osm_manifest_sample.get(field)
                    right = spectral_manifest_sample.get(field)
                    if left not in (None, "", []) and right not in (None, "", []) and left != right:
                        raise ValueError(
                            f"OSM/spectral manifest {field} mismatch for {stem}: {left!r} != {right!r}"
                        )
            region_id = manifest_sample.get("region_id", -1) if isinstance(manifest_sample, Mapping) else -1
            if region_id in (None, "", -1) and region_for_sample is not None:
                try:
                    region_id = region_for_sample(stem)
                except Exception:
                    region_id = -1
            try:
                region_id = int(region_id)
            except (TypeError, ValueError):
                region_id = -1
            source_bbox = manifest_sample.get("bbox", []) if isinstance(manifest_sample, Mapping) else []
            if (
                (not isinstance(source_bbox, (list, tuple)) or len(source_bbox) != 4)
                and isinstance(spectral_manifest_sample, Mapping)
            ):
                source_bbox = spectral_manifest_sample.get("bbox", [])
            if not isinstance(source_bbox, (list, tuple)) or len(source_bbox) != 4:
                source_bbox = []
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
                    spatial_tile_bbox: List[float] = []
                    if source_bbox and tile_bbox is not None:
                        try:
                            spatial_tile_bbox = tile_bbox(
                                source_bbox,
                                int(x),
                                int(y),
                                self.crop_size,
                                self.crop_size,
                                int(w),
                                int(h),
                            )
                        except Exception:
                            spatial_tile_bbox = []
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
                        "source_width": int(w),
                        "source_height": int(h),
                        "region_id": region_id,
                        "source_bbox": [float(value) for value in source_bbox],
                        "tile_bbox": [float(value) for value in spatial_tile_bbox],
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
        self.filtering_stats["tile_counts"] = {
            "retained": len(self.samples),
            "effective_train_length": len(self.samples) * self.train_repeat if self.split == "train" else len(self.samples),
        }

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

    def _validate_auxiliary_timestamps(self) -> None:
        """Enforce one shared month policy across cached OSM and spectral data."""

        manifests = (
            ("OSM", self.osm_manifest),
            ("spectral", self.spectral_manifest),
        )
        recorded: Dict[str, str] = {}
        for label, manifest in manifests:
            if not manifest.enabled:
                continue
            metadata = manifest.document.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            policy = _safe_text(
                metadata.get(
                    "timestamp_policy",
                    manifest.document.get("timestamp_policy", ""),
                )
            )
            if policy:
                recorded[label] = policy
            if self.osm_timestamp_policy and policy and policy != self.osm_timestamp_policy:
                raise ValueError(
                    f"Configured shared timestamp policy does not match the {label} manifest: "
                    f"{self.osm_timestamp_policy!r} != {policy!r}"
                )

        if len(set(recorded.values())) > 1:
            raise ValueError(
                "OSM and spectral manifests use different timestamp policies: "
                + ", ".join(f"{key}={value!r}" for key, value in recorded.items())
            )
        policy = self.osm_timestamp_policy or next(iter(recorded.values()), "")
        if not policy or month_timestamp is None:
            return
        for label, manifest in manifests:
            if not manifest.enabled:
                continue
            for stem, sample in manifest.records.items():
                for temporal_key in ("t1", "t2"):
                    if label == "OSM" and temporal_key not in self.osm_temporal_keys:
                        continue
                    record = sample.get(temporal_key)
                    if not isinstance(record, Mapping):
                        continue
                    image_month = _safe_text(
                        record.get("image_month", record.get("month", ""))
                    )
                    query_timestamp = _safe_text(record.get("query_timestamp", ""))
                    if image_month and query_timestamp:
                        expected = month_timestamp(image_month, policy)
                        if query_timestamp != expected:
                            raise ValueError(
                                f"{label} timestamp mismatch for {stem}/{temporal_key}: "
                                f"{query_timestamp!r} != {expected!r}"
                            )

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

    def _load_full_spectral_date(
        self,
        stem: str,
        temporal_key: str,
        expected_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        assessment = self.source_auxiliary[stem]["spectral"]
        available = bool(assessment["available"][temporal_key]) and self.use_spectral
        if not available:
            h, w = expected_shape
            return (
                np.zeros((h, w, len(SPECTRAL_CHANNEL_ORDER)), dtype=np.float32),
                np.zeros((h, w), dtype=np.float32),
            )
        cache_key = (stem, temporal_key)
        cached = self._spectral_cache.get(cache_key)
        if cached is not None:
            return cached
        sample = assessment["sample"]
        date_record = sample.get(temporal_key) if isinstance(sample, Mapping) else None
        if not isinstance(date_record, Mapping):
            raise RuntimeError(f"Missing {temporal_key} spectral record for {stem}")
        loaded = load_spectral_date(
            date_record,
            self.spectral_manifest,
            expected_shape=expected_shape,
            sample_record=sample,
            temporal_key=temporal_key,
        )
        self._spectral_cache.put(cache_key, loaded)
        return loaded

    def _load_tile(self, meta: Dict):
        t1 = np.asarray(Image.open(meta["path_a"]).convert("RGB"), dtype=np.float32) / 255.0
        t2 = np.asarray(Image.open(meta["path_b"]).convert("RGB"), dtype=np.float32) / 255.0
        lb = np.asarray(Image.open(meta["path_l"]).convert("L"), dtype=np.float32)
        expected_shape = (int(lb.shape[0]), int(lb.shape[1]))
        spectral_t1, valid_t1 = self._load_full_spectral_date(meta["stem"], "t1", expected_shape)
        spectral_t2, valid_t2 = self._load_full_spectral_date(meta["stem"], "t2", expected_shape)
        temporal_osm = self._resolve_temporal_osm(meta)
        y, x = meta["y"], meta["x"]
        t1 = t1[y:y + self.crop_size, x:x + self.crop_size]
        t2 = t2[y:y + self.crop_size, x:x + self.crop_size]
        lb = lb[y:y + self.crop_size, x:x + self.crop_size]
        spectral_t1 = spectral_t1[y:y + self.crop_size, x:x + self.crop_size]
        spectral_t2 = spectral_t2[y:y + self.crop_size, x:x + self.crop_size]
        valid_t1 = valid_t1[y:y + self.crop_size, x:x + self.crop_size]
        valid_t2 = valid_t2[y:y + self.crop_size, x:x + self.crop_size]
        osm_maps = temporal_osm.pop("osm_maps")
        return t1, t2, lb, spectral_t1, spectral_t2, valid_t1, valid_t2, osm_maps, temporal_osm

    def _augment(self, t1, t2, lb, spectral_t1, spectral_t2, valid_t1, valid_t2, osm_maps):
        arrays = [t1, t2, lb, spectral_t1, spectral_t2, valid_t1, valid_t2, osm_maps]
        if not self.random_aug:
            return tuple(arrays)
        if random.random() < 0.5:
            arrays = [np.flip(value, axis=1).copy() for value in arrays]
        if random.random() < 0.5:
            arrays = [np.flip(value, axis=0).copy() for value in arrays]
        k = random.randint(0, 3)
        if k > 0:
            arrays = [np.rot90(value, k=k, axes=(0, 1)).copy() for value in arrays]
        return tuple(arrays)

    def _resolve_temporal_osm(self, meta: Mapping[str, Any]) -> Dict[str, Any]:
        stem = str(meta["stem"])
        assessment = self.source_auxiliary[stem]["osm"]
        sample = assessment["sample"]
        spatial_tile_bbox = meta.get("tile_bbox", [])
        has_tile_bbox = isinstance(spatial_tile_bbox, (list, tuple)) and len(spatial_tile_bbox) == 4
        output: Dict[str, Any] = {
            "osm_maps": np.zeros(
                (self.crop_size, self.crop_size, len(OSM_SPATIAL_CHANNEL_ORDER)),
                dtype=np.float32,
            ),
            "osm_reliability": 0.0,
            "osm_reliability_source": "unavailable",
            "osm_spatial_reliability": 0.0,
            "osm_geometry_is_exact": 0,
            "osm_geometry_semantics": "unavailable",
            "osm_clip_category_gate": torch.zeros(5, dtype=torch.float32),
            "osm_clip_similarity": torch.zeros(5, dtype=torch.float32),
            "osm_clip_confidence": 0.0,
            "osm_clip_has_candidates": 0,
            "has_osm_clip_filter": 0,
        }
        for temporal_key in ("t1", "t2"):
            if temporal_key not in self.osm_temporal_keys:
                output[f"osm_struct_{temporal_key}"] = torch.zeros(16, dtype=torch.float32)
                output[f"osm_text_{temporal_key}"] = ""
                output[f"osm_texts_{temporal_key}"] = []
                output[f"has_osm_{temporal_key}"] = 0
                output[f"osm_status_{temporal_key}"] = "not_requested_runtime"
                output[f"osm_tile_feature_count_{temporal_key}"] = 0
                continue
            status = assessment["status"][temporal_key]
            available = bool(assessment["available"][temporal_key]) and self.use_temporal_osm
            date_record = sample.get(temporal_key) if isinstance(sample, Mapping) else None
            if available and isinstance(date_record, Mapping):
                paths = assessment.get("paths", {}).get(temporal_key, {})
                raw_path = paths.get("raw_geojson") if isinstance(paths, Mapping) else None
                if raw_path is not None and has_tile_bbox:
                    document = self._osm_geojson_cache.get_or_load(
                        (stem, temporal_key), Path(raw_path)
                    )
                    entry = summarize_tile(document, spatial_tile_bbox)
                    if temporal_key == "t2":
                        output["osm_maps"] = rasterize_feature_collection(
                            document,
                            spatial_tile_bbox,
                            self.crop_size,
                            self.crop_size,
                        )
                else:
                    # Compatibility fallback for a source-level manifest or a
                    # dataset without spatial metadata. Canonical temporal use
                    # validates and normally takes the raw-cache branch above.
                    entry = date_record
                struct = osm_struct_from_entry(entry, dim=16).float()
                text, phrases = osm_text_from_entry(entry, fallback="")
                tile_feature_count = int(
                    entry.get("tile_feature_count", date_record.get("feature_count", 0))
                )
                if temporal_key == "t2":
                    extraction_geometry = _safe_text(
                        date_record.get("extraction_geometry", "geometry")
                    ).lower()
                    reliability_value = date_record.get("osm_reliability")
                    if reliability_value is None:
                        # Confidence in the timestamped query/cache geometry,
                        # never a claim of physical OSM completeness.
                        reliability_value = 0.75 if extraction_geometry == "geometry" else 0.50
                        reliability_source = "derived_query_geometry"
                    else:
                        reliability_source = "manifest"
                    output["osm_reliability"] = float(
                        np.clip(float(reliability_value), 0.0, 1.0)
                    )
                    output["osm_reliability_source"] = reliability_source
                    geometry_semantics = _safe_text(
                        date_record.get("geometry_semantics", "")
                    )
                    geometry_is_exact = int(
                        extraction_geometry == "geometry"
                        and geometry_semantics != "feature_bounding_boxes_not_exact_geometries"
                    )
                    output["osm_geometry_is_exact"] = geometry_is_exact
                    output["osm_geometry_semantics"] = (
                        geometry_semantics
                        or (
                            "exact_osm_feature_geometries"
                            if geometry_is_exact
                            else "feature_bounding_boxes_not_exact_geometries"
                        )
                    )
                    # Bbox snapshots are still valid historical structured/late
                    # context, but they must not drive an early spatial branch as
                    # if road/building rectangles were exact geometries.
                    output["osm_spatial_reliability"] = (
                        output["osm_reliability"] if geometry_is_exact else 0.0
                    )
            else:
                struct = torch.zeros(16, dtype=torch.float32)
                text, phrases = "", []
                tile_feature_count = 0
            output[f"osm_struct_{temporal_key}"] = struct
            output[f"osm_text_{temporal_key}"] = text
            output[f"osm_texts_{temporal_key}"] = phrases
            output[f"has_osm_{temporal_key}"] = int(available)
            output[f"osm_status_{temporal_key}"] = status
            output[f"osm_tile_feature_count_{temporal_key}"] = tile_feature_count

        clip_record = self.osm_clip_filter_records.get(stem)
        if isinstance(clip_record, Mapping):
            output["osm_clip_category_gate"] = torch.tensor(
                clip_record.get("category_gate", [0.0] * 5),
                dtype=torch.float32,
            )
            output["osm_clip_similarity"] = torch.tensor(
                clip_record.get("similarities", [0.0] * 5),
                dtype=torch.float32,
            )
            output["osm_clip_confidence"] = float(
                clip_record.get("confidence", 0.0)
            )
            output["osm_clip_has_candidates"] = int(
                bool(clip_record.get("has_candidates", False))
            )
            output["has_osm_clip_filter"] = 1
        return output

    def _normalize_spectral(
        self,
        channels: np.ndarray,
        valid: np.ndarray,
        sensor_id: int = 0,
    ) -> np.ndarray:
        mode = self.index_normalization["mode"]
        if mode == "natural":
            return channels
        if mode == "sensor_train_stats":
            sensor_stats = self.index_normalization.get("sensor_stats", {})
            record = sensor_stats.get(int(sensor_id))
            if record is None:
                if np.any(np.asarray(valid) > 0):
                    raise ValueError(
                        "sensor_train_stats received valid spectral pixels with an "
                        f"unknown sensor_id={int(sensor_id)}; refusing to normalize "
                        "with another Landsat family's statistics"
                    )
                return np.zeros_like(channels, dtype=np.float32)
            mean = np.asarray(record["mean"], dtype=np.float32).reshape(
                1, 1, len(SPECTRAL_CHANNEL_ORDER)
            )
            std = np.asarray(record["std"], dtype=np.float32).reshape(
                1, 1, len(SPECTRAL_CHANNEL_ORDER)
            )
        else:
            mean = self.index_normalization["mean"].reshape(
                1, 1, len(SPECTRAL_CHANNEL_ORDER)
            )
            std = self.index_normalization["std"].reshape(
                1, 1, len(SPECTRAL_CHANNEL_ORDER)
            )
        normalized = (channels.astype(np.float32, copy=False) - mean) / std
        normalized = np.where(valid[..., None] > 0, normalized, 0.0)
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _as_string_list(value: Any) -> List[str]:
        if isinstance(value, (list, tuple)):
            return [_safe_text(item) for item in value if _safe_text(item)]
        text = _safe_text(value)
        return [text] if text else []

    @staticmethod
    def _spectral_sensor_id(value: Any) -> int:
        """Stable local sensor-family IDs; zero is unknown/unavailable."""

        text = _safe_text(value).lower().replace("_", "-")
        if any(token in text for token in ("landsat-7", "le07", "etm")):
            return 2
        if "landsat-5" in text or "lt05" in text or text in {"tm", "landsat tm"}:
            return 1
        if any(token in text for token in ("landsat-8", "lc08", "oli")):
            return 3
        return 0

    def _temporal_trace(self, stem: str) -> Dict[str, Any]:
        osm_sample = self.source_auxiliary[stem]["osm"].get("sample")
        spectral_sample = self.source_auxiliary[stem]["spectral"].get("sample")
        output: Dict[str, Any] = {}
        for temporal_key in ("t1", "t2"):
            osm_record = (
                osm_sample.get(temporal_key, {})
                if temporal_key in self.osm_temporal_keys and isinstance(osm_sample, Mapping)
                else {}
            )
            spectral_record = (
                spectral_sample.get(temporal_key, {})
                if isinstance(spectral_sample, Mapping)
                else {}
            )
            if not isinstance(osm_record, Mapping):
                osm_record = {}
            if not isinstance(spectral_record, Mapping):
                spectral_record = {}
            osm_month = _safe_text(osm_record.get("image_month", osm_record.get("month", "")))
            spectral_month = _safe_text(
                spectral_record.get(
                    "image_month",
                    spectral_record.get("month", spectral_record.get("target_month", "")),
                )
            )
            image_month = osm_month or spectral_month
            osm_timestamp = _safe_text(osm_record.get("query_timestamp", ""))
            spectral_timestamp = _safe_text(
                spectral_record.get(
                    "query_timestamp",
                    spectral_record.get("selected_timestamp", ""),
                )
            )
            scene_dates = self._as_string_list(
                spectral_record.get(
                    "scene_dates",
                    spectral_record.get(
                        "acquisition_dates",
                        spectral_record.get("acquisition_datetime", []),
                    ),
                )
            )
            scene_ids = self._as_string_list(
                spectral_record.get(
                    "scene_ids",
                    spectral_record.get("scenes", spectral_record.get("scene_id", [])),
                )
            )
            sensors = self._as_string_list(
                spectral_record.get("sensors", spectral_record.get("sensor", []))
            )
            actual_window = self._as_string_list(
                spectral_record.get("actual_window", spectral_record.get("window", []))
            )
            # Names specified by the runtime batch contract.
            output[f"{temporal_key}_image_month"] = image_month
            output[f"{temporal_key}_osm_query_timestamp"] = osm_timestamp
            output[f"{temporal_key}_spectral_scene_dates"] = scene_dates
            # Date-last aliases make programmatic paired-date handling convenient.
            output[f"osm_image_month_{temporal_key}"] = osm_month
            output[f"osm_query_timestamp_{temporal_key}"] = osm_timestamp
            output[f"spectral_image_month_{temporal_key}"] = spectral_month
            output[f"spectral_query_timestamp_{temporal_key}"] = spectral_timestamp
            output[f"spectral_scene_dates_{temporal_key}"] = scene_dates
            output[f"spectral_scene_ids_{temporal_key}"] = scene_ids
            output[f"spectral_sensors_{temporal_key}"] = sensors
            output[f"spectral_actual_window_{temporal_key}"] = actual_window
            output[f"spectral_provider_{temporal_key}"] = _safe_text(
                spectral_record.get("provider", "")
            )
            output[f"spectral_collection_{temporal_key}"] = _safe_text(
                spectral_record.get("collection", "")
            )
            output[f"spectral_cache_sha256_{temporal_key}"] = _safe_text(
                spectral_record.get("spectral_sha256", "")
            )
            output[f"spectral_valid_mask_sha256_{temporal_key}"] = _safe_text(
                spectral_record.get("valid_mask_sha256", "")
            )
            output[f"spectral_physical_bands_{temporal_key}"] = json.dumps(
                spectral_record.get("physical_bands", {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            output[f"spectral_surface_reflectance_units_{temporal_key}"] = _safe_text(
                spectral_record.get(
                    "surface_reflectance_units",
                    "scaled unitless surface reflectance",
                )
            )
        return output

    def _source_spectral_sensor_ids(self, stem: str) -> Tuple[int, int]:
        assessment = self.source_auxiliary[stem]["spectral"]
        sample = assessment.get("sample")
        if not isinstance(sample, Mapping):
            return 0, 0
        values: List[int] = []
        for temporal_key in ("t1", "t2"):
            record = sample.get(temporal_key, {})
            if not isinstance(record, Mapping):
                values.append(0)
                continue
            values.append(
                self._spectral_sensor_id(record.get("sensor", record.get("sensors", "")))
            )
        return int(values[0]), int(values[1])

    def __getitem__(self, idx: int):
        ridx = self._choose_index(idx)
        meta = self.samples[ridx]
        (
            t1,
            t2,
            lb,
            spectral_t1,
            spectral_t2,
            valid_t1,
            valid_t2,
            osm_maps,
            temporal_osm,
        ) = self._load_tile(meta)
        t1, t2, lb, spectral_t1, spectral_t2, valid_t1, valid_t2, osm_maps = self._augment(
            t1,
            t2,
            lb,
            spectral_t1,
            spectral_t2,
            valid_t1,
            valid_t2,
            osm_maps,
        )
        sensor_id_t1, sensor_id_t2 = self._source_spectral_sensor_ids(meta["stem"])
        spectral_t1 = self._normalize_spectral(spectral_t1, valid_t1, sensor_id_t1)
        spectral_t2 = self._normalize_spectral(spectral_t2, valid_t2, sensor_id_t2)

        gt = (lb > self.mask_threshold).astype(np.float32)
        if self.label_smoothing > 0:
            gt = gt * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        patch_id = meta["patch_id"]
        temporal_trace = self._temporal_trace(meta["stem"])
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

        # When no explicit legacy single-OSM file is configured, expose T2 as
        # the legacy alias.  This never fabricates a missing T1 and keeps old
        # single-OSM consumers usable with a paired manifest.
        if self.osm_helper is None and self.osm_manifest.enabled:
            if temporal_osm["has_osm_t2"]:
                osm_struct = temporal_osm["osm_struct_t2"].clone()
                has_osm_text = int(temporal_osm["has_osm_t2"])
                osm_texts = list(temporal_osm["osm_texts_t2"])
                main_text = temporal_osm["osm_text_t2"] or self.fixed_text
            elif self.temporal_osm_mode != "t2" and temporal_osm["has_osm_t1"]:
                osm_struct = temporal_osm["osm_struct_t1"].clone()
                has_osm_text = int(temporal_osm["has_osm_t1"])
                osm_texts = list(temporal_osm["osm_texts_t1"])
                main_text = temporal_osm["osm_text_t1"] or self.fixed_text

        enc = self.tokenizer(
            main_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        image_t1 = torch.from_numpy(t1.transpose(2, 0, 1).astype(np.float32))
        image_t2 = torch.from_numpy(t2.transpose(2, 0, 1).astype(np.float32))
        gt_mask = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0)
        spectral_channels_t1 = torch.from_numpy(
            spectral_t1.transpose(2, 0, 1).astype(np.float32)
        )
        spectral_channels_t2 = torch.from_numpy(
            spectral_t2.transpose(2, 0, 1).astype(np.float32)
        )
        spectral_reflectance_t1 = spectral_channels_t1[:3]
        spectral_reflectance_t2 = spectral_channels_t2[:3]
        spectral_indices_t1 = spectral_channels_t1[3:5]
        spectral_indices_t2 = spectral_channels_t2[3:5]
        spectral_valid_t1 = torch.from_numpy(valid_t1.astype(np.float32)).unsqueeze(0)
        spectral_valid_t2 = torch.from_numpy(valid_t2.astype(np.float32)).unsqueeze(0)
        osm_maps_tensor = torch.from_numpy(
            osm_maps.transpose(2, 0, 1).astype(np.float32)
        )
        spectral_assessment = self.source_auxiliary[meta["stem"]]["spectral"]
        source_spectral_available_t1 = int(
            bool(spectral_assessment["available"]["t1"]) and self.use_spectral
        )
        source_spectral_available_t2 = int(
            bool(spectral_assessment["available"]["t2"]) and self.use_spectral
        )
        source_bbox = meta.get("source_bbox", [])
        spatial_tile_bbox = meta.get("tile_bbox", [])
        source_bbox_tensor = torch.tensor(
            source_bbox if len(source_bbox) == 4 else [float("nan")] * 4,
            dtype=torch.float64,
        )
        tile_bbox_tensor = torch.tensor(
            spatial_tile_bbox if len(spatial_tile_bbox) == 4 else [float("nan")] * 4,
            dtype=torch.float64,
        )
        valid_fraction_t1 = spectral_assessment["valid_fraction"]["t1"]
        valid_fraction_t2 = spectral_assessment["valid_fraction"]["t2"]
        spectral_pair_reliability = float(
            min(
                float(spectral_valid_t1.mean()),
                float(spectral_valid_t2.mean()),
            )
            if source_spectral_available_t1 and source_spectral_available_t2
            else 0.0
        )
        legacy_indices_only = (
            spectral_assessment.get("compatibility_tier") == "legacy_indices_only"
        )
        # Sensor IDs were resolved before normalization so each date uses its
        # own TRAIN-only Landsat-family statistics when configured.
        result = {
            "image": [image_t2],
            "image_t1": [image_t1],
            "image_t2": [image_t2],
            "gt_mask": gt_mask,
            "change_mask": gt_mask,
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
            # Canonical paired temporal OSM fields.
            "osm_struct_t1": temporal_osm["osm_struct_t1"],
            "osm_struct_t2": temporal_osm["osm_struct_t2"],
            "osm_t2_struct": temporal_osm["osm_struct_t2"],
            "osm_maps": osm_maps_tensor,
            "osm_reliability": torch.tensor(
                temporal_osm["osm_reliability"], dtype=torch.float32
            ),
            "osm_clip_category_gate": temporal_osm["osm_clip_category_gate"],
            "osm_clip_similarity": temporal_osm["osm_clip_similarity"],
            "osm_clip_confidence": torch.tensor(
                temporal_osm["osm_clip_confidence"], dtype=torch.float32
            ),
            "osm_clip_has_candidates": torch.tensor(
                temporal_osm["osm_clip_has_candidates"], dtype=torch.long
            ),
            "has_osm_clip_filter": torch.tensor(
                temporal_osm["has_osm_clip_filter"], dtype=torch.long
            ),
            "osm_reliability_source": temporal_osm["osm_reliability_source"],
            "osm_spatial_reliability": torch.tensor(
                temporal_osm["osm_spatial_reliability"], dtype=torch.float32
            ),
            "osm_geometry_is_exact": torch.tensor(
                temporal_osm["osm_geometry_is_exact"], dtype=torch.long
            ),
            "osm_geometry_semantics": temporal_osm["osm_geometry_semantics"],
            "osm_spatial_channel_order": list(OSM_SPATIAL_CHANNEL_ORDER),
            "has_osm_t1": torch.tensor(temporal_osm["has_osm_t1"], dtype=torch.long),
            "has_osm_t2": torch.tensor(temporal_osm["has_osm_t2"], dtype=torch.long),
            "osm_status_t1": temporal_osm["osm_status_t1"],
            "osm_status_t2": temporal_osm["osm_status_t2"],
            "osm_tile_feature_count_t1": torch.tensor(
                temporal_osm["osm_tile_feature_count_t1"], dtype=torch.long
            ),
            "osm_tile_feature_count_t2": torch.tensor(
                temporal_osm["osm_tile_feature_count_t2"], dtype=torch.long
            ),
            "osm_text_t1": temporal_osm["osm_text_t1"],
            "osm_text_t2": temporal_osm["osm_text_t2"],
            "osm_temporal_text": " [T1] " + temporal_osm["osm_text_t1"]
            + " [T2] " + temporal_osm["osm_text_t2"],
            "osm_patch_id": patch_id,
            # Canonical paired physical/index fields.  The five-channel tensor
            # is the only model input; the views below keep older analysis code
            # usable without reconstructing physical bands from RGB.
            "spectral_channels_t1": spectral_channels_t1,
            "spectral_channels_t2": spectral_channels_t2,
            "spectral_reflectance_t1": spectral_reflectance_t1,
            "spectral_reflectance_t2": spectral_reflectance_t2,
            "spectral_indices_t1": spectral_indices_t1,
            "spectral_indices_t2": spectral_indices_t2,
            "spectral_valid_t1": spectral_valid_t1,
            "spectral_valid_t2": spectral_valid_t2,
            "has_spectral_t1": torch.tensor(source_spectral_available_t1, dtype=torch.long),
            "has_spectral_t2": torch.tensor(source_spectral_available_t2, dtype=torch.long),
            "spectral_status_t1": spectral_assessment["status"]["t1"],
            "spectral_status_t2": spectral_assessment["status"]["t2"],
            "spectral_valid_fraction_t1": torch.tensor(
                -1.0 if valid_fraction_t1 is None else float(valid_fraction_t1), dtype=torch.float32
            ),
            "spectral_valid_fraction_t2": torch.tensor(
                -1.0 if valid_fraction_t2 is None else float(valid_fraction_t2), dtype=torch.float32
            ),
            "spectral_tile_valid_fraction_t1": spectral_valid_t1.mean(),
            "spectral_tile_valid_fraction_t2": spectral_valid_t2.mean(),
            "spectral_reliability": torch.tensor(
                spectral_pair_reliability, dtype=torch.float32
            ),
            "sensor_id_t1": torch.tensor(sensor_id_t1, dtype=torch.long),
            "sensor_id_t2": torch.tensor(sensor_id_t2, dtype=torch.long),
            "spectral_has_physical_bands": torch.tensor(
                int(not legacy_indices_only), dtype=torch.long
            ),
            "spectral_provenance_tier": (
                "legacy_indices_only_unverified_physical_bands"
                if legacy_indices_only
                else "verified_physical_bands_and_indices"
            ),
            # Source/tile traceability for reproducible retained lists and maps.
            "source_pair_id": meta["stem"],
            "filename": meta["stem"] + ".png",
            "tile_id": patch_id,
            "split": self.split,
            "region_id": torch.tensor(int(meta.get("region_id", -1)), dtype=torch.long),
            "source_bbox": source_bbox_tensor,
            "tile_bbox": tile_bbox_tensor,
            "has_spatial_metadata": torch.tensor(int(len(source_bbox) == 4), dtype=torch.long),
            "tile_window": torch.tensor(
                [meta["x"], meta["y"], self.crop_size, self.crop_size], dtype=torch.long
            ),
            "tile_xy": torch.tensor([meta["x"], meta["y"]], dtype=torch.long),
            "tile_x": torch.tensor(meta["x"], dtype=torch.long),
            "tile_y": torch.tensor(meta["y"], dtype=torch.long),
            "source_width": torch.tensor(meta["source_width"], dtype=torch.long),
            "source_height": torch.tensor(meta["source_height"], dtype=torch.long),
            "osm_manifest_sha256": self.osm_manifest.sha256,
            "spectral_manifest_sha256": self.spectral_manifest.sha256,
            "osm_manifest_schema_version": self.osm_manifest.schema_version,
            "osm_clip_filter_schema_version": _safe_text(
                self.osm_clip_filter_document.get("schema_version", "")
            ),
            "osm_clip_filter_model": _safe_text(
                self.osm_clip_filter_document.get("clip_model", "")
            ),
            "spectral_manifest_schema_version": self.spectral_manifest.schema_version,
            "spectral_channel_order": list(SPECTRAL_CHANNEL_ORDER),
            "spectral_reflectance_order": list(SURFACE_REFLECTANCE_ORDER),
            "spectral_index_order": list(SPECTRAL_INDEX_ORDER),
            "spectral_index_normalization": self.index_normalization["mode"],
        }
        result.update(temporal_trace)
        # Lightweight compatibility aliases.  The custom collate method below
        # reuses canonical batch tensors instead of stacking duplicate copies.
        result["osm_t1_struct"] = result["osm_struct_t1"]
        result["osm_t2_struct"] = result["osm_struct_t2"]
        result["spectral_t1"] = result["spectral_indices_t1"]
        result["spectral_t2"] = result["spectral_indices_t2"]
        result["spectral_features_t1"] = result["spectral_channels_t1"]
        result["spectral_features_t2"] = result["spectral_channels_t2"]
        result["spectral_mask_t1"] = result["spectral_valid_t1"]
        result["spectral_mask_t2"] = result["spectral_valid_t2"]
        return result

    @staticmethod
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default collation with zero-copy aliases for large spatial tensors."""

        alias_to_canonical = {
            "image": "image_t2",
            "change_mask": "gt_mask",
            "osm_t1_struct": "osm_struct_t1",
            "osm_t2_struct": "osm_struct_t2",
            "spectral_t1": "spectral_indices_t1",
            "spectral_t2": "spectral_indices_t2",
            "spectral_features_t1": "spectral_channels_t1",
            "spectral_features_t2": "spectral_channels_t2",
            "spectral_mask_t1": "spectral_valid_t1",
            "spectral_mask_t2": "spectral_valid_t2",
        }
        spectral_views = {
            "spectral_reflectance_t1": ("spectral_channels_t1", slice(0, 3)),
            "spectral_reflectance_t2": ("spectral_channels_t2", slice(0, 3)),
            "spectral_indices_t1": ("spectral_channels_t1", slice(3, 5)),
            "spectral_indices_t2": ("spectral_channels_t2", slice(3, 5)),
        }
        list_metadata_fields = {
            "osm_texts",
            "t1_spectral_scene_dates",
            "t2_spectral_scene_dates",
            "spectral_scene_dates_t1",
            "spectral_scene_dates_t2",
            "spectral_scene_ids_t1",
            "spectral_scene_ids_t2",
            "spectral_sensors_t1",
            "spectral_sensors_t2",
            "spectral_actual_window_t1",
            "spectral_actual_window_t2",
            "spectral_channel_order",
            "spectral_reflectance_order",
            "spectral_index_order",
            "osm_spatial_channel_order",
        }
        list_metadata = {
            key: [sample.get(key, []) for sample in batch] for key in list_metadata_fields
        }
        # LEVIR historically wraps these values in a one-element list at item
        # level.  Present the same clean batch contract as the other project
        # datamodules without changing individual-item compatibility.
        for key in ("text", "main_text", "file"):
            list_metadata[key] = [
                value[0] if isinstance(value, list) and len(value) == 1 else value
                for value in (sample.get(key, "") for sample in batch)
            ]
        list_metadata["osm_texts"] = [
            value[0]
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
            else value
            for value in (sample.get("osm_texts", []) for sample in batch)
        ]
        passthrough_fields = list_metadata_fields | {"text", "main_text", "file"}
        trimmed = [
            {
                key: value
                for key, value in sample.items()
                if key not in alias_to_canonical
                and key not in spectral_views
                and key not in passthrough_fields
            }
            for sample in batch
        ]
        collated = default_collate(trimmed)
        collated.update(list_metadata)
        for key, (canonical, channel_slice) in spectral_views.items():
            collated[key] = collated[canonical][:, channel_slice]
        for alias, canonical in alias_to_canonical.items():
            collated[alias] = collated[canonical]
        return collated