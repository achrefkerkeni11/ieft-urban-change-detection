import os
import json
from typing import Any, Dict, List, Tuple


def _clean_text(x: Any) -> str:
    s = str(x).strip()
    s = " ".join(s.split())
    return s


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_clip_debug_json(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLIP debug JSON introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("CLIP debug JSON doit être un dict patch_id -> debug_record")
    return {str(k): v for k, v in data.items()}


def extract_candidate_texts_from_record(record: Any) -> List[str]:
    if isinstance(record, str):
        return [_clean_text(record)] if _clean_text(record) else []

    if isinstance(record, list):
        return _dedup_keep_order([_clean_text(x) for x in record if _clean_text(x)])

    if isinstance(record, dict):
        phrases = record.get("phrases", [])
        if isinstance(phrases, str):
            phrases = [phrases]
        elif not isinstance(phrases, list):
            phrases = []

        texts = [_clean_text(x) for x in phrases if _clean_text(x)]
        texts = _dedup_keep_order(texts)

        if len(texts) > 0:
            return texts

        fallback = []
        for key in ["text_v12", "summary", "source_text"]:
            v = _clean_text(record.get(key, ""))
            if v:
                fallback.append(v)
        return _dedup_keep_order(fallback)

    return []


def extract_ranked_texts_from_clip_debug(debug_record: Any) -> List[Tuple[str, float]]:
    if not isinstance(debug_record, dict):
        return []

    ranked = debug_record.get("ranked_texts_with_scores", [])
    out = []

    if isinstance(ranked, list):
        for item in ranked:
            if isinstance(item, dict):
                text = _clean_text(item.get("text", ""))
                score = float(item.get("score", 0.0))
                if text:
                    out.append((text, score))

    if len(out) > 0:
        return out

    kept = debug_record.get("kept_texts", [])
    kept_scores = debug_record.get("kept_scores", [])

    if isinstance(kept, list) and isinstance(kept_scores, list):
        for t, s in zip(kept, kept_scores):
            text = _clean_text(t)
            if text:
                out.append((text, float(s)))

    return out


def apply_clip_filter_to_record(
    raw_record: Any,
    clip_debug_record: Any,
    clip_top_k: int = 2,
    clip_min_score: float = -1e9,
    clip_fallback_mode: str = "keep_original",
) -> Any:
    """
    Intégration douce de CLIP:
    - on NE remplace PAS tout le record riche
    - on filtre / réordonne surtout le champ 'phrases'
    - on garde text_v12 / summary / tags / source_text intacts
    """
    clip_top_k = max(1, int(clip_top_k))
    clip_fallback_mode = str(clip_fallback_mode).strip().lower()

    ranked = extract_ranked_texts_from_clip_debug(clip_debug_record)
    ranked = [(t, s) for (t, s) in ranked if s >= float(clip_min_score)]
    kept_texts = [t for (t, _) in ranked[:clip_top_k]]
    kept_texts = _dedup_keep_order(kept_texts)

    if len(kept_texts) == 0:
        if clip_fallback_mode == "drop":
            kept_texts = []
        else:
            kept_texts = extract_candidate_texts_from_record(raw_record)[:clip_top_k]

    if isinstance(raw_record, str):
        if len(kept_texts) == 0:
            return raw_record
        return kept_texts[0]

    if isinstance(raw_record, list):
        if len(kept_texts) == 0:
            return raw_record
        return kept_texts

    if isinstance(raw_record, dict):
        out = dict(raw_record)
        if len(kept_texts) > 0:
            out["phrases"] = kept_texts
        out["clip_kept_texts"] = kept_texts
        out["clip_top_k_used"] = len(kept_texts)
        return out

    return raw_record