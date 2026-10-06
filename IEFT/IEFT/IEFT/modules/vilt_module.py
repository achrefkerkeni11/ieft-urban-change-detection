# -*- coding: utf-8 -*-
"""IEFT model used by the final memory-safe 118k LEVIR-CD protocol."""

import math
import os
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from IEFT.modules import objectives
from IEFT.modules.cross_modal_fusion import BidirectionalFIEResidual
from IEFT.modules.instance_targets import InstanceLoss
from IEFT.modules.multiscale_change_head import MultiScalePixelObjectChangeDecoder
from IEFT.modules.safe_spectral_residual import SafeSpectralLateResidual
from IEFT.modules.safe_osm_guidance import SafeOSMLateGuidance
from IEFT.modules.temporal_auxiliary import (
    OSMT2EarlyAdapter,
    PairedOSMFusion,
    TemporalIndexAdapter,
)


def _canonical_dense_logits(
    dense_logits: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Return dense semantic logits as ``[B,H,W]`` without broadcasting B."""

    if dense_logits.ndim == 4 and dense_logits.shape[1] == 1:
        dense_logits = dense_logits.squeeze(1)
    if dense_logits.ndim != 3 or dense_logits.shape[0] != int(batch_size):
        raise ValueError(
            "Dense semantic logits must be [B,H,W] after removing the singleton "
            f"channel, got {tuple(dense_logits.shape)} for batch size {batch_size}"
        )
    return dense_logits


def _unwrap_checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Extract a model state dictionary from common checkpoint containers."""

    current: Mapping[str, Any] = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        value = current.get(key) if isinstance(current, Mapping) else None
        if isinstance(value, Mapping):
            current = value
            break
    return current  # type: ignore[return-value]


def _best_prefix_normalization(
    state_dict: Mapping[str, torch.Tensor],
    target_keys: Sequence[str],
) -> Mapping[str, torch.Tensor]:
    """Strip a wrapper prefix only when it increases target-key overlap."""

    target = set(target_keys)
    best = OrderedDict(state_dict)
    best_overlap = len(target.intersection(best.keys()))
    for prefix in ("module.", "model."):
        candidate = OrderedDict(
            (key[len(prefix):] if key.startswith(prefix) else key, value)
            for key, value in state_dict.items()
        )
        overlap = len(target.intersection(candidate.keys()))
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap
    return best


def load_compatible_checkpoint(
    module: nn.Module,
    checkpoint: Union[str, os.PathLike, Mapping[str, Any]],
    map_location: Union[str, torch.device] = "cpu",
    allowed_auxiliary_prefixes: Sequence[str] = (
        "spectral_adapter.",
        "osm_temporal_fusion.",
        "osm_t2_early_adapter.",
        "osm_t2_late_residual_scale",
        "bidirectional_fie_adapters.",
        "ms_change_decoder.pixel_decoder.instance_head.",
        # These names are deliberately retained for old single-OSM checkpoints.
        # Treat them as auxiliary when moving between image-only and OSM modes.
        "osm_struct_proj.",
        "osm_reliability_head.",
        "osm_global_head.",
        "osm_patch_query.",
        "osm_patch_scale.",
        "osm_dense_bias_head.",
    ),
    minimum_parameter_coverage: float = 0.50,
    strict_compatibility: bool = False,
) -> Dict[str, Any]:
    """Load every shape-compatible parameter and return a detailed report.

    ``strict=False`` alone still raises for tensor-shape mismatches and can hide a
    nearly unrelated checkpoint behind a long missing-key list.  This helper
    filters shape mismatches explicitly, reports every incompatibility category,
    and rejects checkpoints that cover too little of the requested model.
    """

    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint_obj = torch.load(os.fspath(checkpoint), map_location=map_location)
        checkpoint_name = os.fspath(checkpoint)
    elif isinstance(checkpoint, Mapping):
        checkpoint_obj = checkpoint
        checkpoint_name = "<in-memory checkpoint>"
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    if not isinstance(checkpoint_obj, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_name} does not contain a mapping")

    raw_state = _unwrap_checkpoint_state_dict(checkpoint_obj)
    if not isinstance(raw_state, Mapping):
        raise TypeError(f"Checkpoint {checkpoint_name} has no usable state dictionary")
    target_state = module.state_dict()
    normalized = _best_prefix_normalization(raw_state, list(target_state.keys()))

    compatible: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    unexpected: List[str] = []
    shape_mismatches: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for key, value in normalized.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        if not torch.is_tensor(value):
            unexpected.append(key)
            continue
        expected = target_state[key]
        if tuple(value.shape) != tuple(expected.shape):
            shape_mismatches[key] = {
                "checkpoint": tuple(value.shape),
                "model": tuple(expected.shape),
            }
            continue
        compatible[key] = value

    incompatible = module.load_state_dict(compatible, strict=False)
    missing = list(incompatible.missing_keys)
    # load_state_dict cannot add new unexpected keys because we pre-filtered,
    # but retain the union defensively.
    unexpected = sorted(set(unexpected).union(incompatible.unexpected_keys))

    total_numel = sum(int(value.numel()) for value in target_state.values())
    loaded_numel = sum(int(target_state[key].numel()) for key in compatible)
    coverage = float(loaded_numel) / float(max(1, total_numel))

    def is_auxiliary(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in allowed_auxiliary_prefixes)

    non_aux_missing = [key for key in missing if not is_auxiliary(key)]
    non_aux_unexpected = [key for key in unexpected if not is_auxiliary(key)]
    non_aux_shape_mismatches = {
        key: value for key, value in shape_mismatches.items() if not is_auxiliary(key)
    }
    report: Dict[str, Any] = {
        "checkpoint": checkpoint_name,
        "loaded_keys": list(compatible.keys()),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "non_auxiliary_missing_keys": non_aux_missing,
        "non_auxiliary_unexpected_keys": non_aux_unexpected,
        "non_auxiliary_shape_mismatches": non_aux_shape_mismatches,
        "parameter_coverage": coverage,
    }

    summary = (
        f"Checkpoint compatibility for {checkpoint_name}: loaded={len(compatible)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, "
        f"shape_mismatches={len(shape_mismatches)}, parameter_coverage={coverage:.2%}."
    )
    if missing or unexpected or shape_mismatches:
        warnings.warn(summary)

        def preview(keys: Sequence[str], limit: int = 20) -> str:
            shown = list(keys[:limit])
            suffix = f" ... (+{len(keys) - limit} more)" if len(keys) > limit else ""
            return ", ".join(shown) + suffix

        if missing:
            warnings.warn("Missing checkpoint keys: " + preview(missing))
        if unexpected:
            warnings.warn("Unexpected checkpoint keys: " + preview(unexpected))
        if shape_mismatches:
            mismatch_rows = [
                f"{key}: checkpoint{value['checkpoint']} != model{value['model']}"
                for key, value in shape_mismatches.items()
            ]
            warnings.warn("Checkpoint shape mismatches: " + preview(mismatch_rows))

    if coverage < float(minimum_parameter_coverage):
        raise RuntimeError(
            summary
            + f" Coverage is below the required {float(minimum_parameter_coverage):.2%}."
        )
    if strict_compatibility and (
        non_aux_missing or non_aux_unexpected or non_aux_shape_mismatches
    ):
        raise RuntimeError(summary + " Non-auxiliary incompatibilities are not allowed in strict mode.")
    return report


def _encoder_checkpoint_candidates(
    state_dict: Mapping[str, torch.Tensor],
    target_keys: Sequence[str],
) -> Tuple[Mapping[str, torch.Tensor], str]:
    """Normalize standalone timm or full IEFT checkpoint keys for an encoder."""

    target = set(target_keys)
    prefixes = (
        "",
        "module.",
        "model.",
        "encoder.",
        "transformer.",
        "module.encoder.",
        "module.transformer.",
        "model.encoder.",
    )
    best: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    best_prefix = ""
    best_overlap = -1
    for prefix in prefixes:
        candidate = OrderedDict()
        for key, value in state_dict.items():
            if prefix and not key.startswith(prefix):
                continue
            normalized = key[len(prefix):] if prefix else key
            candidate[normalized] = value
        overlap = len(target.intersection(candidate.keys()))
        # On equal overlap prefer the more selective subtree.  A ViLT
        # pretraining checkpoint contains text/task heads alongside
        # ``transformer.*``; those unrelated tensors must not be reported as
        # visual-encoder keys merely because an empty wrapper also overlaps.
        if overlap > best_overlap or (
            overlap == best_overlap and len(prefix) > len(best_prefix)
        ):
            best = candidate
            best_prefix = prefix
            best_overlap = overlap
    return best, best_prefix


def load_local_encoder_checkpoint(
    encoder: nn.Module,
    checkpoint_path: Union[str, os.PathLike],
    partial_load: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load a local visual encoder checkpoint without invoking timm downloads."""

    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Local visual encoder checkpoint does not exist: {checkpoint_path}")
    # This path is explicitly restricted to a user-selected local checkpoint.
    # Old PyTorch-Lightning files can contain callback metadata that the newer
    # weights-only unpickler rejects even though the state_dict tensors are
    # valid, so request the legacy local-file loader explicitly.  The caller is
    # responsible for pinning/checking the file hash before using an untrusted
    # file; no remote identifier is accepted here.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:  # PyTorch releases predating ``weights_only``.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Visual encoder checkpoint is not a mapping: {checkpoint_path}")
    raw_state = _unwrap_checkpoint_state_dict(checkpoint)
    target_state = encoder.state_dict()
    normalized, stripped_prefix = _encoder_checkpoint_candidates(
        raw_state, list(target_state.keys())
    )

    compatible: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    unexpected: List[str] = []
    shape_mismatches: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for key, value in normalized.items():
        if key not in target_state or not torch.is_tensor(value):
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches[key] = {
                "checkpoint": tuple(value.shape),
                "model": tuple(target_state[key].shape),
            }
            continue
        compatible[key] = value

    missing = [key for key in target_state if key not in compatible]
    parameter_numel = {
        key: int(value.numel()) for key, value in encoder.named_parameters()
    }
    total_numel = sum(parameter_numel.values())
    loaded_numel = sum(parameter_numel.get(key, 0) for key in compatible)
    coverage = float(loaded_numel) / float(max(1, total_numel))
    report: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "stripped_prefix": stripped_prefix,
        "loaded_keys": list(compatible.keys()),
        "matched_key_count": len(compatible),
        "target_key_count": len(target_state),
        "missing_keys": missing,
        "unexpected_keys": sorted(set(unexpected)),
        "shape_mismatches": shape_mismatches,
        "shape_mismatch_count": len(shape_mismatches),
        "encoder_parameter_numel_matched": loaded_numel,
        "encoder_parameter_numel_total": total_numel,
        "parameter_coverage": coverage,
        "partial_load": bool(partial_load),
        # These target tensors are intentionally left exactly as the encoder
        # constructor initialized them.  In particular, no patch-kernel or
        # positional-embedding resizing/interpolation is performed here.
        "retained_initialization_keys": missing,
    }
    if not compatible:
        raise RuntimeError(
            f"Visual encoder checkpoint {checkpoint_path} has no shape-compatible encoder keys."
        )
    if not partial_load and (missing or report["unexpected_keys"] or shape_mismatches):
        raise RuntimeError(
            f"Strict visual encoder load failed for {checkpoint_path}: "
            f"missing={missing}, unexpected={report['unexpected_keys']}, "
            f"shape_mismatches={shape_mismatches}."
        )
    # Mutate the encoder only after strict validation has succeeded.  This keeps
    # callers from receiving a half-loaded module when strict mode raises.
    encoder.load_state_dict(compatible, strict=False)
    if missing or report["unexpected_keys"] or shape_mismatches:
        warnings.warn(
            f"Partial visual encoder load from {checkpoint_path}: "
            f"coverage={coverage:.2%}, missing={missing}, "
            f"unexpected={report['unexpected_keys']}, shape_mismatches={shape_mismatches}."
        )
    return report


# ---------------------------------------------------------------------------
# FIEBlock — fusion bitemporelle inspirée de IEFT/FIE
# ---------------------------------------------------------------------------

class FIEBlock(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, dropout: float = 0.20):
        super().__init__()
        self.ca = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.sa = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        o, _ = self.ca(self.n1(v), t, t)
        v = v + o
        o, _ = self.sa(self.n2(v), self.n2(v), self.n2(v))
        v = v + o
        return v + self.ff(self.n3(v))


# ---------------------------------------------------------------------------
# FrozenCLIPTextBranch — chargement défensif avec double fallback
# ---------------------------------------------------------------------------

class FrozenCLIPTextBranch(nn.Module):
    """
    Tente de charger CLIP de 2 façons différentes avant de se désactiver.
    Tentative 1 : CLIPTextModelWithProjection  (transformers >= 4.26)
    Tentative 2 : CLIPModel complet → text_model + text_projection
    Si les deux échouent : self.enabled = False, pas d'exception.
    """

    def __init__(self, model_name: str, max_length: int = 48, freeze_text: bool = True):
        super().__init__()
        self.enabled = False
        self.model_name = model_name
        self.max_length = int(max_length)
        self.output_dim = 512
        self.tokenizer = None
        self.text_model = None
        self._use_full_model_fallback = False

        # --- Tentative 1 : CLIPTextModelWithProjection ---
        try:
            from transformers import CLIPTokenizerFast, CLIPTextModelWithProjection
            tok = CLIPTokenizerFast.from_pretrained(model_name)
            mdl = CLIPTextModelWithProjection.from_pretrained(model_name)
            self.output_dim = int(mdl.config.projection_dim)
            self.tokenizer = tok
            self.text_model = mdl
            if freeze_text:
                for p in self.text_model.parameters():
                    p.requires_grad = False
                self.text_model.eval()
            self.enabled = True
            return
        except Exception as e1:
            warnings.warn(
                f"[CLIP] CLIPTextModelWithProjection non disponible ({e1}). "
                "Tentative fallback CLIPModel..."
            )

        # --- Tentative 2 : CLIPModel complet ---
        try:
            from transformers import CLIPTokenizerFast, CLIPModel
            tok = CLIPTokenizerFast.from_pretrained(model_name)
            full = CLIPModel.from_pretrained(model_name)
            self.text_model = full.text_model
            # On garde text_projection comme attribut séparé pour le fallback
            self.text_projection_w = full.text_projection   # nn.Linear
            self.output_dim = int(full.config.projection_dim)
            self.tokenizer = tok
            if freeze_text:
                for p in self.text_model.parameters():
                    p.requires_grad = False
                self.text_model.eval()
                for p in self.text_projection_w.parameters():
                    p.requires_grad = False
                self.text_projection_w.eval()
            self._use_full_model_fallback = True
            self.enabled = True
            return
        except Exception as e2:
            warnings.warn(
                f"[CLIP] CLIPModel fallback aussi échoué ({e2}). "
                "CLIP définitivement désactivé — signal CLIP = zéro."
            )

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        if not self.enabled or not texts:
            return torch.zeros((0, self.output_dim), dtype=torch.float32, device=device)

        toks = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        grad_on = any(p.requires_grad for p in self.text_model.parameters())
        with torch.set_grad_enabled(grad_on):
            if self._use_full_model_fallback:
                out = self.text_model(**toks)
                # Pooled = hidden state au token EOS
                eos_idx = toks["input_ids"].argmax(dim=-1)
                pooled = out.last_hidden_state[
                    torch.arange(out.last_hidden_state.shape[0], device=device), eos_idx
                ].float()
                feats = self.text_projection_w(pooled)
            else:
                out = self.text_model(**toks)
                feats = out.text_embeds

        return F.normalize(feats.float(), dim=-1)


# ---------------------------------------------------------------------------
# OSMSemanticInjector
# Injection du signal sémantique OSM à 3 niveaux du pipeline :
#   1. Global  — biais sur le logit du token TEMP
#   2. Patch   — pondération spatiale par similarité cosinus avec diff ViT
#   3. Dense   — biais interpolé sur la carte de probabilité pixel
#
# Principe : "Semantic information from the geographic database is injected
# at various stages of the pipeline to provide contextual guidance."
# La décision finale reste image-first. OSM module légèrement le signal.
# ---------------------------------------------------------------------------

class OSMSemanticInjector(nn.Module):
    def __init__(self, osm_struct_dim: int, osm_struct_hidden: int, patch_dim: int):
        super().__init__()
        # Projection OSM 16D → espace commun (patch_dim)
        # LayerNorm finale pour stabiliser la norme du signal
        self.struct_proj = nn.Sequential(
            nn.Linear(osm_struct_dim, osm_struct_hidden),
            nn.LayerNorm(osm_struct_hidden),
            nn.GELU(),
            nn.Linear(osm_struct_hidden, patch_dim),
            nn.LayerNorm(patch_dim),
        )
        # Fiabilité du signal OSM pour ce patch
        self.reliability_head = nn.Sequential(
            nn.Linear(patch_dim, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1),
        )
        # Niveau 1 : guidage global (TEMP token)
        self.global_head = nn.Sequential(
            nn.Linear(patch_dim * 2, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1),
        )
        # Niveau 2 : guidage patch (cosine similarity avec diff temporelle)
        self.patch_query = nn.Linear(patch_dim, patch_dim)
        self.patch_scale = nn.Sequential(
            nn.Linear(patch_dim * 2, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1),
        )
        # Niveau 3 : guidage dense (biais spatial)
        self.dense_bias_head = nn.Sequential(
            nn.Linear(patch_dim * 2, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1),
        )

    def forward(
        self,
        osm_struct: torch.Tensor,       # (B, 16)
        has_osm_text: torch.Tensor,     # (B,)   — masque binaire
        temp_feat: torch.Tensor,         # (B, D) — token TEMP fusionné
        top_diff: torch.Tensor,          # (B, N, D) — |feat_t2 - feat_t1| dernier niveau ViT
        dense_logits_shape: tuple,       # (H, W)
        grid_size: int,
        reliability_bias: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        device = temp_feat.device
        has_osm = has_osm_text.float().view(-1, 1).to(device)

        # Projection du vecteur géosémantique
        osm_feat = self.struct_proj(osm_struct.to(device).float())      # (B, D)

        # Fiabilité du signal — masquée si pas de données OSM pour ce patch
        rel_raw = torch.sigmoid(self.reliability_head(osm_feat))        # (B, 1)
        osm_rel = torch.clamp(rel_raw + reliability_bias, 0.0, 1.0) * has_osm

        # --- Niveau 1 : guidage global ---
        global_bias = self.global_head(
            torch.cat([temp_feat, osm_feat], dim=-1)
        ).squeeze(-1)   # (B,)

        # --- Niveau 2 : guidage patch ---
        # Le vecteur OSM sert de "requête géographique" pour identifier
        # quelles positions spatiales sont cohérentes avec le contexte géo.
        osm_query = F.normalize(self.patch_query(osm_feat), dim=-1).unsqueeze(1)  # (B,1,D)
        top_diff_n = F.normalize(top_diff, dim=-1)                                 # (B,N,D)
        cosine_match = (top_diff_n * osm_query).sum(dim=-1)                        # (B,N)
        patch_scale = self.patch_scale(
            torch.cat([temp_feat, osm_feat], dim=-1)
        ).squeeze(-1)   # (B,)
        patch_bias = cosine_match * patch_scale.unsqueeze(1)                       # (B,N)

        # --- Niveau 3 : guidage dense ---
        b = temp_feat.size(0)
        dense_bias_scalar = self.dense_bias_head(
            torch.cat([temp_feat, osm_feat], dim=-1)
        ).squeeze(-1)   # (B,)
        dense_bias = F.interpolate(
            patch_bias.view(b, 1, grid_size, grid_size),
            size=dense_logits_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)    # (B, H, W)
        dense_bias = dense_bias + dense_bias_scalar.view(-1, 1, 1)

        return {
            "osm_feat":        osm_feat,
            "osm_reliability": osm_rel,
            "osm_global_bias": global_bias,
            "osm_patch_bias":  patch_bias,
            "osm_dense_bias":  dense_bias,
        }


# ---------------------------------------------------------------------------
# CLIPSemanticInjector
# CLIP sert uniquement pour :
#   - ranking sémantique des textes OSM (top-K les plus pertinents)
#   - signal complémentaire très faible (ne remplace pas OSM ni le visuel)
# ---------------------------------------------------------------------------

class CLIPSemanticInjector(nn.Module):
    def __init__(self, clip_output_dim: int, patch_dim: int):
        super().__init__()
        # ViT features → espace CLIP pour calcul de similarité
        self.visual_proj = nn.Linear(patch_dim, clip_output_dim)
        # CLIP features → espace modèle pour guidage
        self.clip_to_model = nn.Linear(clip_output_dim, patch_dim)
        # Mêmes 3 têtes que OSMSemanticInjector
        self.global_head = nn.Sequential(
            nn.Linear(patch_dim * 2, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1),
        )
        self.patch_query = nn.Linear(patch_dim, patch_dim)
        self.patch_scale = nn.Sequential(
            nn.Linear(patch_dim * 2, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1),
        )
        self.dense_bias_head = nn.Sequential(
            nn.Linear(patch_dim * 2, patch_dim // 4),
            nn.GELU(),
            nn.Linear(patch_dim // 4, 1),
        )

    def forward(
        self,
        clip_branch: FrozenCLIPTextBranch,
        text_candidates_batch: List[List[str]],
        temp_feat: torch.Tensor,
        top_diff: torch.Tensor,
        has_osm_text: torch.Tensor,
        dense_logits_shape: tuple,
        grid_size: int,
        clip_topk: int = 2,
        clip_min_keep: int = 1,
    ) -> Dict[str, torch.Tensor]:
        b = temp_feat.size(0)
        device = temp_feat.device
        has_osm = has_osm_text.float().view(-1, 1).to(device)

        # Projection visuelle → espace CLIP pour le ranking
        visual_query = F.normalize(self.visual_proj(temp_feat), dim=-1)  # (B, clip_dim)

        clip_feats_list = []
        clip_rels_list = []

        for i in range(b):
            candidates = text_candidates_batch[i]
            if not candidates or not clip_branch.enabled:
                clip_feats_list.append(torch.zeros(clip_branch.output_dim, device=device))
                clip_rels_list.append(torch.zeros(1, device=device))
                continue

            text_embeds = clip_branch.encode_texts(candidates, device)  # (N, clip_dim)
            if text_embeds.numel() == 0:
                clip_feats_list.append(torch.zeros(clip_branch.output_dim, device=device))
                clip_rels_list.append(torch.zeros(1, device=device))
                continue

            # Ranking sémantique : similarité cosinus entre visuel et textes OSM
            sims = (text_embeds * visual_query[i].unsqueeze(0)).sum(dim=-1)
            k = max(clip_min_keep, min(clip_topk, sims.numel()))
            vals, idxs = torch.topk(sims, k=k, dim=0)
            weights = torch.softmax(vals, dim=0)
            # Embedding agrégé (pondéré par similarité)
            agg = (weights.unsqueeze(1) * text_embeds[idxs]).sum(dim=0)
            clip_feats_list.append(agg)
            clip_rels_list.append(torch.sigmoid(vals.mean()).view(1))

        clip_feat_raw = torch.stack(clip_feats_list, dim=0)          # (B, clip_dim)
        clip_rel = torch.stack(clip_rels_list, dim=0) * has_osm       # (B, 1)

        # Projection vers l'espace du modèle
        clip_feat_model = self.clip_to_model(clip_feat_raw)           # (B, D)

        # --- Niveau 1 : global ---
        global_bias = self.global_head(
            torch.cat([temp_feat, clip_feat_model], dim=-1)
        ).squeeze(-1)

        # --- Niveau 2 : patch ---
        clip_query = F.normalize(self.patch_query(clip_feat_model), dim=-1).unsqueeze(1)
        top_diff_n = F.normalize(top_diff, dim=-1)
        cosine_match = (top_diff_n * clip_query).sum(dim=-1)
        patch_scale = self.patch_scale(
            torch.cat([temp_feat, clip_feat_model], dim=-1)
        ).squeeze(-1)
        patch_bias = cosine_match * patch_scale.unsqueeze(1)

        # --- Niveau 3 : dense ---
        dense_bias_scalar = self.dense_bias_head(
            torch.cat([temp_feat, clip_feat_model], dim=-1)
        ).squeeze(-1)
        dense_bias = F.interpolate(
            patch_bias.view(b, 1, grid_size, grid_size),
            size=dense_logits_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        dense_bias = dense_bias + dense_bias_scalar.view(-1, 1, 1)

        return {
            "clip_feat":        clip_feat_model,
            "clip_reliability": clip_rel,
            "clip_global_bias": global_bias,
            "clip_patch_bias":  patch_bias,
            "clip_dense_bias":  dense_bias,
        }


# ---------------------------------------------------------------------------
# Scheduler cosinus
# ---------------------------------------------------------------------------

def _build_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, end_lr_ratio: float = 0.0):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    end_lr_ratio = float(max(0.0, min(1.0, end_lr_ratio)))

    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_lr_ratio + (1.0 - end_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# ViLTransformerSS — modèle principal
# ---------------------------------------------------------------------------

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters({"config": dict(config)})
        self.cfg = dict(config)

        # ---- Dimensions ----
        self.image_size = int(self.cfg.get("image_size", 256))
        self.patch_size = int(self.cfg.get("patch_size", 16))
        self.patch_dim = int(self.cfg.get("hidden_size", 384))
        self.dropout = float(self.cfg.get("drop_rate", 0.10))
        self.num_fie_layers = int(self.cfg.get("num_layers", 4))
        self.encoder_num_heads = int(self.cfg.get("num_heads", 6))

        # ---- Gate global ----
        self.global_gate_floor = float(max(0.0, min(0.99, self.cfg.get("change_global_gate_floor", 0.80))))
        self.apply_global_gate_to_dense = bool(self.cfg.get("change_apply_global_gate_to_dense", False))
        self.apply_global_gate_to_coarse = bool(self.cfg.get("change_apply_global_gate_to_coarse", False))
        self.global_aux_scale = float(self.cfg.get("change_global_aux_scale", 0.05))
        self.dense_coarse_fuse_weight = float(self.cfg.get("change_dense_coarse_fuse_weight", 0.08))
        self.dense_global_bias_scale = float(self.cfg.get("change_dense_global_bias_scale", 0.08))

        # ---- OSM ----
        # No explicit mode in an old config means exactly the historical path.
        temporal_osm_enabled = bool(self.cfg.get("levir_temporal_osm_enabled", False))
        explicit_osm_mode = self.cfg.get("change_osm_mode", None)
        if explicit_osm_mode is None:
            explicit_osm_mode = self.cfg.get("levir_temporal_osm_mode", None)
        if temporal_osm_enabled and explicit_osm_mode is None:
            explicit_osm_mode = "paired"
        if explicit_osm_mode is None:
            if bool(
                self.cfg.get("change_use_paired_osm", False)
            ):
                explicit_osm_mode = "paired"
            else:
                explicit_osm_mode = (
                    "legacy_single"
                    if bool(self.cfg.get("change_use_osm_struct", False))
                    else "none"
                )
        osm_mode_aliases = {
            "off": "none",
            "disabled": "none",
            "single": "legacy_single",
            "legacy": "legacy_single",
            "t2_only": "t2",
            "single_t2": "t2",
            "temporal": "paired",
            "paired_temporal": "paired",
        }
        self.osm_mode = osm_mode_aliases.get(
            str(explicit_osm_mode).strip().lower(),
            str(explicit_osm_mode).strip().lower(),
        )
        if self.osm_mode not in {"none", "legacy_single", "t2", "paired"}:
            raise ValueError(
                "change_osm_mode must be one of none, legacy_single, t2, or paired; "
                f"got {explicit_osm_mode!r}."
            )
        self.use_osm_struct = self.osm_mode != "none"
        self.osm_struct_dim = int(self.cfg.get("change_osm_struct_dim", 16))
        self.osm_struct_hidden = int(self.cfg.get("change_osm_struct_hidden", 128))
        self.osm_temporal_hidden = int(
            self.cfg.get("change_osm_temporal_hidden", self.osm_struct_hidden)
        )
        self.osm_temporal_residual_scale = float(
            self.cfg.get("change_osm_temporal_residual_scale", 0.01)
        )
        self.osm_global_weight = float(self.cfg.get("change_osm_global_weight", 0.10))
        self.osm_coarse_weight = float(self.cfg.get("change_osm_coarse_weight", 0.08))
        self.osm_dense_weight = float(self.cfg.get("change_osm_dense_weight", 0.05))
        self.osm_patch_weight = float(self.cfg.get("change_osm_patch_weight", 0.10))
        self.osm_gate_floor = float(self.cfg.get("change_osm_gate_floor", 0.10))
        self.osm_reliability_bias = float(self.cfg.get("change_osm_reliability_bias", 0.0))
        self.use_osm_t2_early = bool(
            self.cfg.get("change_use_osm_t2_early", False)
        )
        configured_late = self.cfg.get("change_use_osm_t2_late", None)
        self.use_osm_late = (
            self.use_osm_struct if configured_late is None else bool(configured_late)
        )
        if self.use_osm_t2_early and self.osm_mode != "t2":
            raise ValueError(
                "change_use_osm_t2_early=True requires change_osm_mode='t2'; "
                "T1 OSM is never synthesized"
            )
        self.osm_t2_early_hidden = int(
            self.cfg.get("change_osm_t2_early_hidden", 32)
        )
        self.osm_t2_early_residual_scale = float(
            self.cfg.get("change_osm_t2_early_residual_scale", 0.01)
        )
        self.osm_t2_late_residual_scale = (
            nn.Parameter(
                torch.tensor(
                    float(self.cfg.get("change_osm_t2_late_residual_scale", 0.0))
                )
            )
            if self.osm_mode == "t2" and self.use_osm_late
            else None
        )

        # ---- Offline CLIP-filtered OSM ----
        # CLIP itself is never loaded by this runtime model. A separate
        # preprocessing script ranks five coarse OSM semantic concepts against
        # the T2 RGB source image and writes a small cache beside the OSM
        # manifest. The cached gate filters only the tile-local 16-D OSM vector.
        self.use_clip_filtered_osm = bool(
            self.cfg.get("change_use_clip_filtered_osm", False)
        )
        self.osm_clip_confidence_floor = float(
            self.cfg.get("change_osm_clip_confidence_floor", 0.75)
        )
        if not 0.0 <= self.osm_clip_confidence_floor <= 1.0:
            raise ValueError("change_osm_clip_confidence_floor must lie in [0,1]")
        if self.use_clip_filtered_osm:
            if self.osm_mode != "t2":
                raise ValueError(
                    "change_use_clip_filtered_osm=True requires change_osm_mode='t2'"
                )
            if not self.use_osm_late:
                raise ValueError(
                    "CLIP-filtered OSM is a late-context experiment; "
                    "change_use_osm_t2_late must be enabled"
                )

        # ---- Final conservative OSM guidance ----
        # This path is intentionally separate from the historical OSM fusion
        # branches.  The dataset may load T2 OSM while change_osm_mode="none";
        # only this bounded late structured guidance then reaches semantics.
        self.use_safe_osm_guidance = bool(
            self.cfg.get("change_use_safe_osm_guidance", False)
        )
        self.safe_osm_hidden = int(
            self.cfg.get("change_safe_osm_hidden", 32)
        )
        self.safe_osm_max_prob_delta = float(
            self.cfg.get("change_safe_osm_max_prob_delta", 0.015)
        )
        self.safe_osm_uncertainty_power = float(
            self.cfg.get("change_safe_osm_uncertainty_power", 2.0)
        )
        self.safe_osm_loss_weight = float(
            self.cfg.get("change_safe_osm_loss_weight", 0.50)
        )
        self.safe_osm_safety_weight = float(
            self.cfg.get("change_safe_osm_safety_weight", 3.0)
        )
        self.safe_osm_l1_weight = float(
            self.cfg.get("change_safe_osm_l1_weight", 0.10)
        )
        self.safe_osm_init_seed = int(
            self.cfg.get("change_safe_osm_init_seed", 1703)
        )
        self.safe_aux_total_max_prob_delta = float(
            self.cfg.get("change_safe_aux_total_max_prob_delta", 0.025)
        )
        if not 0.0 <= self.safe_aux_total_max_prob_delta <= 0.10:
            raise ValueError(
                "change_safe_aux_total_max_prob_delta must lie in [0,0.10]"
            )

        # ---- Genuine paired surface-reflectance + index channels ----
        spectral_enabled = self.cfg.get("levir_spectral_indices_enabled", None)
        spectral_mode = self.cfg.get(
            "levir_spectral_fusion",
            self.cfg.get("spectral_fusion", self.cfg.get("change_spectral_fusion", None)),
        )
        if spectral_enabled is not None and not bool(spectral_enabled):
            # `levir_spectral_fusion=adapter` may be the configured default even
            # for image-only ablations; the explicit enable flag is authoritative.
            spectral_mode = "none"
        elif spectral_mode is None:
            spectral_mode = "adapter" if bool(spectral_enabled) else "none"
        spectral_aliases = {"off": "none", "disabled": "none", "patch_adapter": "adapter"}
        self.spectral_fusion = spectral_aliases.get(
            str(spectral_mode).strip().lower(), str(spectral_mode).strip().lower()
        )
        if self.spectral_fusion == "input_concat":
            raise NotImplementedError(
                "spectral_fusion=input_concat is intentionally unavailable: the safe default "
                "is adapter, which preserves the pretrained three-channel encoder."
            )
        if self.spectral_fusion not in {"none", "adapter"}:
            raise ValueError(
                "spectral fusion must be none or adapter; "
                f"got {spectral_mode!r}."
            )
        self.use_spectral_adapter = self.spectral_fusion == "adapter"
        self.spectral_channel_count = 5
        self.spectral_adapter_hidden = int(
            self.cfg.get("change_spectral_adapter_hidden", self.cfg.get("spectral_adapter_hidden", 32))
        )
        self.spectral_residual_scale = float(
            self.cfg.get("change_spectral_residual_scale", 0.01)
        )

        # Final conservative spectral path.  It is deliberately separate from
        # the historical early TemporalIndexAdapter: genuine NDVI/NDWI are used
        # only as a bounded late correction after the RGB dense prediction.
        self.use_safe_spectral_late = bool(
            self.cfg.get("change_use_safe_spectral_late", False)
        )
        self.safe_spectral_hidden = int(
            self.cfg.get("change_safe_spectral_hidden", 24)
        )
        self.safe_spectral_max_prob_delta = float(
            self.cfg.get("change_safe_spectral_max_prob_delta", 0.02)
        )
        self.safe_spectral_min_reliability = float(
            self.cfg.get("change_safe_spectral_min_reliability", 0.50)
        )
        self.safe_spectral_uncertainty_power = float(
            self.cfg.get("change_safe_spectral_uncertainty_power", 2.0)
        )
        self.safe_spectral_loss_weight = float(
            self.cfg.get("change_safe_spectral_loss_weight", 1.0)
        )
        self.safe_spectral_safety_weight = float(
            self.cfg.get("change_safe_spectral_safety_weight", 2.0)
        )
        self.safe_spectral_l1_weight = float(
            self.cfg.get("change_safe_spectral_l1_weight", 0.05)
        )
        self.safe_spectral_init_seed = int(
            self.cfg.get("change_safe_spectral_init_seed", 1701)
        )
        if self.use_safe_spectral_late and self.use_spectral_adapter:
            raise ValueError(
                "Final safe spectral mode forbids the old early spectral adapter; "
                "set change_spectral_fusion='none' and levir_spectral_fusion='none'."
            )

        self.use_bidirectional_fie = bool(
            self.cfg.get("use_bidirectional_fie", False)
        )
        self.use_instance_head = bool(self.cfg.get("use_instance_head", False))
        self.instance_detach_shared_features = bool(
            self.cfg.get("instance_detach_shared_features", True)
        )
        self.instance_init_seed = int(self.cfg.get("instance_init_seed", 1702))

        # Controlled optimization scope for new modality experiments.
        scope_aliases = {"default": "legacy", "all": "legacy", "aux": "aux_only"}
        raw_scope = str(self.cfg.get("change_train_scope", "legacy")).strip().lower()
        self.change_train_scope = scope_aliases.get(raw_scope, raw_scope)
        if self.change_train_scope not in {"legacy", "aux_only"}:
            raise ValueError(
                "change_train_scope must be 'legacy' or 'aux_only', got "
                f"{raw_scope!r}"
            )
        self.aux_learning_rate = float(
            self.cfg.get("change_aux_learning_rate", self.cfg.get("learning_rate", 1e-4))
        )
        self.aux_weight_decay = float(
            self.cfg.get("change_aux_weight_decay", self.cfg.get("weight_decay", 0.01))
        )
        self.osm_t2_early_require_exact_geometry = bool(
            self.cfg.get("change_osm_t2_early_require_exact_geometry", True)
        )

        # ---- CLIP ----
        self.use_clip_filter_runtime = bool(self.cfg.get("change_clip_filter_runtime", False))
        self.clip_topk = int(self.cfg.get("clip_topk", 2))
        self.clip_min_keep = int(self.cfg.get("clip_min_keep", 1))
        self.clip_use_main_text = bool(self.cfg.get("clip_use_main_text", True))
        # Poids CLIP volontairement faibles : CLIP est un correcteur complémentaire
        self.clip_weight_global = float(self.cfg.get("clip_weight_global", 0.06))
        self.clip_weight_patch = float(self.cfg.get("clip_weight_patch", 0.06))
        self.clip_weight_dense = float(self.cfg.get("clip_weight_dense", 0.03))

        self.vit_encoder_freeze_steps = int(self.cfg.get("vit_encoder_freeze_steps", 0))
        self.vit_pretrained = bool(self.cfg.get("vit_pretrained", False))
        self.vit_encoder_ckpt_path = str(
            self.cfg.get("vit_encoder_ckpt_path", "") or ""
        ).strip()
        self.vit_encoder_partial_load = bool(
            self.cfg.get("vit_encoder_partial_load", True)
        )
        if self.vit_pretrained and not self.vit_encoder_ckpt_path:
            raise RuntimeError(
                "vit_pretrained=True cannot use timm's network-backed weight retrieval. "
                "Set vit_encoder_ckpt_path to a readable local timm/IEFT checkpoint, "
                "or set vit_pretrained=False for random initialization."
            )

        self.single_image_token_len = (self.image_size // self.patch_size) ** 2
        self.change_grid_size = int(self.single_image_token_len ** 0.5)

        # ----------------------------------------------------------------
        # Encodeur ViT partagé (Siamese)
        # ----------------------------------------------------------------
        vit_name = str(self.cfg.get("vit", "vit_small_patch16_224"))
        if vit_name.lower().startswith(("hf-hub:", "hf_hub:", "http://", "https://")):
            raise ValueError(
                f"Remote timm model identifiers are disabled for offline training: {vit_name!r}. "
                "Use a registered local timm architecture and vit_encoder_ckpt_path."
            )
        self.encoder = timm.create_model(
            vit_name,
            # Offline invariant: timm is never authorized to fetch weights.
            pretrained=False,
            num_classes=0,
            img_size=self.image_size,
            in_chans=3,
        )
        self.vit_encoder_load_report = None
        if self.vit_encoder_ckpt_path:
            self.vit_encoder_load_report = load_local_encoder_checkpoint(
                self.encoder,
                self.vit_encoder_ckpt_path,
                partial_load=self.vit_encoder_partial_load,
                map_location="cpu",
            )
            print(
                "[INFO] Local visual encoder initialized from "
                f"{self.vit_encoder_ckpt_path}: "
                f"coverage={self.vit_encoder_load_report['parameter_coverage']:.2%}."
            )
        encoder_dim = int(getattr(self.encoder, "num_features", self.patch_dim))
        if encoder_dim != self.patch_dim:
            raise ValueError(
                f"hidden_size={self.patch_dim} does not match {self.cfg.get('vit')} "
                f"encoder dimension {encoder_dim}."
            )
        patch_embed = getattr(self.encoder, "patch_embed", None)
        patch_projection = getattr(patch_embed, "proj", None)
        if patch_projection is not None and int(getattr(patch_projection, "in_channels", 3)) != 3:
            raise ValueError("The RGB encoder must retain exactly three input channels")
        encoder_num_patches = int(
            getattr(patch_embed, "num_patches", self.single_image_token_len)
        )
        if encoder_num_patches != self.single_image_token_len:
            raise ValueError(
                f"Configured image/patch geometry implies {self.single_image_token_len} tokens, "
                f"but the encoder produces {encoder_num_patches}."
            )

        # ----------------------------------------------------------------
        # Tokens learnable
        # ----------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_dim) * 0.02)
        self.temp_token = nn.Parameter(torch.randn(1, 1, self.patch_dim) * 0.02)
        self.token_type_embeddings = nn.Embedding(2, self.patch_dim)
        self.task_tokens = nn.Parameter(torch.randn(1, 5, self.patch_dim) * 0.02)

        # ----------------------------------------------------------------
        # Blocs FIE
        # ----------------------------------------------------------------
        self.fies = nn.ModuleList([
            FIEBlock(self.patch_dim, self.encoder_num_heads, dropout=0.20)
            for _ in range(self.num_fie_layers)
        ])
        self.bidirectional_fie_adapters = (
            nn.ModuleList(
                [
                    BidirectionalFIEResidual(
                        self.patch_dim,
                        self.encoder_num_heads,
                        dropout=0.20,
                    )
                    for _ in range(self.num_fie_layers)
                ]
            )
            if self.use_bidirectional_fie
            else None
        )
        self.norm = nn.LayerNorm(self.patch_dim)

        self.global_head = nn.Sequential(
            nn.Linear(self.patch_dim, self.patch_dim // 4),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(self.patch_dim // 4, 1),
        )

        # ----------------------------------------------------------------
        # Décodeur dense multi-échelle
        # ----------------------------------------------------------------
        self.ms_change_num_levels = int(self.cfg.get("ms_change_num_levels", 4))
        self.ms_change_decoder_dim = int(self.cfg.get("ms_change_decoder_dim", 160))
        self.ms_change_dropout = float(self.cfg.get("ms_change_dropout", 0.08))
        default_indices = [2, 5, 8, 11]
        self.level_indices = list(self.cfg.get("ms_change_level_indices", default_indices))
        if len(self.level_indices) != self.ms_change_num_levels:
            self.level_indices = default_indices[:self.ms_change_num_levels]

        # Build the semantic decoder first without the instance branch.  The
        # instance head is attached only after all RGB/base children have been
        # initialized, which avoids changing the base initialization stream.
        self.ms_change_decoder = MultiScalePixelObjectChangeDecoder(
            hidden_size=self.patch_dim,
            grid_size=self.change_grid_size,
            num_levels=self.ms_change_num_levels,
            decoder_dim=self.ms_change_decoder_dim,
            dropout=self.ms_change_dropout,
            use_instance_head=False,
            instance_detach_shared_features=self.instance_detach_shared_features,
            instance_init_seed=self.instance_init_seed,
        )

        # Properly supervise the optional center/offset instance head.
        self.instance_loss_weight = float(
            self.cfg.get("instance_loss_weight", 0.30)
        )
        self.instance_loss_fn = (
            InstanceLoss(
                w_center=float(self.cfg.get("instance_w_center", 1.0)),
                w_offset=float(self.cfg.get("instance_w_offset", 0.05)),
                sigma=float(self.cfg.get("instance_center_sigma", 6.0)),
                use_watershed=bool(
                    self.cfg.get("instance_target_use_watershed", True)
                ),
                min_peak_distance=int(
                    self.cfg.get("instance_target_min_peak_distance", 12)
                ),
                min_peak_height=float(
                    self.cfg.get("instance_target_min_peak_height", 3.0)
                ),
                min_instance_area=int(
                    self.cfg.get("instance_target_min_area", 64)
                ),
            )
            if self.use_instance_head and self.instance_loss_weight > 0.0
            else None
        )

        # ----------------------------------------------------------------
        # OSM struct — version stable v20d compatible checkpoint
        # IMPORTANT : on garde les anciens noms de paramètres du checkpoint :
        #   osm_struct_proj.*, osm_reliability_head.*, osm_global_head.*,
        #   osm_patch_query.*, osm_patch_scale.*, osm_dense_bias_head.*
        # Ne pas remplacer par osm_injector.* si on veut préserver v20d.
        # ----------------------------------------------------------------
        if self.use_osm_struct:
            self.osm_struct_proj = nn.Sequential(
                nn.Linear(self.osm_struct_dim, self.osm_struct_hidden),
                nn.LayerNorm(self.osm_struct_hidden),
                nn.GELU(),
                nn.Linear(self.osm_struct_hidden, self.patch_dim),
            )
            self.osm_reliability_head = nn.Sequential(
                nn.Linear(self.patch_dim, self.patch_dim // 4),
                nn.GELU(),
                nn.Linear(self.patch_dim // 4, 1),
            )
            self.osm_global_head = nn.Sequential(
                nn.Linear(self.patch_dim * 2, self.patch_dim // 4),
                nn.GELU(),
                nn.Linear(self.patch_dim // 4, 1),
            )
            self.osm_patch_query = nn.Linear(self.patch_dim, self.patch_dim)
            self.osm_patch_scale = nn.Sequential(
                nn.Linear(self.patch_dim * 2, self.patch_dim // 4),
                nn.GELU(),
                nn.Linear(self.patch_dim // 4, 1),
            )
            self.osm_dense_bias_head = nn.Sequential(
                nn.Linear(self.patch_dim * 2, self.patch_dim // 4),
                nn.GELU(),
                nn.Linear(self.patch_dim // 4, 1),
            )
        else:
            self.osm_struct_proj = None
            self.osm_reliability_head = None
            self.osm_global_head = None
            self.osm_patch_query = None
            self.osm_patch_scale = None
            self.osm_dense_bias_head = None

        # New parameters live under a separate prefix, while both temporal dates
        # pass through the legacy `osm_struct_proj` shared encoder above.
        if self.osm_mode == "paired":
            self.osm_temporal_fusion = PairedOSMFusion(
                feature_dim=self.patch_dim,
                hidden_dim=self.osm_temporal_hidden,
                residual_scale_init=self.osm_temporal_residual_scale,
            )
        else:
            self.osm_temporal_fusion = None

        if self.use_spectral_adapter:
            self.spectral_adapter = TemporalIndexAdapter(
                output_dim=self.patch_dim,
                grid_size=self.change_grid_size,
                hidden_dim=self.spectral_adapter_hidden,
                input_channels=self.spectral_channel_count,
                residual_scale_init=self.spectral_residual_scale,
            )
        else:
            self.spectral_adapter = None
        # Created after base initialization so final late guidance branches do
        # not alter the RGB/base initialization stream.
        self.safe_spectral_residual = None
        self.safe_osm_guidance = None

        if self.use_osm_t2_early:
            self.osm_t2_early_adapter = OSMT2EarlyAdapter(
                output_dim=self.patch_dim,
                grid_size=self.change_grid_size,
                struct_dim=self.osm_struct_dim,
                hidden_dim=self.osm_t2_early_hidden,
                map_channels=5,
                residual_scale_init=self.osm_t2_early_residual_scale,
            )
        else:
            self.osm_t2_early_adapter = None

        # ----------------------------------------------------------------
        # CLIP désactivé dans le modèle stable.
        # La baseline v20d ne doit pas injecter CLIP dans les logits denses.
        # CLIP pourra être utilisé plus tard hors modèle pour générer un JSON OSM.
        # ----------------------------------------------------------------
        self.use_clip_filter_runtime = False
        self.clip_branch = None
        self.clip_injector = None

        # ----------------------------------------------------------------
        # Init des poids
        # ----------------------------------------------------------------
        # Do not call `self.apply(...)` here: doing so also reinitializes timm's
        # pretrained RGB encoder.  Only model-specific children receive the IEFT
        # initializer; timm keeps either its pretrained or native initialization.
        for child_name, child_module in self.named_children():
            if child_name == "encoder":
                continue
            child_module.apply(objectives.init_weights)
        if self.osm_temporal_fusion is not None:
            self.osm_temporal_fusion.reset_residual_parameters()
        if self.spectral_adapter is not None:
            self.spectral_adapter.reset_residual_parameters()
        if self.osm_t2_early_adapter is not None:
            self.osm_t2_early_adapter.reset_residual_parameters()
        if self.bidirectional_fie_adapters is not None:
            for adapter in self.bidirectional_fie_adapters:
                adapter.reset_residual_parameters()

        # Attach auxiliary branches only after RGB/base initialization.  Both are
        # gradient-isolated from the semantic trunk in the final architecture.
        if self.use_instance_head:
            self.ms_change_decoder.enable_instance_head(
                detach_shared_features=self.instance_detach_shared_features,
                init_seed=self.instance_init_seed,
            )
        if self.use_safe_osm_guidance:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.safe_osm_init_seed)
                self.safe_osm_guidance = SafeOSMLateGuidance(
                    struct_dim=self.osm_struct_dim,
                    hidden_dim=self.safe_osm_hidden,
                    max_probability_delta=self.safe_osm_max_prob_delta,
                    uncertainty_power=self.safe_osm_uncertainty_power,
                )
        if self.use_safe_spectral_late:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.safe_spectral_init_seed)
                self.safe_spectral_residual = SafeSpectralLateResidual(
                    hidden_dim=self.safe_spectral_hidden,
                    coarse_grid_size=self.change_grid_size,
                    max_probability_delta=self.safe_spectral_max_prob_delta,
                    min_reliability=self.safe_spectral_min_reliability,
                    uncertainty_power=self.safe_spectral_uncertainty_power,
                )

        self._encoder_frozen_now = False
        self._set_encoder_trainable(self.vit_encoder_freeze_steps <= 0)
        self._apply_training_scope()

    def load_compatible_checkpoint(
        self,
        checkpoint: Union[str, os.PathLike, Mapping[str, Any]],
        map_location: Union[str, torch.device] = "cpu",
        minimum_parameter_coverage: float = 0.50,
        strict_compatibility: bool = False,
    ) -> Dict[str, Any]:
        """Warm-start this model while explicitly auditing compatibility."""

        return load_compatible_checkpoint(
            self,
            checkpoint,
            map_location=map_location,
            minimum_parameter_coverage=minimum_parameter_coverage,
            strict_compatibility=strict_compatibility,
        )

    # ------------------------------------------------------------------
    # Training scope / freeze safety
    # ------------------------------------------------------------------

    def _set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = bool(trainable)
        self._encoder_frozen_now = not bool(trainable)

    def _auxiliary_trainable_prefixes(self) -> Tuple[str, ...]:
        prefixes: List[str] = []
        if self.spectral_adapter is not None and self.use_spectral_adapter:
            prefixes.append("spectral_adapter.")
        if self.safe_spectral_residual is not None and self.use_safe_spectral_late:
            prefixes.append("safe_spectral_residual.")
        if self.safe_osm_guidance is not None and self.use_safe_osm_guidance:
            prefixes.append("safe_osm_guidance.")
        if self.osm_t2_early_adapter is not None and self.use_osm_t2_early:
            prefixes.append("osm_t2_early_adapter.")
        if self.osm_t2_late_residual_scale is not None and self.use_osm_late:
            prefixes.append("osm_t2_late_residual_scale")
        if self.osm_temporal_fusion is not None and self.osm_mode == "paired":
            prefixes.append("osm_temporal_fusion.")
        if self.bidirectional_fie_adapters is not None and self.use_bidirectional_fie:
            prefixes.append("bidirectional_fie_adapters.")
        if self.use_instance_head:
            prefixes.append("ms_change_decoder.pixel_decoder.instance_head.")
        return tuple(prefixes)

    @staticmethod
    def _name_matches_prefix(name: str, prefix: str) -> bool:
        return name == prefix or name.startswith(prefix)

    def _apply_training_scope(self) -> None:
        """Make the controlled auxiliary-only contract explicit in requires_grad.

        Historical v20d behavior remains unchanged for ``legacy``.  In
        ``aux_only`` every parameter is frozen first, then only enabled *new*
        residual/adaptor parameters are re-enabled.  Historical OSM heads stay
        frozen; the new T2 late scalar is the only trainable late-OSM parameter.
        """

        if self.change_train_scope == "legacy":
            return
        for parameter in self.parameters():
            parameter.requires_grad = False
        prefixes = self._auxiliary_trainable_prefixes()
        if not prefixes:
            raise RuntimeError(
                "change_train_scope='aux_only' has no enabled auxiliary module to train"
            )
        matched: List[str] = []
        for name, parameter in self.named_parameters():
            if any(self._name_matches_prefix(name, prefix) for prefix in prefixes):
                parameter.requires_grad = True
                matched.append(name)
        if not matched:
            raise RuntimeError(
                "Aux-only scope did not match any model parameter; refusing unsafe training"
            )
        self._encoder_frozen_now = True

    def training_scope_report(self) -> Dict[str, Any]:
        trainable_names = [name for name, p in self.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        prefixes = self._auxiliary_trainable_prefixes()
        unexpected: List[str] = []
        if self.change_train_scope == "aux_only":
            unexpected = [
                name
                for name in trainable_names
                if not any(self._name_matches_prefix(name, prefix) for prefix in prefixes)
            ]
        return {
            "scope": self.change_train_scope,
            "total_parameters": int(total_count),
            "trainable_parameters": int(trainable_count),
            "trainable_names": trainable_names,
            "allowed_auxiliary_prefixes": list(prefixes),
            "unexpected_trainable_names": unexpected,
        }

    def assert_training_scope_safe(self) -> Dict[str, Any]:
        report = self.training_scope_report()
        if self.change_train_scope == "aux_only":
            unexpected = report["unexpected_trainable_names"]
            if unexpected:
                raise RuntimeError(
                    "Aux-only scope exposed historical trainable parameters: "
                    + ", ".join(unexpected[:20])
                )
            if report["trainable_parameters"] <= 0:
                raise RuntimeError("Aux-only scope contains zero trainable parameters")
        return report

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.change_train_scope == "aux_only":
            # Never let the legacy delayed-unfreeze schedule escape the
            # controlled auxiliary-only experiment.
            if not self._encoder_frozen_now:
                self._set_encoder_trainable(False)
            return
        if self.vit_encoder_freeze_steps <= 0:
            return
        should_train = int(self.global_step) >= self.vit_encoder_freeze_steps
        if should_train and self._encoder_frozen_now:
            self._set_encoder_trainable(True)
            warnings.warn(f"[ViLTransformerSS] Encodeur dégelé à step={int(self.global_step)}")
        elif not should_train and not self._encoder_frozen_now:
            self._set_encoder_trainable(False)

    # ------------------------------------------------------------------
    # Encodage ViT multi-échelle (partagé T1 / T2)
    # ------------------------------------------------------------------

    def _encode_one_multiscale(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        vit = self.encoder
        x = vit.patch_embed(x)
        if getattr(vit, "cls_token", None) is not None:
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
        if getattr(vit, "pos_embed", None) is not None:
            x = x + vit.pos_embed[:, :x.shape[1], :]
        x = vit.pos_drop(x)

        level_feats: List[torch.Tensor] = []
        for idx, block in enumerate(vit.blocks):
            x = block(x)
            if idx in self.level_indices:
                level_feats.append(x[:, 1:, :])

        x = vit.norm(x)
        final_feats = x[:, 1:, :]

        if not level_feats:
            level_feats = [final_feats]
        if len(level_feats) < self.ms_change_num_levels:
            level_feats.extend([final_feats] * (self.ms_change_num_levels - len(level_feats)))
        else:
            level_feats = level_feats[:self.ms_change_num_levels]

        return final_feats, level_feats

    def _encode_bitemporal(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        spectral: Optional[Dict[str, torch.Tensor]] = None,
    ):
        f1, lvl1 = self._encode_one_multiscale(t1)
        f2, lvl2 = self._encode_one_multiscale(t2)

        spectral_temporal = None
        if spectral is not None:
            spatial_t1 = spectral["spatial_t1"].flatten(2).transpose(1, 2)
            spatial_t2 = spectral["spatial_t2"].flatten(2).transpose(1, 2)
            temporal_spatial = spectral["temporal_spatial"].flatten(2).transpose(1, 2)
            spectral_temporal = spectral["pooled_temporal"]
            if (
                spatial_t1.shape != f1.shape
                or spatial_t2.shape != f2.shape
                or temporal_spatial.shape != f1.shape
            ):
                raise ValueError(
                    "Spectral patch grid must align exactly with the visual tokens: "
                    f"spectral={tuple(spatial_t1.shape)}, visual={tuple(f1.shape)}."
                )
            # Split temporal guidance across dates so its signed, directional
            # component reaches every multiscale comparison without introducing
            # an additional high-resolution decoder.
            spatial_t1 = spatial_t1 - 0.5 * temporal_spatial
            spatial_t2 = spatial_t2 + 0.5 * temporal_spatial
            f1 = f1 + spatial_t1
            f2 = f2 + spatial_t2
            # Every captured ViT scale receives the same aligned patch residual.
            # The adapter is intentionally small and shared rather than adding a
            # second expensive multiscale backbone.
            lvl1 = [level + spatial_t1 for level in lvl1]
            lvl2 = [level + spatial_t2 for level in lvl2]

        b, n, _ = f1.shape
        type0 = self.token_type_embeddings(torch.zeros(b, n, dtype=torch.long, device=t1.device))
        type1 = self.token_type_embeddings(torch.ones(b, n, dtype=torch.long, device=t1.device))
        f1_t = F.dropout(f1 + type0, p=self.dropout, training=self.training)
        f2_t = F.dropout(f2 + type1, p=self.dropout, training=self.training)

        # Séquence bitemporelle : [CLS | F_t1 | TEMP | F_t2]
        seq = torch.cat(
            [self.cls_token.expand(b, -1, -1), f1_t, self.temp_token.expand(b, -1, -1), f2_t],
            dim=1,
        )
        tasks = self.task_tokens.expand(b, -1, -1)
        for index, fie in enumerate(self.fies):
            seq = fie(seq, tasks)
            if self.bidirectional_fie_adapters is not None:
                seq = seq + self.bidirectional_fie_adapters[index](seq, tasks)
        h = self.norm(seq)
        if spectral_temporal is not None:
            # The pooled signed/absolute temporal representation reaches the
            # global TEMP path without perturbing RGB tokens when unavailable.
            h = torch.cat(
                [
                    h[:, : n + 1, :],
                    h[:, n + 1 : n + 2, :] + spectral_temporal.unsqueeze(1),
                    h[:, n + 2 :, :],
                ],
                dim=1,
            )
        return h, lvl1, lvl2

    def _compute_spectral_guidance(self, batch: Dict) -> Optional[Dict[str, torch.Tensor]]:
        if not self.use_spectral_adapter or self.spectral_adapter is None:
            return None

        required = (
            "spectral_channels_t1",
            "spectral_channels_t2",
            "spectral_valid_t1",
            "spectral_valid_t2",
            "has_spectral_t1",
            "has_spectral_t2",
            "sensor_id_t1",
            "sensor_id_t2",
            "spectral_reliability",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "spectral_fusion=adapter requires explicit cached spectral tensors and "
                f"availability masks; missing keys: {missing}. The input must contain "
                "Green/Red/NIR/NDVI/McFeeters-NDWI; no RGB-derived fallback is allowed."
            )

        def tensor_value(key: str) -> torch.Tensor:
            value = batch[key]
            if (
                isinstance(value, (list, tuple))
                and len(value) == 1
                and torch.is_tensor(value[0])
            ):
                value = value[0]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return value.to(self.device)

        return self.spectral_adapter(
            tensor_value("spectral_channels_t1").float(),
            tensor_value("spectral_channels_t2").float(),
            tensor_value("spectral_valid_t1").float(),
            tensor_value("spectral_valid_t2").float(),
            tensor_value("has_spectral_t1"),
            tensor_value("has_spectral_t2"),
            tensor_value("sensor_id_t1"),
            tensor_value("sensor_id_t2"),
            tensor_value("spectral_reliability").float(),
        )

    def _compute_safe_osm_guidance(
        self,
        batch: Dict,
        rgb_probability: torch.Tensor,
        dense_shape: Tuple[int, int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute bounded structured T2 OSM guidance.

        The retained OSM geometry is bbox-approximate, so this final path uses
        only the tile-local 16-D structure vector.  It never rasterizes bbox
        footprints as exact geometry and never injects OSM into ViT/FIE.
        """
        if not self.use_safe_osm_guidance or self.safe_osm_guidance is None:
            return None

        required = ("osm_struct_t2", "has_osm_t2", "osm_reliability")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "Safe OSM guidance requires T2 structured OSM context; "
                f"missing keys: {missing}"
            )

        def tensor_value(key: str) -> torch.Tensor:
            value = batch[key]
            if (
                isinstance(value, (list, tuple))
                and len(value) == 1
                and torch.is_tensor(value[0])
            ):
                value = value[0]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return value.to(self.device)

        return self.safe_osm_guidance(
            tensor_value("osm_struct_t2").float(),
            tensor_value("has_osm_t2"),
            tensor_value("osm_reliability").float(),
            rgb_probability,
            dense_shape,
        )

    def _compute_safe_spectral_late(
        self,
        batch: Dict,
        rgb_probability: torch.Tensor,
        dense_shape: Tuple[int, int],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute the final bounded NDVI/NDWI correction.

        This path consumes genuine cached indices only.  It never injects
        spectral features into ViT/FIE/decoder features and therefore remains
        independent from the RGB semantic trunk.
        """
        if not self.use_safe_spectral_late or self.safe_spectral_residual is None:
            return None

        required = (
            "spectral_indices_t1",
            "spectral_indices_t2",
            "spectral_valid_t1",
            "spectral_valid_t2",
            "has_spectral_t1",
            "has_spectral_t2",
            "spectral_reliability",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "Safe spectral late correction requires cached NDVI/NDWI, validity "
                f"and reliability tensors; missing keys: {missing}. No RGB-derived "
                "spectral fallback is permitted."
            )

        def tensor_value(key: str) -> torch.Tensor:
            value = batch[key]
            if (
                isinstance(value, (list, tuple))
                and len(value) == 1
                and torch.is_tensor(value[0])
            ):
                value = value[0]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return value.to(self.device)

        return self.safe_spectral_residual(
            tensor_value("spectral_indices_t1").float(),
            tensor_value("spectral_indices_t2").float(),
            tensor_value("spectral_valid_t1").float(),
            tensor_value("spectral_valid_t2").float(),
            tensor_value("has_spectral_t1"),
            tensor_value("has_spectral_t2"),
            tensor_value("spectral_reliability").float(),
            rgb_probability,
            dense_shape,
        )

    def _apply_osm_t2_early_guidance(
        self,
        batch: Dict,
        lvl2: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Guide only T2 multiscale features with a zero-init OSM residual."""

        if not self.use_osm_t2_early or self.osm_t2_early_adapter is None:
            return lvl2, None
        required = ("osm_maps", "osm_t2_struct", "osm_reliability", "has_osm_t2")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "Early T2 OSM requires explicit cached spatial/structured/reliability "
                f"inputs; missing keys: {missing}. T1/legacy fallback is forbidden."
            )

        def tensor_value(key: str) -> torch.Tensor:
            value = batch[key]
            if isinstance(value, (list, tuple)) and len(value) == 1 and torch.is_tensor(value[0]):
                value = value[0]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return value.to(self.device)

        spatial_reliability = (
            tensor_value("osm_spatial_reliability").float()
            if "osm_spatial_reliability" in batch
            else tensor_value("osm_reliability").float()
        )
        if self.osm_t2_early_require_exact_geometry:
            if "osm_geometry_is_exact" in batch:
                exact = tensor_value("osm_geometry_is_exact").float().reshape(-1)
                spatial_reliability = spatial_reliability.reshape(-1) * exact
            else:
                # Old/synthetic batches do not prove geometry provenance.  The
                # scientifically safe behavior is to bypass only the early
                # spatial path, while late/structured OSM remains available.
                spatial_reliability = torch.zeros_like(spatial_reliability.reshape(-1))
        residual = self.osm_t2_early_adapter(
            tensor_value("osm_maps").float(),
            tensor_value("osm_t2_struct").float(),
            spatial_reliability,
            tensor_value("has_osm_t2"),
        )
        guided: List[torch.Tensor] = []
        for level in lvl2:
            if residual.shape != level.shape:
                raise ValueError(
                    "Early OSM token grid must align with every T2 multiscale level: "
                    f"residual={tuple(residual.shape)}, level={tuple(level.shape)}"
                )
            guided.append(level + residual)
        return guided, residual

    # ------------------------------------------------------------------
    # Collecte des candidats texte pour CLIP
    # ------------------------------------------------------------------

    def _gather_text_candidates(self, batch: Dict, index: int) -> List[str]:
        candidates = []
        if self.clip_use_main_text:
            main_text = batch.get("main_text", batch.get("text", []))
            if isinstance(main_text, list) and index < len(main_text):
                txt = str(main_text[index]).strip()
                if txt:
                    candidates.append(txt)
        osm_texts = batch.get("osm_texts", [])
        if isinstance(osm_texts, list) and index < len(osm_texts):
            seq = osm_texts[index]
            if isinstance(seq, list):
                candidates.extend([str(x).strip() for x in seq if str(x).strip()])
        uniq = []
        for t in candidates:
            if t and t not in uniq:
                uniq.append(t)
        return uniq

    # ------------------------------------------------------------------
    # Guidage OSM (3 niveaux)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_clip_category_gate_to_osm_struct(
        osm_struct: torch.Tensor,
        category_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply five offline CLIP category gates to the 16-D OSM summary.

        Gate order:
        building, transport, water, vegetation, land_use.

        The operation is deterministic and only attenuates semantic values that
        already exist in the tile-local historical OSM vector. It never creates
        or treats bbox-approximate geometry as exact spatial evidence.
        """
        if osm_struct.ndim != 2 or osm_struct.shape[1] != 16:
            raise ValueError(
                "CLIP-filtered OSM expects osm_struct [B,16], got "
                f"{tuple(osm_struct.shape)}"
            )
        if category_gate.ndim != 2 or category_gate.shape != (
            osm_struct.shape[0],
            5,
        ):
            raise ValueError(
                "osm_clip_category_gate must be [B,5], got "
                f"{tuple(category_gate.shape)}"
            )

        gate = torch.nan_to_num(
            category_gate.to(device=osm_struct.device, dtype=osm_struct.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        building = gate[:, 0]
        transport = gate[:, 1]
        water = gate[:, 2]
        vegetation = gate[:, 3]
        land_use = gate[:, 4]

        per_dim = torch.stack(
            [
                building,                              # 0 building presence
                transport,                             # 1 road presence
                transport,                             # 2 railway presence
                water,                                 # 3 water presence
                vegetation,                            # 4 vegetation presence
                land_use,                              # 5 residential
                land_use,                              # 6 industrial/commercial
                land_use,                              # 7 construction
                land_use,                              # 8 amenity / urban land use
                building,                              # 9 building count
                transport,                             # 10 transport count
                water,                                 # 11 water count
                vegetation,                            # 12 vegetation count
                torch.maximum(building, land_use),     # 13 urban / built-up
                transport,                             # 14 transport network
                torch.maximum(
                    torch.maximum(building, transport),
                    land_use,
                ),                                     # 15 overall urban context
            ],
            dim=1,
        )
        return osm_struct * per_dim

    def _compute_osm_guidance(
        self,
        batch: Dict,
        temp_feat: torch.Tensor,
        lvl1: List[torch.Tensor],
        lvl2: List[torch.Tensor],
        dense_shape: tuple,
    ) -> Dict[str, torch.Tensor]:
        b = temp_feat.size(0)
        device = temp_feat.device
        N = self.single_image_token_len

        def _zeros():
            return {
                "osm_feat":        temp_feat.new_zeros((b, self.patch_dim)),
                "osm_feat_t1":     temp_feat.new_zeros((b, self.patch_dim)),
                "osm_feat_t2":     temp_feat.new_zeros((b, self.patch_dim)),
                "osm_absolute_delta": temp_feat.new_zeros((b, self.patch_dim)),
                "osm_signed_delta": temp_feat.new_zeros((b, self.patch_dim)),
                "osm_reliability": temp_feat.new_zeros((b, 1)),
                "osm_global_bias": temp_feat.new_zeros((b,)),
                "osm_patch_bias":  temp_feat.new_zeros((b, N)),
                "osm_dense_bias":  temp_feat.new_zeros((b, *dense_shape)),
                "osm_clip_category_gate": temp_feat.new_zeros((b, 5)),
                "osm_clip_confidence": temp_feat.new_zeros((b, 1)),
                "osm_clip_filter_used": temp_feat.new_zeros((b, 1)),
                "osm_clip_has_candidates": temp_feat.new_zeros((b, 1)),
            }

        if not self.use_osm_late or not self.use_osm_struct or self.osm_struct_proj is None:
            return _zeros()

        def batch_tensor(key: str) -> torch.Tensor:
            value = batch[key]
            if (
                isinstance(value, (list, tuple))
                and len(value) == 1
                and torch.is_tensor(value[0])
            ):
                value = value[0]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return value.to(device)

        osm_feat_t1 = temp_feat.new_zeros((b, self.patch_dim))
        osm_feat_t2 = temp_feat.new_zeros((b, self.patch_dim))
        osm_absolute_delta = temp_feat.new_zeros((b, self.patch_dim))
        osm_signed_delta = temp_feat.new_zeros((b, self.patch_dim))
        explicit_osm_reliability = None
        clip_category_gate = temp_feat.new_zeros((b, 5))
        clip_confidence = temp_feat.new_zeros((b, 1))
        clip_filter_used = temp_feat.new_zeros((b, 1))
        clip_has_candidates = temp_feat.new_zeros((b, 1))

        if self.osm_mode == "paired":
            required = ("osm_struct_t1", "osm_struct_t2", "has_osm_t1", "has_osm_t2")
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(
                    "change_osm_mode=paired requires both historical date-specific OSM "
                    f"representations and availability masks; missing keys: {missing}. "
                    "Legacy/current OSM fallback is forbidden in paired mode."
                )
            osm_struct_t1 = batch_tensor("osm_struct_t1").float()
            osm_struct_t2 = batch_tensor("osm_struct_t2").float()
            if osm_struct_t1.ndim == 1:
                osm_struct_t1 = osm_struct_t1.unsqueeze(0)
            if osm_struct_t2.ndim == 1:
                osm_struct_t2 = osm_struct_t2.unsqueeze(0)
            if osm_struct_t1.shape != (b, self.osm_struct_dim) or osm_struct_t2.shape != (b, self.osm_struct_dim):
                raise ValueError(
                    f"Paired OSM structs must be [B,{self.osm_struct_dim}], got "
                    f"{tuple(osm_struct_t1.shape)} and {tuple(osm_struct_t2.shape)}."
                )
            has_t1 = batch_tensor("has_osm_t1")
            has_t2 = batch_tensor("has_osm_t2")
            osm_feat_t1 = self.osm_struct_proj(osm_struct_t1)
            osm_feat_t2 = self.osm_struct_proj(osm_struct_t2)
            if self.osm_temporal_fusion is None:
                raise RuntimeError("Paired OSM mode was configured without its temporal fusion module")
            temporal_osm = self.osm_temporal_fusion(osm_feat_t1, osm_feat_t2, has_t1, has_t2)
            osm_feat = temporal_osm["fused"]
            osm_feat_t1 = temporal_osm["z_t1"]
            osm_feat_t2 = temporal_osm["z_t2"]
            osm_absolute_delta = temporal_osm["absolute_delta"]
            osm_signed_delta = temporal_osm["signed_delta"]
            has_osm = temporal_osm["any_available"]
        elif self.osm_mode == "t2":
            required = ("osm_struct_t2", "has_osm_t2", "osm_reliability")
            if self.use_clip_filtered_osm:
                required = required + (
                    "osm_clip_category_gate",
                    "osm_clip_confidence",
                    "osm_clip_has_candidates",
                    "has_osm_clip_filter",
                )
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(
                    "change_osm_mode=t2 requires the source-backed T2 OSM "
                    f"representation and availability mask; missing keys: {missing}. "
                    "Legacy/current OSM fallback is forbidden in T2 mode."
                )

            osm_struct_t2 = batch_tensor("osm_struct_t2").float()
            if osm_struct_t2.ndim == 1:
                osm_struct_t2 = osm_struct_t2.unsqueeze(0)
            if osm_struct_t2.shape != (b, self.osm_struct_dim):
                raise ValueError(
                    f"T2 OSM structs must be [B,{self.osm_struct_dim}], got "
                    f"{tuple(osm_struct_t2.shape)}."
                )

            has_osm = batch_tensor("has_osm_t2").float()
            if has_osm.ndim == 0:
                has_osm = has_osm.unsqueeze(0)
            has_osm = has_osm.reshape(-1, 1)
            if has_osm.shape[0] != b:
                raise ValueError(
                    f"has_osm_t2 must contain one value per sample, got {tuple(has_osm.shape)}."
                )

            if self.use_clip_filtered_osm:
                has_clip_filter = batch_tensor("has_osm_clip_filter").float().reshape(-1, 1)
                if has_clip_filter.shape != (b, 1) or torch.any(has_clip_filter < 0.5):
                    raise RuntimeError(
                        "CLIP-filtered OSM is enabled but one or more source scenes "
                        "are missing from data_osm_t2_v2/clip_filter.json"
                    )
                clip_category_gate = batch_tensor("osm_clip_category_gate").float()
                if clip_category_gate.ndim == 1:
                    clip_category_gate = clip_category_gate.unsqueeze(0)
                if clip_category_gate.shape != (b, 5):
                    raise ValueError(
                        "osm_clip_category_gate must be [B,5], got "
                        f"{tuple(clip_category_gate.shape)}"
                    )
                clip_confidence = batch_tensor("osm_clip_confidence").float().reshape(-1, 1)
                clip_has_candidates = batch_tensor("osm_clip_has_candidates").float().reshape(-1, 1)
                if clip_confidence.shape != (b, 1) or clip_has_candidates.shape != (b, 1):
                    raise ValueError(
                        "osm_clip_confidence and osm_clip_has_candidates must "
                        "contain one scalar per sample"
                    )
                clip_confidence = torch.nan_to_num(
                    clip_confidence, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp(0.0, 1.0)
                clip_has_candidates = clip_has_candidates.clamp(0.0, 1.0)
                osm_struct_t2 = self._apply_clip_category_gate_to_osm_struct(
                    osm_struct_t2,
                    clip_category_gate,
                )
                clip_filter_used = has_clip_filter

            osm_feat_t2 = self.osm_struct_proj(osm_struct_t2)
            osm_feat = osm_feat_t2
            explicit_osm_reliability = batch_tensor("osm_reliability").float()
            if explicit_osm_reliability.ndim == 0:
                explicit_osm_reliability = explicit_osm_reliability.unsqueeze(0)
            explicit_osm_reliability = explicit_osm_reliability.reshape(-1, 1)
            if explicit_osm_reliability.shape != (b, 1):
                raise ValueError(
                    "osm_reliability must contain one scalar per T2 sample, got "
                    f"{tuple(explicit_osm_reliability.shape)}"
                )
            if self.use_clip_filtered_osm:
                confidence_gate = (
                    self.osm_clip_confidence_floor
                    + (1.0 - self.osm_clip_confidence_floor) * clip_confidence
                )
                explicit_osm_reliability = (
                    explicit_osm_reliability
                    * clip_has_candidates
                    * confidence_gate
                )
        else:
            # This is deliberately the unchanged historical single-OSM contract.
            if "osm_struct" not in batch:
                return _zeros()
            osm_struct = batch_tensor("osm_struct").float()
            if osm_struct.dim() == 1:
                osm_struct = osm_struct.unsqueeze(0)
            if osm_struct.shape != (b, self.osm_struct_dim):
                raise ValueError(
                    f"osm_struct must be [B,{self.osm_struct_dim}], got {tuple(osm_struct.shape)}."
                )
            has_osm_text = batch.get("has_osm_text", torch.ones(b, device=device))
            if not torch.is_tensor(has_osm_text):
                has_osm_text = torch.tensor(has_osm_text, device=device)
            if has_osm_text.dim() == 0:
                has_osm_text = has_osm_text.unsqueeze(0)
            has_osm = has_osm_text.float().view(-1, 1).to(device)
            osm_feat = self.osm_struct_proj(osm_struct)

        if explicit_osm_reliability is not None:
            osm_rel = torch.nan_to_num(
                explicit_osm_reliability,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0) * has_osm
        else:
            rel_raw = torch.sigmoid(self.osm_reliability_head(osm_feat))
            osm_rel = torch.clamp(rel_raw + self.osm_reliability_bias, 0.0, 1.0) * has_osm
        if self.osm_mode == "paired" and self.osm_temporal_fusion is not None:
            # Legacy heads also see `temp_feat`; scaling only `osm_feat` would not
            # make their initial paired-OSM residual small.  Gate the complete
            # paired path, while leaving the historical single-OSM path untouched.
            paired_gate = self.osm_temporal_fusion.residual_scale.abs().clamp(0.0, 1.0)
            osm_rel = osm_rel * paired_gate

        global_bias = self.osm_global_head(torch.cat([temp_feat, osm_feat], dim=-1)).squeeze(-1)

        top_diff = torch.abs(lvl2[-1] - lvl1[-1])
        osm_query = F.normalize(self.osm_patch_query(osm_feat), dim=-1).unsqueeze(1)
        top_diff_n = F.normalize(top_diff, dim=-1)
        cosine_match = (top_diff_n * osm_query).sum(dim=-1)
        patch_scale = self.osm_patch_scale(torch.cat([temp_feat, osm_feat], dim=-1)).squeeze(-1)
        patch_bias = cosine_match * patch_scale.unsqueeze(1)

        dense_bias_scalar = self.osm_dense_bias_head(torch.cat([temp_feat, osm_feat], dim=-1)).squeeze(-1)
        dense_bias = F.interpolate(
            patch_bias.view(b, 1, self.change_grid_size, self.change_grid_size),
            size=dense_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        dense_bias = dense_bias + dense_bias_scalar.view(-1, 1, 1)

        return {
            "osm_feat":        osm_feat,
            "osm_feat_t1":     osm_feat_t1,
            "osm_feat_t2":     osm_feat_t2,
            "osm_absolute_delta": osm_absolute_delta,
            "osm_signed_delta": osm_signed_delta,
            "osm_reliability": osm_rel,
            "osm_global_bias": global_bias,
            "osm_patch_bias":  patch_bias,
            "osm_dense_bias":  dense_bias,
            "osm_clip_category_gate": clip_category_gate,
            "osm_clip_confidence": clip_confidence,
            "osm_clip_filter_used": clip_filter_used,
            "osm_clip_has_candidates": clip_has_candidates,
        }

    # ------------------------------------------------------------------
    # Guidage CLIP (signal complémentaire)
    # ------------------------------------------------------------------

    def _compute_clip_guidance(
        self,
        batch: Dict,
        temp_feat: torch.Tensor,
        lvl1: List[torch.Tensor],
        lvl2: List[torch.Tensor],
        dense_shape: tuple,
    ) -> Dict[str, torch.Tensor]:
        # Version stable v20d : CLIP n'est pas injecté dans le modèle.
        b = temp_feat.size(0)
        N = self.single_image_token_len
        return {
            "clip_feat":        temp_feat.new_zeros((b, self.patch_dim)),
            "clip_reliability": temp_feat.new_zeros((b, 1)),
            "clip_global_bias": temp_feat.new_zeros((b,)),
            "clip_patch_bias":  temp_feat.new_zeros((b, N)),
            "clip_dense_bias":  temp_feat.new_zeros((b, *dense_shape)),
        }

    # ------------------------------------------------------------------
    # Inférence principale
    # ------------------------------------------------------------------

    def infer(self, batch: Dict, mask_text: bool = False, mask_image: bool = False) -> Dict:
        t1 = batch["image_t1"][0].to(self.device).float()
        t2 = batch["image_t2"][0].to(self.device).float()

        # Encodage bitemporel
        spectral = self._compute_spectral_guidance(batch)
        h, lvl1, lvl2 = self._encode_bitemporal(t1, t2, spectral=spectral)
        lvl2, osm_t2_early_residual = self._apply_osm_t2_early_guidance(batch, lvl2)
        n = self.single_image_token_len
        temp_feat = h[:, n + 1, :]   # token TEMP (mémoire temporelle)

        # Score global image-only (base)
        global_logits_img = self.global_head(temp_feat).squeeze(-1)
        global_probs_img = torch.sigmoid(global_logits_img).unsqueeze(1)

        # Décodeur dense multi-échelle
        decoder_outputs = self.ms_change_decoder(lvl1, lvl2, t1, t2)
        if self.use_instance_head:
            (
                coarse_logits_raw,
                dense_logits_img,
                boundary_logits,
                level_logits,
                instance_center_logits,
                instance_offset,
            ) = decoder_outputs
        else:
            coarse_logits_raw, dense_logits_img, boundary_logits, level_logits = decoder_outputs
            instance_center_logits = None
            instance_offset = None

        dense_logits_img = _canonical_dense_logits(dense_logits_img, t1.shape[0])

        dense_shape = tuple(dense_logits_img.shape[-2:])

        # --- Guidage OSM (injection sémantique principale) ---
        osm = self._compute_osm_guidance(batch, temp_feat, lvl1, lvl2, dense_shape)

        # Version stable v20d : pas d'injection CLIP dans les logits.
        clip = self._compute_clip_guidance(batch, temp_feat, lvl1, lvl2, dense_shape)

        # ---- Gate sémantique OSM ----
        # Proportionnel au score global image et à la fiabilité OSM.
        guide_gate = self.osm_gate_floor + (1.0 - self.osm_gate_floor) * global_probs_img
        guide_gate = guide_gate * torch.clamp(osm["osm_reliability"], 0.0, 1.0)   # (B, 1)
        if self.osm_t2_late_residual_scale is not None:
            # T2 is a new source contract, so the retained legacy heads are
            # reached through their own learnable zero-init gate. Historical
            # legacy_single checkpoints do not instantiate or use this gate.
            guide_gate = guide_gate * self.osm_t2_late_residual_scale

        # ---- Niveau 1 : logit global enrichi OSM ----
        global_logits = global_logits_img
        global_logits = global_logits \
            + self.osm_global_weight * guide_gate.squeeze(1) * osm["osm_global_bias"]
        global_probs = torch.sigmoid(global_logits).unsqueeze(1)

        # ---- Niveau 2 : coarse (patch-level) ----
        coarse_logits = coarse_logits_raw + self.global_aux_scale * global_logits.unsqueeze(1)
        coarse_logits = coarse_logits \
            + self.osm_coarse_weight * guide_gate * osm["osm_patch_bias"]

        coarse_map = torch.sigmoid(coarse_logits).view(
            coarse_logits.size(0), self.change_grid_size, self.change_grid_size
        )
        up_coarse = F.interpolate(
            coarse_logits.view(coarse_logits.size(0), 1, self.change_grid_size, self.change_grid_size),
            size=dense_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # ---- Niveau 3 : dense (pixel-level) ----
        dense_logits = (
            dense_logits_img
            + self.dense_coarse_fuse_weight * up_coarse
            + self.dense_global_bias_scale * global_logits.view(-1, 1, 1)
        )
        dense_logits = dense_logits \
            + self.osm_dense_weight * guide_gate.view(-1, 1, 1) * osm["osm_dense_bias"]

        # Gates optionnelles (désactivées par défaut)
        if self.apply_global_gate_to_dense:
            dense_logits = dense_logits * (
                self.global_gate_floor + (1.0 - self.global_gate_floor) * global_probs.view(-1, 1, 1)
            )
        if self.apply_global_gate_to_coarse:
            coarse_logits = coarse_logits * (
                self.global_gate_floor + (1.0 - self.global_gate_floor) * global_probs
            )

        # Preserve the pure RGB semantic result explicitly.  Final OSM and
        # spectral guidance are both trained locally against this protected RGB
        # prediction and cannot backpropagate through it.
        dense_rgb_logits = dense_logits
        dense_rgb_map = torch.sigmoid(dense_rgb_logits)

        safe_osm = self._compute_safe_osm_guidance(
            batch, dense_rgb_map, dense_shape
        )
        safe_spectral = self._compute_safe_spectral_late(
            batch, dense_rgb_map, dense_shape
        )

        safe_zero = dense_rgb_map.new_zeros(dense_rgb_map.shape)
        safe_scalar = dense_rgb_map.new_zeros((dense_rgb_map.shape[0], 1, 1))

        if safe_osm is None:
            safe_osm = {
                "delta_probability": safe_zero,
                "confidence": safe_zero,
                "uncertainty": safe_zero,
                "effective_gate": safe_zero,
                "source_gate": safe_scalar,
                "content_strength": safe_scalar,
                "reliability": safe_scalar,
            }
        if safe_spectral is None:
            safe_spectral = {
                "delta_probability": safe_zero,
                "confidence": safe_zero,
                "uncertainty": safe_zero,
                "effective_gate": safe_zero,
                "reliability_gate": safe_scalar,
                "pair_valid": safe_zero,
            }

        # Both auxiliary modalities are guidance only.  Their combined authority
        # is hard bounded in probability space.
        safe_aux_delta = (
            safe_osm["delta_probability"]
            + safe_spectral["delta_probability"]
        )
        safe_aux_delta = torch.clamp(
            safe_aux_delta,
            -self.safe_aux_total_max_prob_delta,
            self.safe_aux_total_max_prob_delta,
        )
        dense_map = torch.clamp(
            dense_rgb_map + safe_aux_delta,
            1e-4,
            1.0 - 1e-4,
        )
        dense_logits = torch.log(dense_map) - torch.log1p(-dense_map)

        boundary_map = torch.sigmoid(boundary_logits) if boundary_logits is not None else None

        return {
            "change_logits":               coarse_logits,
            "change_probs":                torch.sigmoid(coarse_logits),
            "change_global_logits":        global_logits,
            "change_probs_global":         global_probs,
            # Authoritative final semantic output.  In the final architecture
            # this is RGB plus the bounded safe spectral correction.
            "change_refined_logits_up":    dense_logits,
            "change_refined_map_up":       dense_map,
            # Protected RGB-only semantic output used for base supervision.
            "change_refined_rgb_logits_up": dense_rgb_logits,
            "change_refined_rgb_map_up":    dense_rgb_map,
            "safe_aux_delta_probability":       safe_aux_delta,
            "safe_osm_delta_probability":       safe_osm["delta_probability"],
            "safe_osm_confidence":              safe_osm["confidence"],
            "safe_osm_uncertainty":             safe_osm["uncertainty"],
            "safe_osm_effective_gate":          safe_osm["effective_gate"],
            "safe_osm_source_gate":             safe_osm["source_gate"],
            "safe_osm_content_strength":        safe_osm["content_strength"],
            "safe_osm_reliability":             safe_osm["reliability"],
            "safe_spectral_delta_probability": safe_spectral["delta_probability"],
            "safe_spectral_confidence":        safe_spectral["confidence"],
            "safe_spectral_uncertainty":       safe_spectral["uncertainty"],
            "safe_spectral_effective_gate":    safe_spectral["effective_gate"],
            "safe_spectral_reliability_gate":  safe_spectral["reliability_gate"],
            "safe_spectral_pair_valid":        safe_spectral["pair_valid"],
            "change_boundary_logits_up":   boundary_logits,
            "change_boundary_map_up":      boundary_map,
            "change_level_logits":         level_logits,
            "change_coarse_map":           coarse_map,
            "change_instance_center_logits": instance_center_logits,
            "change_instance_center_map":  (
                torch.sigmoid(instance_center_logits)
                if instance_center_logits is not None
                else None
            ),
            "change_instance_offset":      instance_offset,
            "osm_reliability":             osm["osm_reliability"],
            "osm_patch_bias":              osm["osm_patch_bias"],
            "osm_dense_bias":              osm["osm_dense_bias"],
            "osm_global_bias":             osm["osm_global_bias"],
            "osm_feat_t1":                 osm["osm_feat_t1"],
            "osm_feat_t2":                 osm["osm_feat_t2"],
            "osm_absolute_delta":          osm["osm_absolute_delta"],
            "osm_signed_delta":            osm["osm_signed_delta"],
            "osm_clip_category_gate":      osm["osm_clip_category_gate"],
            "osm_clip_confidence":         osm["osm_clip_confidence"],
            "osm_clip_filter_used":        osm["osm_clip_filter_used"],
            "osm_clip_has_candidates":     osm["osm_clip_has_candidates"],
            "osm_t2_early_residual":       (
                osm_t2_early_residual
                if osm_t2_early_residual is not None
                else temp_feat.new_zeros(
                    (temp_feat.shape[0], self.single_image_token_len, self.patch_dim)
                )
            ),
            "osm_t2_late_residual_scale": (
                self.osm_t2_late_residual_scale
                if self.osm_t2_late_residual_scale is not None
                else temp_feat.new_tensor(1.0)
            ),
            "spectral_pooled_temporal":    (
                spectral["pooled_temporal"]
                if spectral is not None
                else temp_feat.new_zeros((temp_feat.shape[0], self.patch_dim))
            ),
            "spectral_temporal_spatial":   (
                spectral["temporal_spatial"]
                if spectral is not None
                else temp_feat.new_zeros(
                    (temp_feat.shape[0], self.patch_dim, self.change_grid_size, self.change_grid_size)
                )
            ),
            "spectral_pair_available":     (
                spectral["pair_available"]
                if spectral is not None
                else temp_feat.new_zeros((temp_feat.shape[0], 1))
            ),
            "spectral_z_t1":               (
                spectral["z_t1"]
                if spectral is not None
                else temp_feat.new_zeros(
                    (
                        temp_feat.shape[0],
                        self.spectral_adapter_hidden,
                        self.change_grid_size,
                        self.change_grid_size,
                    )
                )
            ),
            "spectral_z_t2":               (
                spectral["z_t2"]
                if spectral is not None
                else temp_feat.new_zeros(
                    (
                        temp_feat.shape[0],
                        self.spectral_adapter_hidden,
                        self.change_grid_size,
                        self.change_grid_size,
                    )
                )
            ),
            "spectral_absolute_delta":      (
                spectral["absolute_delta"]
                if spectral is not None
                else temp_feat.new_zeros(
                    (
                        temp_feat.shape[0],
                        self.spectral_adapter_hidden,
                        self.change_grid_size,
                        self.change_grid_size,
                    )
                )
            ),
            "spectral_signed_delta":        (
                spectral["signed_delta"]
                if spectral is not None
                else temp_feat.new_zeros(
                    (
                        temp_feat.shape[0],
                        self.spectral_adapter_hidden,
                        self.change_grid_size,
                        self.change_grid_size,
                    )
                )
            ),
            "clip_reliability":            clip["clip_reliability"],
            "clip_patch_bias":             clip["clip_patch_bias"],
            "clip_dense_bias":             clip["clip_dense_bias"],
            "clip_global_bias":            clip["clip_global_bias"],
        }

    def forward(self, batch: Dict):
        return self.infer(batch)

    # ------------------------------------------------------------------
    # Eval partagée
    # ------------------------------------------------------------------

    def _shared_eval(self, batch: Dict, stage: str):
        infer_output = self.infer(batch)

        # Semantic supervision is intentionally applied to the protected RGB
        # output, not to the spectral-corrected output.  Therefore the safe
        # spectral auxiliary loss cannot change the RGB semantic trajectory.
        semantic_infer = infer_output
        if self.use_safe_spectral_late or self.use_safe_osm_guidance:
            semantic_infer = dict(infer_output)
            semantic_infer["change_refined_logits_up"] = infer_output[
                "change_refined_rgb_logits_up"
            ]
            semantic_infer["change_refined_map_up"] = infer_output[
                "change_refined_rgb_map_up"
            ]

        ret = objectives.compute_change_supervised(
            self,
            batch,
            infer=semantic_infer,
        )
        loss = ret.get("loss", None)
        if loss is None:
            loss = torch.tensor(0.0, device=self.device)

        if self.use_safe_osm_guidance and self.safe_osm_guidance is not None:
            if "gt_mask" not in batch:
                raise KeyError("Safe OSM supervision requires batch['gt_mask']")
            safe_osm_losses = self.safe_osm_guidance.loss(
                infer_output["change_refined_rgb_map_up"].detach(),
                infer_output["safe_osm_delta_probability"],
                batch["gt_mask"],
                safety_weight=self.safe_osm_safety_weight,
                l1_weight=self.safe_osm_l1_weight,
            )
            weighted_safe_osm = (
                self.safe_osm_loss_weight
                * safe_osm_losses["loss_safe_osm"]
            )
            loss = loss + weighted_safe_osm
            ret.update(safe_osm_losses)
            ret["loss_safe_osm_weighted"] = weighted_safe_osm
            ret["loss"] = loss

        if self.use_safe_spectral_late and self.safe_spectral_residual is not None:
            if "gt_mask" not in batch:
                raise KeyError("Safe spectral supervision requires batch['gt_mask']")
            safe_losses = self.safe_spectral_residual.loss(
                infer_output["change_refined_rgb_map_up"].detach(),
                infer_output["safe_spectral_delta_probability"],
                batch["gt_mask"],
                safety_weight=self.safe_spectral_safety_weight,
                l1_weight=self.safe_spectral_l1_weight,
            )
            weighted_safe_spectral = (
                self.safe_spectral_loss_weight
                * safe_losses["loss_safe_spectral"]
            )
            loss = loss + weighted_safe_spectral
            ret.update(safe_losses)
            ret["loss_safe_spectral_weighted"] = weighted_safe_spectral
            ret["loss"] = loss

        if self.instance_loss_fn is not None:
            center_logits = infer_output.get("change_instance_center_logits")
            offset_pred = infer_output.get("change_instance_offset")
            if center_logits is None or offset_pred is None:
                raise RuntimeError(
                    "Instance supervision requires center logits and offsets "
                    "from the enabled instance head"
                )
            if "gt_mask" not in batch:
                raise KeyError("Instance supervision requires batch['gt_mask']")
            instance_losses = self.instance_loss_fn(
                center_logits,
                offset_pred,
                batch["gt_mask"],
            )
            weighted_instance = (
                self.instance_loss_weight * instance_losses["loss_instance"]
            )
            loss = loss + weighted_instance
            ret["loss_instance"] = instance_losses["loss_instance"]
            ret["loss_instance_weighted"] = weighted_instance
            ret["loss_center"] = instance_losses["loss_center"]
            ret["loss_offset"] = instance_losses["loss_offset"]
            ret["loss"] = loss

        # IMPORTANT FOR THE OLD PYTORCH LIGHTNING VERSION USED HERE:
        # do not call Lightning metric logging at all during training.
        #
        # The 56,960-batch epoch showed a near-linear CUDA allocation growth
        # (~0.26 GiB / 1000 batches). Even detached step metrics can remain
        # referenced by old Lightning result/logger caches until epoch end.
        # Validation/test logging is still allowed.
        if stage != "train":
            log_dict = {}
            for key, value in ret.items():
                if not torch.is_tensor(value) or key == "loss":
                    continue
                value = value.detach()
                if value.numel() == 1:
                    log_dict[f"{stage}/{key}"] = value
                elif key in {
                    "osm_reliability",
                    "clip_reliability",
                    "change_probs_global",
                    "osm_clip_confidence",
                    "osm_clip_filter_used",
                    "osm_clip_has_candidates",
                }:
                    log_dict[f"{stage}/{key}_mean"] = value.float().mean()

            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        return loss

    def training_step(self, batch: Dict, batch_idx: int):
        # Return only the optimization loss. There is no Lightning train
        # metric logging in this method.
        return self._shared_eval(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int):
        loss = self._shared_eval(batch, "val")
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        return loss

    def test_step(self, batch: Dict, batch_idx: int):
        loss = self._shared_eval(batch, "test")
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        return loss

    # ------------------------------------------------------------------
    # Optimiseur
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        encoder_lr = float(self.cfg.get("encoder_learning_rate", 2e-5))
        other_lr = float(self.cfg.get("learning_rate", 3e-4))
        weight_decay = float(self.cfg.get("weight_decay", 0.05))

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "norm.weight", "norm.bias"]

        if self.change_train_scope == "aux_only":
            report = self.assert_training_scope_safe()
            aux_decay: List[torch.nn.Parameter] = []
            aux_no_decay: List[torch.nn.Parameter] = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if any(token in name for token in no_decay):
                    aux_no_decay.append(param)
                else:
                    aux_decay.append(param)
            param_groups = []
            if aux_decay:
                param_groups.append(
                    {
                        "params": aux_decay,
                        "lr": self.aux_learning_rate,
                        "weight_decay": self.aux_weight_decay,
                        "group_name": "aux_decay",
                    }
                )
            if aux_no_decay:
                param_groups.append(
                    {
                        "params": aux_no_decay,
                        "lr": self.aux_learning_rate,
                        "weight_decay": 0.0,
                        "group_name": "aux_no_decay",
                    }
                )
            if not param_groups:
                raise RuntimeError("Aux-only optimizer has no parameters")
            print(
                "[TRAIN_SCOPE] aux_only | "
                f"trainable={report['trainable_parameters']:,}/"
                f"{report['total_parameters']:,} | lr={self.aux_learning_rate:.3e}"
            )
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), foreach=False)
        else:
            # Historical optimizer behavior.
            clip_param_ids: set = set()
            if self.clip_branch is not None:
                clip_param_ids = {id(p) for p in self.clip_branch.parameters()}

            enc_decay, enc_no_decay, other_decay, other_no_decay = [], [], [], []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if id(param) in clip_param_ids:
                    continue
                is_no_decay = any(nd in name for nd in no_decay)
                if name.startswith("encoder."):
                    (enc_no_decay if is_no_decay else enc_decay).append(param)
                else:
                    (other_no_decay if is_no_decay else other_decay).append(param)

            param_groups = [
                {"params": enc_decay, "lr": encoder_lr, "weight_decay": weight_decay},
                {"params": enc_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
                {"params": other_decay, "lr": other_lr, "weight_decay": weight_decay},
                {"params": other_no_decay, "lr": other_lr, "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=other_lr,
                betas=(0.9, 0.999),
                foreach=False,
            )

        total_steps = max(1, int(self.cfg.get("max_steps", 0) or 0))
        warmup_steps = int(self.cfg.get("warmup_steps", 0))
        end_lr = float(self.cfg.get("end_lr", 0.0))
        scheduler = _build_cosine_scheduler(
            optimizer, warmup_steps, total_steps, end_lr_ratio=end_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
