# -*- coding: utf-8 -*-
# Rollback stable v20d-compatible version generated from the user's provided code.
# Purpose: restore checkpoint-compatible OSM parameter names and disable in-model CLIP injection.

import math
import os
import warnings
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from IEFT.modules import objectives
from IEFT.modules.multiscale_change_head import MultiScalePixelObjectChangeDecoder


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
        self.use_osm_struct = bool(self.cfg.get("change_use_osm_struct", False))
        self.osm_struct_dim = int(self.cfg.get("change_osm_struct_dim", 16))
        self.osm_struct_hidden = int(self.cfg.get("change_osm_struct_hidden", 128))
        self.osm_global_weight = float(self.cfg.get("change_osm_global_weight", 0.10))
        self.osm_coarse_weight = float(self.cfg.get("change_osm_coarse_weight", 0.08))
        self.osm_dense_weight = float(self.cfg.get("change_osm_dense_weight", 0.05))
        self.osm_patch_weight = float(self.cfg.get("change_osm_patch_weight", 0.10))
        self.osm_gate_floor = float(self.cfg.get("change_osm_gate_floor", 0.10))
        self.osm_reliability_bias = float(self.cfg.get("change_osm_reliability_bias", 0.0))

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

        self.single_image_token_len = (self.image_size // self.patch_size) ** 2
        self.change_grid_size = int(self.single_image_token_len ** 0.5)

        # ----------------------------------------------------------------
        # Encodeur ViT partagé (Siamese)
        # ----------------------------------------------------------------
        self.encoder = timm.create_model(
            self.cfg.get("vit", "vit_small_patch16_224"),
            pretrained=self.vit_pretrained,
            num_classes=0,
            img_size=self.image_size,
            in_chans=3,
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

        self.ms_change_decoder = MultiScalePixelObjectChangeDecoder(
            hidden_size=self.patch_dim,
            grid_size=self.change_grid_size,
            num_levels=self.ms_change_num_levels,
            decoder_dim=self.ms_change_decoder_dim,
            dropout=self.ms_change_dropout,
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
        self.apply(objectives.init_weights)
        self._encoder_frozen_now = False
        self._set_encoder_trainable(self.vit_encoder_freeze_steps <= 0)

    # ------------------------------------------------------------------
    # Freeze / unfreeze de l'encodeur
    # ------------------------------------------------------------------

    def _set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = bool(trainable)
        self._encoder_frozen_now = not bool(trainable)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
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

    def _encode_bitemporal(self, t1: torch.Tensor, t2: torch.Tensor):
        f1, lvl1 = self._encode_one_multiscale(t1)
        f2, lvl2 = self._encode_one_multiscale(t2)

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
        for fie in self.fies:
            seq = fie(seq, tasks)
        h = self.norm(seq)
        return h, lvl1, lvl2

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
                "osm_reliability": temp_feat.new_zeros((b, 1)),
                "osm_global_bias": temp_feat.new_zeros((b,)),
                "osm_patch_bias":  temp_feat.new_zeros((b, N)),
                "osm_dense_bias":  temp_feat.new_zeros((b, *dense_shape)),
            }

        if not self.use_osm_struct or self.osm_struct_proj is None:
            return _zeros()
        if "osm_struct" not in batch:
            return _zeros()

        osm_struct = batch["osm_struct"]
        if osm_struct.dim() == 1:
            osm_struct = osm_struct.unsqueeze(0)
        osm_struct = osm_struct.to(device).float()

        has_osm_text = batch.get("has_osm_text", torch.ones(b, device=device))
        if not torch.is_tensor(has_osm_text):
            has_osm_text = torch.tensor(has_osm_text, device=device)
        if has_osm_text.dim() == 0:
            has_osm_text = has_osm_text.unsqueeze(0)
        has_osm = has_osm_text.float().view(-1, 1).to(device)

        osm_feat = self.osm_struct_proj(osm_struct)
        rel_raw = torch.sigmoid(self.osm_reliability_head(osm_feat))
        osm_rel = torch.clamp(rel_raw + self.osm_reliability_bias, 0.0, 1.0) * has_osm

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
            "osm_reliability": osm_rel,
            "osm_global_bias": global_bias,
            "osm_patch_bias":  patch_bias,
            "osm_dense_bias":  dense_bias,
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
        h, lvl1, lvl2 = self._encode_bitemporal(t1, t2)
        n = self.single_image_token_len
        temp_feat = h[:, n + 1, :]   # token TEMP (mémoire temporelle)

        # Score global image-only (base)
        global_logits_img = self.global_head(temp_feat).squeeze(-1)
        global_probs_img = torch.sigmoid(global_logits_img).unsqueeze(1)

        # Décodeur dense multi-échelle
        coarse_logits_raw, dense_logits_img, boundary_logits, level_logits = \
            self.ms_change_decoder(lvl1, lvl2, t1, t2)

        dense_shape = tuple(dense_logits_img.shape[-2:])

        # --- Guidage OSM (injection sémantique principale) ---
        osm = self._compute_osm_guidance(batch, temp_feat, lvl1, lvl2, dense_shape)

        # Version stable v20d : pas d'injection CLIP dans les logits.
        clip = self._compute_clip_guidance(batch, temp_feat, lvl1, lvl2, dense_shape)

        # ---- Gate sémantique OSM ----
        # Proportionnel au score global image et à la fiabilité OSM.
        guide_gate = self.osm_gate_floor + (1.0 - self.osm_gate_floor) * global_probs_img
        guide_gate = guide_gate * torch.clamp(osm["osm_reliability"], 0.0, 1.0)   # (B, 1)

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

        dense_map = torch.sigmoid(dense_logits)
        boundary_map = torch.sigmoid(boundary_logits) if boundary_logits is not None else None

        return {
            "change_logits":               coarse_logits,
            "change_probs":                torch.sigmoid(coarse_logits),
            "change_global_logits":        global_logits,
            "change_probs_global":         global_probs,
            "change_refined_logits_up":    dense_logits,
            "change_refined_map_up":       dense_map,
            "change_boundary_logits_up":   boundary_logits,
            "change_boundary_map_up":      boundary_map,
            "change_level_logits":         level_logits,
            "change_coarse_map":           coarse_map,
            "osm_reliability":             osm["osm_reliability"],
            "osm_patch_bias":              osm["osm_patch_bias"],
            "osm_dense_bias":              osm["osm_dense_bias"],
            "osm_global_bias":             osm["osm_global_bias"],
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
        ret = objectives.compute_change_supervised(self, batch, infer=self.infer(batch))
        loss = ret.get("loss", None)
        if loss is None:
            loss = torch.tensor(0.0, device=self.device)

        log_dict = {}
        for k, v in ret.items():
            if not torch.is_tensor(v):
                continue
            v = v.detach()
            if v.numel() == 1:
                log_dict[f"{stage}/{k}"] = v
            elif k in {"osm_reliability", "clip_reliability", "change_probs_global"}:
                log_dict[f"{stage}/{k}_mean"] = v.float().mean()

        self.log_dict(
            log_dict,
            prog_bar=(stage != "train"),
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=False,
        )
        return loss

    def training_step(self, batch: Dict, batch_idx: int):
        loss = self._shared_eval(batch, "train")
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=False)
        return loss

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

        # Paramètres CLIP gelés → exclus de l'optimiseur
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
            {"params": enc_decay,     "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": enc_no_decay,  "lr": encoder_lr, "weight_decay": 0.0},
            {"params": other_decay,   "lr": other_lr,   "weight_decay": weight_decay},
            {"params": other_no_decay,"lr": other_lr,   "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=other_lr, betas=(0.9, 0.999))

        total_steps = max(1, int(self.cfg.get("max_steps", 0) or 0))
        warmup_steps = int(self.cfg.get("warmup_steps", 0))
        end_lr = float(self.cfg.get("end_lr", 0.0))
        scheduler = _build_cosine_scheduler(optimizer, warmup_steps, total_steps, end_lr_ratio=end_lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }