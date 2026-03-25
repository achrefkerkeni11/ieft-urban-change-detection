import time
from IEFT.modules.light_multiscale_change_decoder import LightMultiScaleChangeDecoder
import torch
import torch.nn as nn
import pytorch_lightning as pl

import IEFT.modules.vision_transformer as vit
from .utils import localTransformer
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from IEFT.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        # 0 = text, 1 = image_t1, 2 = image_t2, 3 = TEMP
        self.token_type_embeddings = nn.Embedding(4, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        self.temp_token = nn.Parameter(torch.zeros(1, 1, config["hidden_size"]))
        nn.init.normal_(self.temp_token, std=0.02)

        ffn_expansion_factor = 2.66
        LayerNorm_type = "WithBias"

        self.local_net = nn.ModuleList(
            [
                localTransformer(
                    dim=768,
                    num_heads=8,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=False,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(3)
            ]
        )

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.single_image_token_len = int(self.transformer.patch_embed.num_patches)

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0 or config["loss_names"]["irtr"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        # ===================== Change Detection Heads ===================== #
        # Legacy MLP head (fallback / ablation)
        self.change_head = nn.Sequential(
            nn.Linear(hs * 4, hs),
            nn.LayerNorm(hs),
            nn.GELU(),
            nn.Linear(hs, 1),
        )
        self.change_head.apply(objectives.init_weights)

        # Global change confidence from TEMP
        self.change_global_head = nn.Sequential(
            nn.Linear(hs, hs),
            nn.LayerNorm(hs),
            nn.GELU(),
            nn.Linear(hs, 1),
        )
        self.change_global_head.apply(objectives.init_weights)

        # New light multi-scale decoder
        self.use_multiscale_change_decoder = bool(
            self.hparams.config.get("use_multiscale_change_decoder", False)
        )
        self.multiscale_decoder_dim = int(
            self.hparams.config.get("multiscale_decoder_dim", 256)
        )
        self.multiscale_decoder_dropout = float(
            self.hparams.config.get("multiscale_decoder_dropout", 0.1)
        )

        self.change_decoder = LightMultiScaleChangeDecoder(
            in_dim=hs * 4,
            decoder_dim=self.multiscale_decoder_dim,
            dropout=self.multiscale_decoder_dropout,
        )
        self.change_decoder.apply(objectives.init_weights)

        # ===================== Load checkpoint flexibly ===================== #
        if self.hparams.config["load_path"] != "":
            self._load_checkpoint_flexible(self.hparams.config["load_path"])

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

    def _load_checkpoint_flexible(self, ckpt_path: str):
        """
        Load checkpoint while tolerating architecture evolution:
        - token_type_embeddings can be 2 or 4 rows
        - new heads may be absent in old checkpoints
        - strict=False for non-critical missing keys
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        state_dict = dict(state_dict)

        tt_key = "token_type_embeddings.weight"
        if tt_key in state_dict:
            ckpt_tt = state_dict.pop(tt_key)
            cur_tt = self.token_type_embeddings.weight.data

            if ckpt_tt.ndim == 2 and cur_tt.ndim == 2 and ckpt_tt.shape[1] == cur_tt.shape[1]:
                rows = min(ckpt_tt.shape[0], cur_tt.shape[0])
                with torch.no_grad():
                    cur_tt[:rows].copy_(ckpt_tt[:rows])

                    if ckpt_tt.shape[0] == 2 and cur_tt.shape[0] >= 4:
                        cur_tt[2].copy_(ckpt_tt[1])
                        cur_tt[3].copy_(ckpt_tt[1])

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        if len(missing) > 0:
            print(f"[ViLTransformerSS] Missing keys while loading '{ckpt_path}': {len(missing)}")
        if len(unexpected) > 0:
            print(f"[ViLTransformerSS] Unexpected keys while loading '{ckpt_path}': {len(unexpected)}")

    @staticmethod
    def _ensure_long_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            return mask.long()
        if mask.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            return mask
        return (mask > 0).long()

    def _build_bitemporal_from_two_streams(
        self,
        img_t1: torch.Tensor,
        img_t2: torch.Tensor,
        mask_image: bool = False,
    ):
        """
        Build bitemporal visual tokens:
          [CLS_t1 | F_t1 | TEMP | F_t2]
        """
        (
            image_embeds_t1,
            image_masks_t1,
            patch_index_t1,
            image_labels_t1,
        ) = self.transformer.visual_embed(
            img_t1,
            max_image_len=self.hparams.config["max_image_len"],
            mask_it=mask_image,
        )

        (
            image_embeds_t2,
            image_masks_t2,
            patch_index_t2,
            image_labels_t2,
        ) = self.transformer.visual_embed(
            img_t2,
            max_image_len=self.hparams.config["max_image_len"],
            mask_it=mask_image,
        )

        image_masks_t1 = self._ensure_long_mask(image_masks_t1)
        image_masks_t2 = self._ensure_long_mask(image_masks_t2)

        cls_t1 = image_embeds_t1[:, :1, :]
        patches_t1 = image_embeds_t1[:, 1:, :]
        patches_t2 = image_embeds_t2[:, 1:, :]

        cls_mask_t1 = image_masks_t1[:, :1]
        patch_mask_t1 = image_masks_t1[:, 1:]
        patch_mask_t2 = image_masks_t2[:, 1:]

        B = image_embeds_t1.size(0)
        temp_token = self.temp_token.expand(B, -1, -1).to(image_embeds_t1.device)
        temp_mask = torch.ones((B, 1), dtype=torch.long, device=image_embeds_t1.device)

        image_embeds = torch.cat(
            [cls_t1, patches_t1, temp_token, patches_t2],
            dim=1,
        )

        image_masks = torch.cat(
            [cls_mask_t1, patch_mask_t1, temp_mask, patch_mask_t2],
            dim=1,
        )

        patch_index = patch_index_t2
        image_labels = image_labels_t2

        return image_embeds, image_masks, patch_index, image_labels

    def _build_bitemporal_from_single_stream(
        self,
        image_embeds: torch.Tensor,
        image_masks: torch.Tensor,
    ):
        """
        Fallback legacy mode:
          [CLS | F | TEMP | F]
        """
        image_masks = self._ensure_long_mask(image_masks)

        cls_img = image_embeds[:, :1, :]
        patches = image_embeds[:, 1:, :]

        cls_mask = image_masks[:, :1]
        patch_mask = image_masks[:, 1:]

        B = image_embeds.size(0)
        temp_token = self.temp_token.expand(B, -1, -1).to(image_embeds.device)
        temp_mask = torch.ones((B, 1), dtype=torch.long, device=image_embeds.device)

        fused_image_embeds = torch.cat(
            [cls_img, patches, temp_token, patches],
            dim=1,
        )
        fused_image_masks = torch.cat(
            [cls_mask, patch_mask, temp_mask, patch_mask],
            dim=1,
        )

        return fused_image_embeds, fused_image_masks

    def _build_image_token_type_ids(
        self,
        image_embeds: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build token type ids for image sequence:
          1 = T1 branch
          2 = T2 branch
          3 = TEMP
        """
        B, L, _ = image_embeds.shape
        n = self.single_image_token_len

        expected_len = 1 + n + 1 + n
        if L == expected_len:
            ids_t1 = torch.full((B, 1 + n), 1, dtype=torch.long, device=device)
            ids_temp = torch.full((B, 1), 3, dtype=torch.long, device=device)
            ids_t2 = torch.full((B, n), 2, dtype=torch.long, device=device)
            return torch.cat([ids_t1, ids_temp, ids_t2], dim=1)

        return torch.full((B, L), 1, dtype=torch.long, device=device)

    def _predict_change_with_legacy_head(self, image_t1_feats, temp_feats, image_t2_feats):
        """
        Legacy MLP per-patch head.
        Returns:
            change_logits [B, N]
            change_global_logits [B]
            change_probs_local [B, N]
            change_probs_global [B, 1]
            change_probs_fused [B, N]
            change_map [B, G, G] or None
        """
        B = image_t1_feats.size(0)
        N = image_t1_feats.size(1)
        temp_expand = temp_feats.expand(B, N, -1)

        change_input = torch.cat(
            [
                image_t1_feats,
                image_t2_feats,
                torch.abs(image_t2_feats - image_t1_feats),
                temp_expand,
            ],
            dim=-1,
        )  # [B, N, 4*hs]

        change_logits = self.change_head(change_input).squeeze(-1)  # [B, N]
        change_global_logits = self.change_global_head(temp_feats.squeeze(1)).squeeze(-1)  # [B]

        change_probs_local = torch.sigmoid(change_logits)  # [B, N]
        change_probs_global = torch.sigmoid(change_global_logits).unsqueeze(1)  # [B, 1]
        change_probs_fused = change_probs_local * change_probs_global  # [B, N]

        g = int(N ** 0.5)
        change_map = None
        if g * g == N:
            change_map = change_probs_fused.view(B, g, g)

        return (
            change_logits,
            change_global_logits,
            change_probs_local,
            change_probs_global,
            change_probs_fused,
            change_map,
        )

    def _predict_change_with_multiscale_decoder(self, image_t1_feats, temp_feats, image_t2_feats):
        """
        New multi-scale decoder head.
        Returns exactly the same API as legacy head.
        """
        B = image_t1_feats.size(0)
        N = image_t1_feats.size(1)
        temp_expand = temp_feats.expand(B, N, -1)

        change_input = torch.cat(
            [
                image_t1_feats,
                image_t2_feats,
                torch.abs(image_t2_feats - image_t1_feats),
                temp_expand,
            ],
            dim=-1,
        )  # [B, N, 4*hs]

        local_logits_2d = self.change_decoder(change_input)  # [B, G, G]
        g = local_logits_2d.shape[-1]

        change_logits = local_logits_2d.view(B, -1)  # [B, N]
        change_global_logits = self.change_global_head(temp_feats.squeeze(1)).squeeze(-1)  # [B]

        change_probs_local = torch.sigmoid(change_logits)  # [B, N]
        change_probs_global = torch.sigmoid(change_global_logits).unsqueeze(1)  # [B, 1]
        change_probs_fused = change_probs_local * change_probs_global  # [B, N]

        change_map = change_probs_fused.view(B, g, g)

        return (
            change_logits,
            change_global_logits,
            change_probs_local,
            change_probs_global,
            change_probs_fused,
            change_map,
        )

    def infer(
        self,
        batch,
        mask_text: bool = False,
        mask_image: bool = False,
        image_token_type_idx: int = 1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch["text_masks"]

        if text_ids.dtype != torch.long:
            text_ids = text_ids.long()

        text_masks = self._ensure_long_mask(text_masks)
        text_embeds = self.text_embeddings(text_ids)

        patch_index, image_labels = None, None

        if image_embeds is None and image_masks is None:
            if "image_t1" in batch and "image_t2" in batch:
                img_t1 = batch["image_t1"][0].to(text_embeds.device)
                img_t2 = batch["image_t2"][0].to(text_embeds.device)

                image_embeds, image_masks, patch_index, image_labels = self._build_bitemporal_from_two_streams(
                    img_t1=img_t1,
                    img_t2=img_t2,
                    mask_image=mask_image,
                )
            else:
                img = batch[imgkey][0].to(text_embeds.device)
                (
                    legacy_image_embeds,
                    legacy_image_masks,
                    patch_index,
                    image_labels,
                ) = self.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_image,
                )
                image_embeds, image_masks = self._build_bitemporal_from_single_stream(
                    legacy_image_embeds,
                    legacy_image_masks,
                )
        else:
            image_masks = self._ensure_long_mask(image_masks)

            expected_single = 1 + self.single_image_token_len
            if image_embeds.size(1) == expected_single:
                image_embeds, image_masks = self._build_bitemporal_from_single_stream(
                    image_embeds,
                    image_masks,
                )

        image_masks = self._ensure_long_mask(image_masks)

        device = text_embeds.device

        text_token_type_ids = torch.zeros(
            (text_ids.size(0), text_ids.size(1)),
            dtype=torch.long,
            device=device,
        )

        image_token_type_ids = self._build_image_token_type_ids(
            image_embeds=image_embeds,
            device=image_embeds.device,
        )

        text_embeds = text_embeds + self.token_type_embeddings(text_token_type_ids)
        image_embeds = image_embeds + self.token_type_embeddings(image_token_type_ids)

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)

        if image_masks is None:
            co_masks = text_masks
        else:
            co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        text_len = text_embeds.shape[1]
        n = self.single_image_token_len
        expected_body_len = 2 * n + 1  # F_t1 + TEMP + F_t2

        for i in range(len(self.transformer.blocks)):
            blk = self.transformer.blocks[i]

            text_feats_block = x[:, :text_len]
            image_feats_block = x[:, text_len:]

            if i in (1, 2, 3):
                block_id = i - 1
                blk_se = self.transformer.SE_blocks[block_id]

                img_cls = image_feats_block[:, :1, :]
                image_body = image_feats_block[:, 1:, :]

                if image_body.shape[1] == expected_body_len:
                    image_t1 = image_body[:, :n, :]
                    temp_tok = image_body[:, n:n + 1, :]
                    image_t2 = image_body[:, n + 1:n + 1 + n, :]

                    image_t1_se = image_t1.permute(0, 2, 1)
                    image_t1_se, _attn = blk_se(image_t1_se)
                    image_t1_se = image_t1_se.permute(0, 2, 1)

                    image_t2_se = image_t2.permute(0, 2, 1)
                    image_t2_se, _attn = blk_se(image_t2_se)
                    image_t2_se = image_t2_se.permute(0, 2, 1)

                    image_t1_se = self.local_net[block_id](image_t1_se)
                    image_t2_se = self.local_net[block_id](image_t2_se)

                    image_feats_block = torch.cat(
                        [img_cls, image_t1_se, temp_tok, image_t2_se],
                        dim=1,
                    )
                else:
                    if image_body.shape[1] == n:
                        image_body_se = image_body.permute(0, 2, 1)
                        image_body_se, _attn = blk_se(image_body_se)
                        image_body_se = image_body_se.permute(0, 2, 1)
                        image_body_se = self.local_net[block_id](image_body_se)
                        image_feats_block = torch.cat([img_cls, image_body_se], dim=1)

                x = torch.cat([text_feats_block, image_feats_block], dim=1)

            if i not in (6, 7, 8, 9, 10, 11):
                x, _attn = blk(x, mask=co_masks)

        out_feats = x
        x = self.transformer.norm(x)

        text_feats = x[:, :text_len]
        image_feats = x[:, text_len:]
        cls_feats = self.pooler(x)

        image_cls = image_feats[:, :1, :]
        image_body = image_feats[:, 1:, :]

        if image_body.shape[1] >= expected_body_len:
            image_t1_feats = image_body[:, :n, :]
            temp_feats = image_body[:, n:n + 1, :]
            image_t2_feats = image_body[:, n + 1:n + 1 + n, :]
        else:
            image_t1_feats = None
            temp_feats = None
            image_t2_feats = None

        # ===================== Change head outputs ===================== #
        change_logits = None
        change_global_logits = None
        change_probs_local = None
        change_probs_global = None
        change_probs_fused = None
        change_map = None

        if image_t1_feats is not None and temp_feats is not None and image_t2_feats is not None:
            if self.use_multiscale_change_decoder:
                (
                    change_logits,
                    change_global_logits,
                    change_probs_local,
                    change_probs_global,
                    change_probs_fused,
                    change_map,
                ) = self._predict_change_with_multiscale_decoder(
                    image_t1_feats=image_t1_feats,
                    temp_feats=temp_feats,
                    image_t2_feats=image_t2_feats,
                )
            else:
                (
                    change_logits,
                    change_global_logits,
                    change_probs_local,
                    change_probs_global,
                    change_probs_fused,
                    change_map,
                ) = self._predict_change_with_legacy_head(
                    image_t1_feats=image_t1_feats,
                    temp_feats=temp_feats,
                    image_t2_feats=image_t2_feats,
                )

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "image_cls": image_cls,
            "image_t1_feats": image_t1_feats,
            "temp_feats": temp_feats,
            "image_t2_feats": image_t2_feats,
            "change_logits": change_logits,
            "change_global_logits": change_global_logits,
            "change_probs_local": change_probs_local,
            "change_probs_global": change_probs_global,
            "change_probs_fused": change_probs_fused,
            "change_map": change_map,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "out_feats": out_feats,
        }
        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        if self.hparams.config.get("change_loss_weight", 1.0) > 0 and "x8" in batch:
            ret.update(objectives.compute_change_pseudo(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        _ = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

        ret = dict()
        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))
        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)