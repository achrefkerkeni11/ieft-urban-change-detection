import os
import glob
import json
import tqdm
import functools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from IEFT.modules.dist_utils import all_gather


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _safe_topk(t: torch.Tensor, k: int, dim: int):
    dim_size = int(t.size(dim))
    k = int(min(int(k), dim_size))
    k = max(k, 1)
    return t.topk(k, dim=dim)


def _flatten_list(x):
    if not isinstance(x, list):
        return x
    out = []
    for item in x:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def cost_matrix_cosine(x, y, eps=1e-5):
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    tr = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return tr


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma

    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1):
    cost = cost_matrix_cosine(txt_emb, img_emb)
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {"mlm_loss": mlm_loss, "mlm_logits": mlm_logits, "mlm_labels": mlm_labels, "mlm_ids": infer["text_ids"]}

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(ret["mlm_logits"], ret["mlm_labels"])
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)
    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack([mpp_logits[:, :, 0:256], mpp_logits[:, :, 256:512], mpp_logits[:, :, 512:768]], dim=2)
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(mpp_logits.view(-1, 256), mpp_labels.view(-1), ignore_index=-100)

    ret = {"mpp_loss": mpp_loss, "mpp_logits": mpp_logits, "mpp_labels": mpp_labels}

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(ret["mpp_logits"], ret["mpp_labels"])
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)
    return ret


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(pl_module.device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack([ti if itm_labels[i] == 1 else fi for i, (ti, fi) in enumerate(zip(bti, bfi))])
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()

        for i, _len in enumerate(txt_mask.sum(dim=1)):
            if _len > 0:
                txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False

        txt_pad, img_pad = ~txt_mask, ~img_mask
        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1)).to(dtype=cost.dtype)
        img_len = (img_pad.size(1) - img_pad.sum(dim=1)).to(dtype=cost.dtype)

        T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1)
        distance = trace(cost.matmul(T.detach()))
        distance = torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    denom = float(dist_pos.size(0) + dist_neg.size(0))
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / max(denom, 1.0)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {"itm_loss": itm_loss, "itm_wpa_loss": 0.1 * ot_loss, "itm_logits": itm_logits, "itm_labels": itm_labels}

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(ret["itm_logits"], ret["itm_labels"])
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)
    return ret


def compute_irtr(pl_module, batch):
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = int(pl_module.hparams.config["draw_false_text"])

    text_ids = torch.stack([batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1)
    text_masks = torch.stack([batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1)
    text_labels = torch.stack([batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1)

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)

    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )

    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {"irtr_loss": irtr_loss}

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])
    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)
    return ret


# ===================== Change detection (pseudo-supervised bootstrap) =====================

def _robust_minmax_per_image(x: torch.Tensor, q_low: float = 0.02, q_high: float = 0.98, eps: float = 1e-6):
    """
    Normalisation robuste image par image avec quantiles.
    Important: torch.quantile exige un tenseur float/double.
    """
    if x is None:
        return x

    x = x.float()

    if x.dim() == 3:
        x = x.unsqueeze(1)

    if x.dim() != 4:
        raise ValueError(f"_robust_minmax_per_image attend [B,C,H,W] ou [B,H,W], reçu shape={tuple(x.shape)}")

    b = x.shape[0]
    flat = x.contiguous().view(b, -1).float()

    lo = torch.quantile(flat, q_low, dim=1, keepdim=True).view(b, 1, 1, 1)
    hi = torch.quantile(flat, q_high, dim=1, keepdim=True).view(b, 1, 1, 1)

    out = (x - lo) / (hi - lo + eps)
    out = out.clamp(0.0, 1.0)
    return out

def _gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=x.dtype, device=x.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=x.dtype, device=x.device)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _compute_patchwise_change_pseudo_labels(
    x8: torch.Tensor,
    grid_size: int,
    s2_scale_div: float = 10000.0,
    pos_quantile: float = 0.80,
    neg_quantile: float = 0.45,
    ndvi_eps: float = 1e-6,
):
    """
    Pseudo-labels plus stables à partir de plusieurs indices spectraux / texturels.

    Retourne:
      pseudo_labels  : [B, G*G] binaire
      pooled_scores  : [B, G*G] score continu
      confident_mask : [B, G*G] masque des patches fiables
    """
    if x8 is None:
        raise ValueError("x8 is required to build pseudo change labels")

    x8 = x8.float() / float(s2_scale_div)

    t1 = x8[:, :4]
    t2 = x8[:, 4:]

    diff_all = torch.abs(t2 - t1).mean(dim=1, keepdim=True)

    rgb_t1 = t1[:, :3]
    rgb_t2 = t2[:, :3]
    diff_rgb = torch.abs(rgb_t2 - rgb_t1).mean(dim=1, keepdim=True)

    red_t1 = t1[:, 2:3]
    nir_t1 = t1[:, 3:4]
    red_t2 = t2[:, 2:3]
    nir_t2 = t2[:, 3:4]

    ndvi_t1 = (nir_t1 - red_t1) / (nir_t1 + red_t1 + ndvi_eps)
    ndvi_t2 = (nir_t2 - red_t2) / (nir_t2 + red_t2 + ndvi_eps)
    diff_ndvi = torch.abs(ndvi_t2 - ndvi_t1)

    diff_nir = torch.abs(nir_t2 - nir_t1)

    edge_t1 = _gradient_magnitude(rgb_t1.mean(dim=1, keepdim=True))
    edge_t2 = _gradient_magnitude(rgb_t2.mean(dim=1, keepdim=True))
    diff_edge = torch.abs(edge_t2 - edge_t1)

    hp_all = diff_all - F.avg_pool2d(diff_all, kernel_size=3, stride=1, padding=1)
    hp_all = torch.abs(hp_all)

    diff_all = _robust_minmax_per_image(diff_all.float())
    diff_rgb = _robust_minmax_per_image(diff_rgb.float())
    diff_ndvi = _robust_minmax_per_image(diff_ndvi.float())
    diff_nir = _robust_minmax_per_image(diff_nir.float())
    diff_edge = _robust_minmax_per_image(diff_edge.float())
    hp_all = _robust_minmax_per_image(hp_all.float())

    score = (
        0.28 * diff_all
        + 0.18 * diff_rgb
        + 0.18 * diff_ndvi
        + 0.14 * diff_nir
        + 0.12 * diff_edge
        + 0.10 * hp_all
    )
    score = _robust_minmax_per_image(score)

    pooled = F.adaptive_avg_pool2d(score, (grid_size, grid_size)).squeeze(1)
    pooled_flat = pooled.flatten(1)

    pos_thr = torch.quantile(pooled_flat, pos_quantile, dim=1, keepdim=True)
    neg_thr = torch.quantile(pooled_flat, neg_quantile, dim=1, keepdim=True)

    pseudo_labels = (pooled_flat >= pos_thr).float()
    confident_mask = ((pooled_flat >= pos_thr) | (pooled_flat <= neg_thr)).float()

    return pseudo_labels, pooled_flat, confident_mask


def _compute_dense_change_pseudo_labels(
    x8: torch.Tensor,
    s2_scale_div: float = 10000.0,
    pos_quantile: float = 0.80,
    neg_quantile: float = 0.45,
    ndvi_eps: float = 1e-6,
):
    x8 = x8.float() / float(s2_scale_div)
    t1 = x8[:, :4]
    t2 = x8[:, 4:]

    diff_all = torch.abs(t2 - t1).mean(dim=1, keepdim=True)
    rgb_t1 = t1[:, :3]
    rgb_t2 = t2[:, :3]
    diff_rgb = torch.abs(rgb_t2 - rgb_t1).mean(dim=1, keepdim=True)

    red_t1 = t1[:, 2:3]
    nir_t1 = t1[:, 3:4]
    red_t2 = t2[:, 2:3]
    nir_t2 = t2[:, 3:4]
    ndvi_t1 = (nir_t1 - red_t1) / (nir_t1 + red_t1 + ndvi_eps)
    ndvi_t2 = (nir_t2 - red_t2) / (nir_t2 + red_t2 + ndvi_eps)
    diff_ndvi = torch.abs(ndvi_t2 - ndvi_t1)
    diff_nir = torch.abs(nir_t2 - nir_t1)

    edge_t1 = _gradient_magnitude(rgb_t1.mean(dim=1, keepdim=True))
    edge_t2 = _gradient_magnitude(rgb_t2.mean(dim=1, keepdim=True))
    diff_edge = torch.abs(edge_t2 - edge_t1)
    hp_all = torch.abs(diff_all - F.avg_pool2d(diff_all, kernel_size=3, stride=1, padding=1))

    diff_all = _robust_minmax_per_image(diff_all.float())
    diff_rgb = _robust_minmax_per_image(diff_rgb.float())
    diff_ndvi = _robust_minmax_per_image(diff_ndvi.float())
    diff_nir = _robust_minmax_per_image(diff_nir.float())
    diff_edge = _robust_minmax_per_image(diff_edge.float())
    hp_all = _robust_minmax_per_image(hp_all.float())

    score = (
        0.28 * diff_all
        + 0.18 * diff_rgb
        + 0.18 * diff_ndvi
        + 0.14 * diff_nir
        + 0.12 * diff_edge
        + 0.10 * hp_all
    )
    score = _robust_minmax_per_image(score)

    b = score.shape[0]
    flat = score.view(b, -1)
    pos_thr = torch.quantile(flat, pos_quantile, dim=1, keepdim=True).view(b, 1, 1, 1)
    neg_thr = torch.quantile(flat, neg_quantile, dim=1, keepdim=True).view(b, 1, 1, 1)

    pseudo_dense = (score >= pos_thr).float()
    confident_dense = ((score >= pos_thr) | (score <= neg_thr)).float()
    return pseudo_dense, score, confident_dense


def _spatial_smoothness_loss_from_map(logits_2d: torch.Tensor) -> torch.Tensor:
    if logits_2d is None:
        return torch.tensor(0.0)
    x = logits_2d.float()
    dh = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
    dw = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
    return dh + dw


def _dense_boundary_target(mask_2d: torch.Tensor) -> torch.Tensor:
    edge = _gradient_magnitude(mask_2d.float())
    edge = _robust_minmax_per_image(edge.float())
    return (edge > 0.10).float()


def _spatial_smoothness_loss_from_logits(change_logits: torch.Tensor) -> torch.Tensor:
    b, n = change_logits.shape
    g = int(n ** 0.5)
    if g * g != n:
        raise ValueError(f"Expected square grid, got N={n}")
    x = change_logits.view(b, g, g)
    dh = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean()
    dw = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
    return dh + dw


def _masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor = None) -> torch.Tensor:
    if logits is None:
        return torch.tensor(0.0, device=targets.device)

    logits = logits.float()
    targets = targets.float()
    mask = mask.float()

    loss_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    denom = mask.sum().clamp_min(1.0)
    return (loss_map * mask).sum() / denom


def _masked_soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).float()
    targets = targets.float()
    mask = mask.float()

    probs = probs * mask
    targets = targets * mask

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    valid = (mask.sum(dim=1) > 0).float()
    dice = (2.0 * inter + eps) / (denom + eps)
    return ((1.0 - dice) * valid).sum() / valid.sum().clamp_min(1.0)


def _masked_focal_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).float()
    targets = targets.float()
    mask = mask.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * torch.pow(1.0 - pt, gamma) * bce

    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def compute_change_pseudo(pl_module, batch):
    """
    Supervision coarse-to-fine :
    - loss coarse au niveau patch
    - loss refined au niveau dense si la refine head est présente
    - loss frontière dense
    - losses globales / OSM structurées
    """
    if "x8" not in batch:
        return {}

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    if infer["change_logits"] is None:
        return {}

    pred_logits = infer["change_logits"]
    pred_probs = torch.sigmoid(pred_logits)

    b, n = pred_logits.shape
    g = int(n ** 0.5)
    if g * g != n:
        raise ValueError(f"Expected square patch grid, got N={n}")

    pseudo_labels, pseudo_scores, confident_mask = _compute_patchwise_change_pseudo_labels(
        batch["x8"].to(pred_logits.device),
        grid_size=g,
        s2_scale_div=pl_module.hparams.config.get("s2_scale_div", 10000.0),
        pos_quantile=float(pl_module.hparams.config.get("change_pos_quantile", 0.80)),
        neg_quantile=float(pl_module.hparams.config.get("change_neg_quantile", 0.45)),
    )

    pseudo_labels = pseudo_labels.to(pred_logits.device)
    pseudo_scores = pseudo_scores.to(pred_logits.device)
    confident_mask = confident_mask.to(pred_logits.device)

    pos_count = (pseudo_labels * confident_mask).sum(dim=1)
    neg_count = ((1.0 - pseudo_labels) * confident_mask).sum(dim=1)
    pos_weight = ((neg_count + 1.0) / (pos_count + 1.0)).mean().detach().clamp_min(1.0)
    pos_weight_t = torch.tensor([float(pos_weight)], device=pred_logits.device)

    coarse_bce = _masked_bce_with_logits(pred_logits, pseudo_labels, confident_mask, pos_weight=pos_weight_t)
    coarse_dice = _masked_soft_dice_loss(pred_logits, pseudo_labels, confident_mask)
    coarse_focal = _masked_focal_with_logits(pred_logits, pseudo_labels, confident_mask, alpha=0.25, gamma=2.0)
    coarse_smooth = _spatial_smoothness_loss_from_logits(pred_logits)

    coarse_patch_loss = (
        float(pl_module.hparams.config.get("change_bce_loss_weight", 0.50)) * coarse_bce
        + float(pl_module.hparams.config.get("change_dice_loss_weight", 0.30)) * coarse_dice
        + float(pl_module.hparams.config.get("change_focal_loss_weight", 0.20)) * coarse_focal
    )

    # ===== Dense refined supervision =====
    refined_loss = torch.tensor(0.0, device=pred_logits.device)
    refined_bce = torch.tensor(0.0, device=pred_logits.device)
    refined_dice = torch.tensor(0.0, device=pred_logits.device)
    refined_focal = torch.tensor(0.0, device=pred_logits.device)
    refined_smooth = torch.tensor(0.0, device=pred_logits.device)
    boundary_loss = torch.tensor(0.0, device=pred_logits.device)
    pseudo_dense = None
    confident_dense = None

    refined_logits_up = infer.get("change_refined_logits_up", None)
    boundary_logits_up = infer.get("change_boundary_logits_up", None)
    if refined_logits_up is not None:
        pseudo_dense, pseudo_dense_scores, confident_dense = _compute_dense_change_pseudo_labels(
            batch["x8"].to(pred_logits.device),
            s2_scale_div=pl_module.hparams.config.get("s2_scale_div", 10000.0),
            pos_quantile=float(pl_module.hparams.config.get("change_pos_quantile", 0.80)),
            neg_quantile=float(pl_module.hparams.config.get("change_neg_quantile", 0.45)),
        )
        pseudo_dense = pseudo_dense.to(pred_logits.device).squeeze(1)
        confident_dense = confident_dense.to(pred_logits.device).squeeze(1)

        dense_pos_count = (pseudo_dense * confident_dense).sum(dim=(1, 2))
        dense_neg_count = ((1.0 - pseudo_dense) * confident_dense).sum(dim=(1, 2))
        dense_pos_weight = ((dense_neg_count + 1.0) / (dense_pos_count + 1.0)).mean().detach().clamp_min(1.0)
        dense_pos_weight_t = torch.tensor([float(dense_pos_weight)], device=pred_logits.device)

        refined_bce = _masked_bce_with_logits(refined_logits_up, pseudo_dense, confident_dense, pos_weight=dense_pos_weight_t)
        refined_dice = _masked_soft_dice_loss(refined_logits_up.view(b, -1), pseudo_dense.view(b, -1), confident_dense.view(b, -1))
        refined_focal = _masked_focal_with_logits(refined_logits_up, pseudo_dense, confident_dense, alpha=0.25, gamma=2.0)
        refined_smooth = _spatial_smoothness_loss_from_map(refined_logits_up)

        refined_loss = (
            float(pl_module.hparams.config.get("change_refined_bce_loss_weight", 0.45)) * refined_bce
            + float(pl_module.hparams.config.get("change_refined_dice_loss_weight", 0.30)) * refined_dice
            + float(pl_module.hparams.config.get("change_refined_focal_loss_weight", 0.25)) * refined_focal
            + float(pl_module.hparams.config.get("change_refined_smoothness_loss_weight", 0.02)) * refined_smooth
        )

        if boundary_logits_up is not None:
            boundary_target = _dense_boundary_target(pseudo_dense.unsqueeze(1)).squeeze(1)
            boundary_mask = confident_dense
            boundary_loss = _masked_bce_with_logits(boundary_logits_up, boundary_target, boundary_mask)

    global_loss = torch.tensor(0.0, device=pred_logits.device)
    if infer.get("change_global_logits") is not None:
        pseudo_global = (pseudo_labels.mean(dim=1) > 0.03).float()
        global_loss = F.binary_cross_entropy_with_logits(infer["change_global_logits"], pseudo_global)

    struct_patch_loss = torch.tensor(0.0, device=pred_logits.device)
    struct_global_loss = torch.tensor(0.0, device=pred_logits.device)
    has_osm = batch.get("has_osm_text", None)
    if has_osm is not None:
        has_osm = has_osm.to(pred_logits.device).float().view(b, 1)

        if infer.get("change_struct_patch_logits") is not None:
            struct_mask = confident_mask * has_osm
            if struct_mask.sum() > 0:
                struct_patch_loss = _masked_bce_with_logits(
                    infer["change_struct_patch_logits"],
                    pseudo_labels,
                    struct_mask,
                    pos_weight=pos_weight_t,
                )

        if infer.get("change_struct_global_logits") is not None:
            pseudo_global = (pseudo_labels.mean(dim=1) > 0.03).float()
            valid = has_osm.squeeze(1) > 0
            if valid.any():
                struct_global_loss = F.binary_cross_entropy_with_logits(
                    infer["change_struct_global_logits"][valid],
                    pseudo_global[valid],
                )

    total_loss = (
        float(pl_module.hparams.config.get("change_coarse_loss_weight", 0.35)) * coarse_patch_loss
        + float(pl_module.hparams.config.get("change_refined_loss_weight", 0.55)) * refined_loss
        + float(pl_module.hparams.config.get("change_boundary_loss_weight", 0.10)) * boundary_loss
        + float(pl_module.hparams.config.get("change_global_loss_weight", 0.15)) * global_loss
        + float(pl_module.hparams.config.get("change_smoothness_loss_weight", 0.03)) * coarse_smooth
        + float(pl_module.hparams.config.get("osm_struct_patch_loss_weight", 0.05)) * struct_patch_loss
        + float(pl_module.hparams.config.get("osm_struct_global_loss_weight", 0.10)) * struct_global_loss
    )
    total_loss = total_loss * float(pl_module.hparams.config.get("change_loss_weight", 1.0))

    preds_bin = (pred_probs > 0.5).float()
    confident_correct = ((preds_bin == pseudo_labels).float() * confident_mask).sum()
    confident_total = confident_mask.sum().clamp_min(1.0)
    confident_acc = confident_correct / confident_total

    pred_positive_rate = preds_bin.mean()
    pseudo_positive_rate = pseudo_labels.mean()
    confident_rate = confident_mask.mean()

    ret = {
        "change_pseudo_loss": total_loss,
        "change_patch_loss": coarse_patch_loss.detach(),
        "change_bce_loss": coarse_bce.detach(),
        "change_dice_loss": coarse_dice.detach(),
        "change_focal_loss": coarse_focal.detach(),
        "change_global_loss": global_loss.detach(),
        "change_struct_patch_loss": struct_patch_loss.detach(),
        "change_struct_global_loss": struct_global_loss.detach(),
        "change_smoothness_loss": coarse_smooth.detach(),
        "change_refined_loss": refined_loss.detach(),
        "change_refined_bce_loss": refined_bce.detach(),
        "change_refined_dice_loss": refined_dice.detach(),
        "change_refined_focal_loss": refined_focal.detach(),
        "change_refined_smoothness_loss": refined_smooth.detach(),
        "change_boundary_loss": boundary_loss.detach(),
        "change_logits": pred_logits,
        "change_probs": pred_probs,
        "change_pseudo_labels": pseudo_labels,
        "change_pseudo_scores": pseudo_scores,
        "change_confident_mask": confident_mask,
    }

    phase = "train" if pl_module.training else "val"
    pl_module.log(f"change/{phase}/pseudo_loss", total_loss)
    pl_module.log(f"change/{phase}/patch_loss", coarse_patch_loss)
    pl_module.log(f"change/{phase}/bce_loss", coarse_bce)
    pl_module.log(f"change/{phase}/dice_loss", coarse_dice)
    pl_module.log(f"change/{phase}/focal_loss", coarse_focal)
    pl_module.log(f"change/{phase}/refined_loss", refined_loss)
    pl_module.log(f"change/{phase}/refined_bce_loss", refined_bce)
    pl_module.log(f"change/{phase}/refined_dice_loss", refined_dice)
    pl_module.log(f"change/{phase}/refined_focal_loss", refined_focal)
    pl_module.log(f"change/{phase}/boundary_loss", boundary_loss)
    pl_module.log(f"change/{phase}/global_loss", global_loss)
    pl_module.log(f"change/{phase}/struct_patch_loss", struct_patch_loss)
    pl_module.log(f"change/{phase}/struct_global_loss", struct_global_loss)
    pl_module.log(f"change/{phase}/smoothness_loss", coarse_smooth)
    pl_module.log(f"change/{phase}/confident_acc", confident_acc)
    pl_module.log(f"change/{phase}/pred_positive_rate", pred_positive_rate)
    pl_module.log(f"change/{phase}/pseudo_positive_rate", pseudo_positive_rate)
    pl_module.log(f"change/{phase}/confident_rate", confident_rate)

    return ret
def tr_top_k(img_idx, relate, idx, k):
    text_idx = idx[:, :k]
    batch, _ = text_idx.shape
    if batch <= 0:
        return 0.0
    top_sum = 0
    for i in range(batch):
        flag = 0
        for j in range(k):
            if flag == 0:
                rep_text = text_idx[i][j]
                all_relation_index = f"{rep_text.item()}"
                all_relation = relate.get(all_relation_index, [])
                if img_idx[0, i] in all_relation:
                    top_sum += 1
                    flag = 1
    return top_sum / batch


@torch.no_grad()
def compute_irtr_recall(pl_module):
    json_path = pl_module.hparams.config.get("json", "")
    if (not json_path) or (not os.path.exists(json_path)):
        warnings.warn(
            f"[compute_irtr_recall] Skip recall: json path not found: {json_path!r}. "
            f"Pass a valid json=... in Sacred config for real evaluation."
        )
        zeros = (torch.tensor(0.0),) * 6
        result = {"skipped": True, "reason": "json_not_found", "json": json_path}
        return zeros, result

    dm0 = pl_module.trainer.datamodule.dms[0]

    text_dset = dm0.make_no_false_val_dset()
    text_dset.tokenizer = dm0.tokenizer
    all_repeat_text = text_dset.get_single_text()

    id2rep_idx = {}
    for rep_i, ids_list in enumerate(all_repeat_text):
        key = tuple(ids_list)
        if key not in id2rep_idx:
            id2rep_idx[key] = rep_i

    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(text_dset.collate, mlm_collator=dm0.mlm_collator),
    )

    image_dset = dm0.make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = dm0.tokenizer

    if _dist_is_initialized():
        dist_sampler = DistributedSampler(image_dset, shuffle=False)
    else:
        dist_sampler = None

    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        shuffle=False if dist_sampler is not None else False,
        pin_memory=True,
        collate_fn=functools.partial(image_dset.collate, mlm_collator=dm0.mlm_collator),
    )

    text_preload = []
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
                "text": _b.get("text", None),
            }
        )

    if len(text_preload) == 0 or len(image_dset) == 0:
        warnings.warn("[compute_irtr_recall] Skip recall: dataset empty or too small.")
        zeros = (torch.tensor(0.0),) * 6
        result = {"skipped": True, "reason": "dataset_empty", "json": json_path}
        return zeros, result

    text_repeat_correspondence = {}
    rep_texi_id = []
    text_num = 0

    for batch_text in tqdm.tqdm(text_preload, desc="building text correspondence"):
        bs = int(batch_text["text_ids"].size(0))
        for i in range(bs):
            ids_list = batch_text["text_ids"][i].tolist()
            image_index_ = batch_text["img_index"][i]
            key = tuple(ids_list)
            rep_index = id2rep_idx.get(key, None)
            if rep_index is None:
                rep_index = len(id2rep_idx)
                id2rep_idx[key] = rep_index
            text_repeat_correspondence[text_num] = (image_index_, rep_index)
            rep_texi_id.append(rep_index)
            text_num += 1

    text_correspondence = {}
    for _, (img_idx, rep_idx) in text_repeat_correspondence.items():
        k = f"{rep_idx}"
        if k not in text_correspondence:
            text_correspondence[k] = []
        text_correspondence[k].append(img_idx)

    tiids = []
    for pre in text_preload:
        tiids += list(pre["img_index"])
    tiids = torch.tensor(tiids)

    image_preload = []
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    if len(image_preload) == 0:
        warnings.warn("[compute_irtr_recall] Skip recall: no images in preload.")
        zeros = (torch.tensor(0.0),) * 6
        result = {"skipped": True, "reason": "no_images", "json": json_path}
        return zeros, result

    rank_scores = []
    rank_iids = []

    for _ie, _im, _iid in tqdm.tqdm(image_preload, desc="rank loop"):
        _, l, c = _ie.shape
        img_batch_score = []
        for txt_batch in text_preload:
            fblen = int(txt_batch["text_ids"].size(0))
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            score = torch.nan_to_num(score, nan=-1e9, posinf=-1e9, neginf=-1e9)
            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.detach().cpu().tolist())
        rank_iids.append(int(_iid))

    if _dist_is_initialized():
        torch.distributed.barrier()
        gather_rank_scores = all_gather(rank_scores)
        gather_rank_iids = all_gather(rank_iids)
        gather_rank_scores = _flatten_list(gather_rank_scores)
        gather_rank_iids = _flatten_list(gather_rank_iids)
    else:
        gather_rank_scores = rank_scores
        gather_rank_iids = rank_iids

    iids = torch.tensor(gather_rank_iids).view(-1)
    scores = torch.tensor(gather_rank_scores).view(len(iids), -1)

    k10 = min(10, scores.size(1))
    k5 = min(5, scores.size(1))
    k1 = 1

    topk10 = _safe_topk(scores, k10, dim=1)
    topk5 = _safe_topk(scores, k5, dim=1)
    topk1 = _safe_topk(scores, k1, dim=1)

    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    rep_texi_id = torch.tensor(rep_texi_id)
    rep_topk10_iids = rep_texi_id[topk10.indices]
    rep_topk5_iids = rep_texi_id[topk5.indices]
    rep_topk1_iids = rep_texi_id[topk1.indices]

    ttr_top10 = tr_top_k(iids.unsqueeze(0), text_correspondence, rep_topk10_iids, topk10.indices.size(1))
    ttr_top5 = tr_top_k(iids.unsqueeze(0), text_correspondence, rep_topk5_iids, topk5.indices.size(1))
    ttr_top1 = tr_top_k(iids.unsqueeze(0), text_correspondence, rep_topk1_iids, topk1.indices.size(1))

    k10_t = min(10, scores.size(0))
    k5_t = min(5, scores.size(0))
    topk10_t = _safe_topk(scores, k10_t, dim=0)
    topk5_t = _safe_topk(scores, k5_t, dim=0)
    topk1_t = _safe_topk(scores, 1, dim=0)

    topk10_iids_t = iids[topk10_t.indices]
    topk5_iids_t = iids[topk5_t.indices]
    topk1_iids_t = iids[topk1_t.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids_t).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids_t).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids_t).float().max(dim=0)[0].mean()

    mean_ttr = float((ttr_top1 + ttr_top5 + ttr_top10) / 3.0)

    result = {
        "ttr_top1": round(ttr_top1 * 100, 2),
        "ttr_top5": round(ttr_top5 * 100, 2),
        "ttr_top10": round(ttr_top10 * 100, 2),
        "mean_ttr": round(mean_ttr * 100, 2),
        "skipped": False,
        "json": json_path,
        "k_used": {"tr": [k1, k5, k10], "ir": [1, k5_t, k10_t]},
    }

    return (ir_r1 * 100, ir_r5 * 100, ir_r10 * 100, tr_r1 * 100, tr_r5 * 100, tr_r10 * 100), result


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank() if _dist_is_initialized() else 0
    qids, preds = [], []
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = [{"question_id": qid, "answer": pred} for qid, pred in zip(qids, preds)]
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    if _dist_is_initialized():
        torch.distributed.barrier()

    if rank == 0:
        jsons = []
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    if _dist_is_initialized():
        torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")