import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        if getattr(module, "weight", None) is not None:
            nn.init.constant_(module.weight, 1.0)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, 0.0)


def _soft_dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if probs.dim() == 3:
        probs = probs.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def _binary_edge_target(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=mask.dtype, device=mask.device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=mask.dtype, device=mask.device)
    gx = F.conv2d(mask.float(), sobel_x, padding=1)
    gy = F.conv2d(mask.float(), sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + 1e-6)
    gmax = g.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-6).view(mask.size(0), 1, 1, 1)
    g = g / gmax
    return (g > 0.10).float()


def _compute_color_only_mask(batch, device):
    if "image_t1" not in batch or "image_t2" not in batch:
        return None
    t1 = batch["image_t1"][0].to(device).float()
    t2 = batch["image_t2"][0].to(device).float()
    diff = torch.abs(t2 - t1).mean(dim=1, keepdim=True)
    gray1 = t1.mean(dim=1, keepdim=True)
    gray2 = t2.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=t1.dtype, device=device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=t1.dtype, device=device)
    e1 = torch.sqrt(F.conv2d(gray1, sobel_x, padding=1) ** 2 + F.conv2d(gray1, sobel_y, padding=1) ** 2 + 1e-6)
    e2 = torch.sqrt(F.conv2d(gray2, sobel_x, padding=1) ** 2 + F.conv2d(gray2, sobel_y, padding=1) ** 2 + 1e-6)
    edge_diff = torch.abs(e2 - e1)

    dmax = diff.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-6).view(diff.size(0), 1, 1, 1)
    emax = edge_diff.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-6).view(edge_diff.size(0), 1, 1, 1)
    diff_n = diff / dmax
    edge_n = edge_diff / emax
    return (diff_n > 0.35).float() * (edge_n < 0.15).float()


def _compute_rural_background_weight(batch, device, config):
    if "image_t1" not in batch or "image_t2" not in batch:
        return None

    t1 = batch["image_t1"][0].to(device).float()
    t2 = batch["image_t2"][0].to(device).float()

    diff = torch.abs(t2 - t1).mean(dim=1, keepdim=True)
    gray1 = t1.mean(dim=1, keepdim=True)
    gray2 = t2.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=t1.dtype, device=device)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=t1.dtype, device=device)

    e1 = torch.sqrt(F.conv2d(gray1, sobel_x, padding=1) ** 2 + F.conv2d(gray1, sobel_y, padding=1) ** 2 + 1e-6)
    e2 = torch.sqrt(F.conv2d(gray2, sobel_x, padding=1) ** 2 + F.conv2d(gray2, sobel_y, padding=1) ** 2 + 1e-6)
    edge_diff = torch.abs(e2 - e1)

    dmax = diff.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-6).view(diff.size(0), 1, 1, 1)
    e2max = e2.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-6).view(e2.size(0), 1, 1, 1)
    edmax = edge_diff.flatten(1).amax(dim=1, keepdim=True).clamp_min(1e-6).view(edge_diff.size(0), 1, 1, 1)

    diff_n = diff / dmax
    e2_n = e2 / e2max
    edge_diff_n = edge_diff / edmax

    low_change_thr = float(config.get("change_rural_low_change_thr", 0.20))
    low_structure_thr = float(config.get("change_rural_low_structure_thr", 0.22))

    low_change = ((low_change_thr - diff_n) / max(low_change_thr, 1e-6)).clamp(0.0, 1.0)
    structure_signal = torch.maximum(e2_n, edge_diff_n)
    low_structure = ((low_structure_thr - structure_signal) / max(low_structure_thr, 1e-6)).clamp(0.0, 1.0)

    return low_change * low_structure


def _masked_topk_penalty(probs: torch.Tensor, weight_map: torch.Tensor, k: int) -> torch.Tensor:
    if weight_map is None:
        return probs.new_tensor(0.0)
    flat_probs = probs.flatten(1)
    flat_w = weight_map.flatten(1)
    vals = []
    for i in range(flat_probs.size(0)):
        valid = flat_w[i] > 0
        if not torch.any(valid):
            continue
        cur = flat_probs[i][valid] * flat_w[i][valid]
        kk = max(1, min(int(k), int(cur.numel())))
        vals.append(torch.topk(cur, k=kk, dim=0).values.mean())
    if not vals:
        return probs.new_tensor(0.0)
    return torch.stack(vals).mean()


def _deep_supervision_loss(level_logits, local_targets, pos_weight):
    if not level_logits:
        return local_targets.new_tensor(0.0)
    total = local_targets.new_tensor(0.0)
    for logit in level_logits:
        total = total + F.binary_cross_entropy_with_logits(logit, local_targets, pos_weight=pos_weight)
    return total / max(len(level_logits), 1)


def _topk_mean_map(probs: torch.Tensor, k: int) -> torch.Tensor:
    flat = probs.flatten(1)
    k = max(1, min(int(k), flat.size(1)))
    topk_vals = torch.topk(flat, k=k, dim=1).values
    return topk_vals.mean(dim=1)




def _prepare_gt_for_change_losses(gt_4d: torch.Tensor, config) -> torch.Tensor:
    """
    Protect change losses from non-binary targets introduced by label smoothing.
    When enabled, all supervised change losses use a hard-binarized GT.
    This does NOT modify dataset files; it only sanitizes the target inside the loss path.
    """
    use_hard = bool(config.get("change_force_binary_gt_for_losses", True))
    thr = float(config.get("change_force_binary_gt_threshold", 0.5))
    if use_hard:
        return (gt_4d > thr).float()
    return gt_4d.float()

def _dilate_binary_mask(mask: torch.Tensor, dilate: int) -> torch.Tensor:
    """
    Binary morphological dilation only.
    Important: this function never creates soft target values like 0.3 or 0.7.
    It first binarizes the mask with a hard threshold, then applies max-pooling,
    then binarizes again.
    """
    mask = (mask > 0.5).float()
    if dilate <= 0:
        return mask
    k = 2 * int(dilate) + 1
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=int(dilate))
    return (dilated > 0.5).float()


def _dense_completion_loss(logits_4d: torch.Tensor, gt_4d: torch.Tensor, dilate: int = 1) -> torch.Tensor:
    gt_bin = (gt_4d > 0.5).float()
    pos_patch = (gt_bin.flatten(1).sum(dim=1) > 0).float().view(-1, 1, 1, 1)
    if pos_patch.sum() <= 0:
        return logits_4d.new_tensor(0.0)
    target = _dilate_binary_mask(gt_bin, dilate=dilate)
    probs = torch.sigmoid(logits_4d)
    miss = (1.0 - probs) * target * pos_patch
    denom = (target * pos_patch).sum().clamp_min(1.0)
    return miss.sum() / denom


def compute_change_supervised(pl_module, batch, infer=None):
    if "gt_mask" not in batch:
        return {}

    if infer is None:
        infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    device = pl_module.device
    gt = batch["gt_mask"].to(device).float()
    if gt.dim() == 4 and gt.size(1) == 1:
        gt_4d = gt
    elif gt.dim() == 3:
        gt_4d = gt.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected gt_mask shape: {tuple(gt.shape)}")

    dense_logits = infer["change_refined_logits_up"]
    boundary_logits = infer.get("change_boundary_logits_up", None)
    local_logits = infer["change_logits"]
    global_logits = infer["change_global_logits"]
    level_logits = infer.get("change_level_logits", []) or []

    gt_loss_4d = _prepare_gt_for_change_losses(gt_4d, pl_module.hparams.config)

    dense_logits_4d = dense_logits.unsqueeze(1) if dense_logits.dim() == 3 else dense_logits

    g = int(local_logits.shape[1] ** 0.5)
    pooled = F.adaptive_avg_pool2d(gt_loss_4d, (g, g))
    local_targets = (pooled > 0.5).float().flatten(1)

    dense_pos_w = torch.tensor([float(pl_module.hparams.config.get("change_dense_pos_weight", 5.0))], device=device)
    local_pos_w = torch.tensor([float(pl_module.hparams.config.get("change_local_pos_weight", 5.0))], device=device)
    boundary_pos_w = torch.tensor([float(pl_module.hparams.config.get("change_boundary_pos_weight", 3.0))], device=device)

    dense_bce = F.binary_cross_entropy_with_logits(dense_logits_4d, gt_loss_4d, pos_weight=dense_pos_w)
    dense_dice = _soft_dice_loss_with_logits(dense_logits_4d, gt_loss_4d)
    local_bce = F.binary_cross_entropy_with_logits(local_logits, local_targets, pos_weight=local_pos_w)
    deep_sup = _deep_supervision_loss(level_logits, local_targets, local_pos_w)

    global_targets = (gt_loss_4d.flatten(1).sum(dim=1) > 0).float()
    global_bce = F.binary_cross_entropy_with_logits(global_logits.view(-1), global_targets)

    boundary_loss = torch.tensor(0.0, device=device)
    boundary_dice = torch.tensor(0.0, device=device)
    if boundary_logits is not None:
        boundary_logits_4d = boundary_logits.unsqueeze(1) if boundary_logits.dim() == 3 else boundary_logits
        boundary_targets = _binary_edge_target(gt_loss_4d)
        boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits_4d, boundary_targets, pos_weight=boundary_pos_w)
        boundary_dice = _soft_dice_loss_with_logits(boundary_logits_4d, boundary_targets)

    completion_kernel = int(pl_module.hparams.config.get("change_dense_completion_kernel", 5))
    completion_dilate = int(pl_module.hparams.config.get("change_dense_completion_dilate", max(0, (completion_kernel - 1) // 2)))
    completion_loss = _dense_completion_loss(
        dense_logits_4d,
        gt_loss_4d,
        dilate=completion_dilate,
    )

    color_only_penalty = torch.tensor(0.0, device=device)
    color_only_mask = _compute_color_only_mask(batch, device)
    probs = torch.sigmoid(dense_logits_4d)
    if color_only_mask is not None:
        bg = 1.0 - gt_loss_4d
        denom = (color_only_mask * bg).sum().clamp_min(1.0)
        color_only_penalty = (probs * color_only_mask * bg).sum() / denom

    no_change_penalty = torch.tensor(0.0, device=device)
    no_change_max_penalty = torch.tensor(0.0, device=device)
    no_change_topk_penalty = torch.tensor(0.0, device=device)
    rural_background_penalty = torch.tensor(0.0, device=device)
    rural_background_topk_penalty = torch.tensor(0.0, device=device)

    no_change_mask = (global_targets == 0)
    if no_change_mask.any():
        probs_nc = probs[no_change_mask]
        no_change_penalty = probs_nc.mean()
        no_change_max_penalty = probs_nc.flatten(1).amax(dim=1).mean()
        topk = int(pl_module.hparams.config.get("change_no_change_topk", 64))
        no_change_topk_penalty = _topk_mean_map(probs_nc, topk).mean()

    rural_weight = _compute_rural_background_weight(batch, device, pl_module.hparams.config)
    if rural_weight is not None:
        bg = 1.0 - gt_loss_4d
        rural_bg = rural_weight * bg
        denom = rural_bg.sum().clamp_min(1.0)
        rural_background_penalty = (probs * rural_bg).sum() / denom
        rural_topk = int(pl_module.hparams.config.get("change_rural_background_topk", 64))
        rural_background_topk_penalty = _masked_topk_penalty(probs, rural_bg, rural_topk)

    total_loss = (
        float(pl_module.hparams.config.get("change_dense_bce_weight", 0.52)) * dense_bce
        + float(pl_module.hparams.config.get("change_dense_dice_weight", 0.48)) * dense_dice
        + float(pl_module.hparams.config.get("change_local_loss_weight", 0.10)) * local_bce
        + float(pl_module.hparams.config.get("change_global_loss_weight", 0.12)) * global_bce
        + float(pl_module.hparams.config.get("change_boundary_loss_weight", 0.10)) * boundary_loss
        + float(pl_module.hparams.config.get("change_boundary_dice_weight", 0.06)) * boundary_dice
        + float(pl_module.hparams.config.get("change_color_only_penalty_weight", 0.02)) * color_only_penalty
        + float(pl_module.hparams.config.get("change_deep_sup_loss_weight", 0.05)) * deep_sup
        + float(pl_module.hparams.config.get("change_no_change_penalty_weight", 0.14)) * no_change_penalty
        + float(pl_module.hparams.config.get("change_no_change_max_penalty_weight", 0.18)) * no_change_max_penalty
        + float(pl_module.hparams.config.get("change_no_change_topk_penalty_weight", 0.14)) * no_change_topk_penalty
        + float(pl_module.hparams.config.get("change_rural_background_penalty_weight", 0.0)) * rural_background_penalty
        + float(pl_module.hparams.config.get("change_rural_background_topk_penalty_weight", 0.0)) * rural_background_topk_penalty
        + float(pl_module.hparams.config.get("change_dense_completion_loss_weight", pl_module.hparams.config.get("change_dense_completion_weight", 0.08))) * completion_loss
    )

    return {
        "loss": total_loss,
        "dense_bce": dense_bce,
        "dense_dice": dense_dice,
        "local_bce": local_bce,
        "deep_sup": deep_sup,
        "global_bce": global_bce,
        "boundary_bce": boundary_loss,
        "boundary_dice": boundary_dice,
        "dense_completion_loss": completion_loss,
        "color_only_penalty": color_only_penalty,
        "no_change_penalty": no_change_penalty,
        "no_change_max_penalty": no_change_max_penalty,
        "no_change_topk_penalty": no_change_topk_penalty,
        "rural_background_penalty": rural_background_penalty,
        "rural_background_topk_penalty": rural_background_topk_penalty,
        "change_probs_global": infer.get("change_probs_global", global_logits.new_zeros((global_logits.shape[0], 1))),
        "osm_reliability": infer.get("osm_reliability", global_logits.new_zeros((global_logits.shape[0], 1))),
        "clip_reliability": infer.get("clip_reliability", global_logits.new_zeros((global_logits.shape[0], 1))),
    }
