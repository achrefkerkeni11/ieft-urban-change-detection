"""Pseudo-instance targets derived from LEVIR-CD semantic change masks.

LEVIR-CD supplies binary semantic masks, not building instance annotations.
Consequently these targets are *pseudo* instances.  A plain connected-component
label merges touching buildings; when SciPy is available this module improves
that baseline with distance-transform peaks and marker-controlled watershed.
The connected-component implementation remains as a dependency-free fallback.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Runtime fallback is useful in lightweight inference environments.
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - exercised only without SciPy installed.
    _ndimage = None


SCIPY_AVAILABLE = _ndimage is not None
_NEIGHBOURS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)


def _fallback_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Small NumPy/Python 8-connected fallback used only when SciPy is absent."""

    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    next_label = 0
    for y0, x0 in np.argwhere(mask):
        if labels[y0, x0] != 0:
            continue
        next_label += 1
        labels[y0, x0] = next_label
        queue = deque([(int(y0), int(x0))])
        while queue:
            y, x = queue.popleft()
            for dy, dx in _NEIGHBOURS_8:
                yy, xx = y + dy, x + dx
                if (
                    0 <= yy < height
                    and 0 <= xx < width
                    and mask[yy, xx]
                    and labels[yy, xx] == 0
                ):
                    labels[yy, xx] = next_label
                    queue.append((yy, xx))
    return labels, next_label


def label_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return deterministic 8-connected labels for a two-dimensional mask."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {binary.shape}")
    if SCIPY_AVAILABLE:
        labels, count = _ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        return labels.astype(np.int32, copy=False), int(count)
    return _fallback_connected_components(binary)


def _peak_markers(
    component: np.ndarray,
    distance: np.ndarray,
    min_peak_distance: int,
    min_peak_height: float,
) -> Tuple[np.ndarray, list]:
    """Create one marker per separated distance-transform maximum plateau."""

    footprint_size = 2 * max(1, int(min_peak_distance)) + 1
    local_max = _ndimage.maximum_filter(
        distance,
        size=footprint_size,
        mode="constant",
        cval=0.0,
    )
    candidates = component & (distance >= float(min_peak_height))
    candidates &= np.isclose(distance, local_max, rtol=0.0, atol=1e-7)
    plateau_labels, plateau_count = _ndimage.label(
        candidates,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    ranked = []
    for plateau_id in range(1, int(plateau_count) + 1):
        ys, xs = np.where(plateau_labels == plateau_id)
        if ys.size == 0:
            continue
        values = distance[ys, xs]
        best = int(np.argmax(values))
        ranked.append((float(values[best]), int(ys[best]), int(xs[best])))

    # Deterministic non-maximum suppression.  Sorting coordinates breaks ties
    # identically on every worker/platform.
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    kept = []
    radius2 = float(max(1, min_peak_distance) ** 2)
    for score, y, x in ranked:
        if all((y - ky) ** 2 + (x - kx) ** 2 >= radius2 for _, ky, kx in kept):
            kept.append((score, y, x))

    if not kept:
        ys, xs = np.where(component)
        best = int(np.argmax(distance[ys, xs]))
        kept = [(float(distance[ys[best], xs[best]]), int(ys[best]), int(xs[best]))]

    markers = np.zeros(component.shape, dtype=np.int32)
    for marker_id, (_, y, x) in enumerate(kept, start=1):
        markers[y, x] = marker_id
    return markers, kept


def _watershed_component(
    component: np.ndarray,
    min_peak_distance: int,
    min_peak_height: float,
    min_instance_area: int,
) -> np.ndarray:
    """Split one connected component using EDT peaks; preserve every FG pixel."""

    distance = _ndimage.distance_transform_edt(component)
    markers, peaks = _peak_markers(
        component,
        distance,
        min_peak_distance=min_peak_distance,
        min_peak_height=min_peak_height,
    )
    if len(peaks) <= 1:
        return component.astype(np.int32)

    max_distance = float(distance.max())
    normalized = distance / max(max_distance, 1e-8)
    gradient = np.rint((1.0 - normalized) * 254.0).astype(np.uint8)
    gradient[~component] = 255
    split = _ndimage.watershed_ift(
        gradient,
        markers.astype(np.int32, copy=False),
        structure=np.ones((3, 3), dtype=np.uint8),
    ).astype(np.int32, copy=False)
    split[~component] = 0

    # scipy.ndimage.watershed_ift has deterministic but strongly label-ordered
    # tie handling on nearly symmetric plateaus.  If a valid deep peak receives
    # only a tiny basin, use the nearest-marker Voronoi partition as a stable
    # tie-breaker before deciding that the peak itself is spurious.
    areas = np.bincount(split.ravel(), minlength=len(peaks) + 1)
    if np.any(areas[1:] < max(1, int(min_instance_area))):
        _, nearest_marker_index = _ndimage.distance_transform_edt(
            markers == 0,
            return_indices=True,
        )
        nearest_split = markers[tuple(nearest_marker_index)].astype(np.int32, copy=False)
        nearest_split[~component] = 0
        nearest_areas = np.bincount(
            nearest_split.ravel(),
            minlength=len(peaks) + 1,
        )
        if np.count_nonzero(nearest_areas[1:] >= max(1, int(min_instance_area))) > np.count_nonzero(
            areas[1:] >= max(1, int(min_instance_area))
        ):
            split, areas = nearest_split, nearest_areas

    # Peaks on genuinely tiny protrusions create unstable pseudo-instances.
    # Remove their markers and repeat once; this merges rather than deletes.
    retained = [
        peak
        for marker_id, peak in enumerate(peaks, start=1)
        if int(areas[marker_id]) >= max(1, int(min_instance_area))
    ]
    if not retained:
        retained = [peaks[int(np.argmax(areas[1:]))]]
    if len(retained) != len(peaks):
        markers.fill(0)
        for marker_id, (_, y, x) in enumerate(retained, start=1):
            markers[y, x] = marker_id
        _, nearest_marker_index = _ndimage.distance_transform_edt(
            markers == 0,
            return_indices=True,
        )
        split = markers[tuple(nearest_marker_index)].astype(np.int32, copy=False)
        split[~component] = 0
    return split


def build_pseudo_instance_labels(
    semantic_mask: np.ndarray,
    *,
    use_watershed: bool = True,
    min_peak_distance: int = 12,
    min_peak_height: float = 3.0,
    min_instance_area: int = 64,
) -> np.ndarray:
    """Convert one binary semantic mask to deterministic pseudo-instance IDs.

    Watershed splitting is attempted only when SciPy is available.  The output
    foreground support is guaranteed to equal the input foreground support.
    """

    binary = np.asarray(semantic_mask) > 0.5
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D semantic mask, got shape {binary.shape}")
    components, component_count = label_connected_components(binary)
    instances = np.zeros(binary.shape, dtype=np.int32)
    next_instance = 0

    for component_id in range(1, component_count + 1):
        component = components == component_id
        if not component.any():
            continue
        if use_watershed and SCIPY_AVAILABLE:
            ys, xs = np.where(component)
            pad = 1
            y0, y1 = max(0, int(ys.min()) - pad), min(binary.shape[0], int(ys.max()) + pad + 1)
            x0, x1 = max(0, int(xs.min()) - pad), min(binary.shape[1], int(xs.max()) + pad + 1)
            local_component = component[y0:y1, x0:x1]
            local_split = _watershed_component(
                local_component,
                min_peak_distance=min_peak_distance,
                min_peak_height=min_peak_height,
                min_instance_area=min_instance_area,
            )
            local_count = int(local_split.max())
            target = instances[y0:y1, x0:x1]
            for local_id in range(1, local_count + 1):
                next_instance += 1
                target[local_split == local_id] = next_instance
        else:
            next_instance += 1
            instances[component] = next_instance

    if not np.array_equal(instances > 0, binary):
        raise RuntimeError("Pseudo-instance generation changed semantic foreground support")
    return instances


def _instance_center(instance_mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(instance_mask)
    if ys.size == 0:
        raise ValueError("Cannot compute a center for an empty instance")
    if SCIPY_AVAILABLE:
        distance = _ndimage.distance_transform_edt(instance_mask)
        values = distance[ys, xs]
        best = int(np.argmax(values))
        return int(ys[best]), int(xs[best])

    cy, cx = float(ys.mean()), float(xs.mean())
    best = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(ys[best]), int(xs[best])


@torch.no_grad()
def build_instance_targets(
    gt_mask: torch.Tensor,
    sigma: float = 6.0,
    *,
    use_watershed: bool = True,
    min_peak_distance: int = 12,
    min_peak_height: float = 3.0,
    min_instance_area: int = 64,
):
    """Build center heatmaps, ``(dy, dx)`` offsets, and foreground masks."""

    if gt_mask.ndim == 4 and gt_mask.shape[1] == 1:
        gt = gt_mask[:, 0]
    elif gt_mask.ndim == 3:
        gt = gt_mask
    else:
        raise ValueError(
            "gt_mask must be [B,H,W] or [B,1,H,W], got "
            f"{tuple(gt_mask.shape)}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    binary = gt.detach().float().cpu().numpy() > 0.5
    batch_size, height, width = binary.shape
    center_target = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    offset_target = np.zeros((batch_size, 2, height, width), dtype=np.float32)

    gaussian_radius = max(1, int(np.ceil(3.0 * float(sigma))))
    for batch_index in range(batch_size):
        labels = build_pseudo_instance_labels(
            binary[batch_index],
            use_watershed=use_watershed,
            min_peak_distance=min_peak_distance,
            min_peak_height=min_peak_height,
            min_instance_area=min_instance_area,
        )
        for instance_id in range(1, int(labels.max()) + 1):
            instance = labels == instance_id
            ys, xs = np.where(instance)
            if ys.size == 0:
                continue
            center_y, center_x = _instance_center(instance)
            offset_target[batch_index, 0, ys, xs] = center_y - ys
            offset_target[batch_index, 1, ys, xs] = center_x - xs

            y0, y1 = max(0, center_y - gaussian_radius), min(height, center_y + gaussian_radius + 1)
            x0, x1 = max(0, center_x - gaussian_radius), min(width, center_x + gaussian_radius + 1)
            grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
            gaussian = np.exp(
                -((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2)
                / (2.0 * float(sigma) ** 2)
            ).astype(np.float32)
            target_view = center_target[batch_index, 0, y0:y1, x0:x1]
            np.maximum(target_view, gaussian, out=target_view)

    device = gt_mask.device
    center_tensor = torch.from_numpy(center_target).to(device=device)
    offset_tensor = torch.from_numpy(offset_target).to(device=device)
    foreground = torch.from_numpy(binary[:, None]).to(device=device, dtype=torch.float32)
    return center_tensor, offset_tensor, foreground


class InstanceLoss(nn.Module):
    """Center heatmap plus foreground-masked offset regression loss."""

    def __init__(
        self,
        w_center: float = 1.0,
        w_offset: float = 0.05,
        sigma: float = 6.0,
        *,
        use_watershed: bool = True,
        min_peak_distance: int = 12,
        min_peak_height: float = 3.0,
        min_instance_area: int = 64,
    ) -> None:
        super().__init__()
        self.w_center = float(w_center)
        self.w_offset = float(w_offset)
        self.sigma = float(sigma)
        self.use_watershed = bool(use_watershed)
        self.min_peak_distance = int(min_peak_distance)
        self.min_peak_height = float(min_peak_height)
        self.min_instance_area = int(min_instance_area)

    def forward(
        self,
        center_logits: torch.Tensor,
        offset_pred: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if center_logits.ndim != 4 or center_logits.shape[1] != 1:
            raise ValueError(
                "center_logits must be [B,1,H,W], got "
                f"{tuple(center_logits.shape)}"
            )
        if offset_pred.ndim != 4 or offset_pred.shape[1] != 2:
            raise ValueError(
                "offset_pred must be [B,2,H,W] in (dy,dx) order, got "
                f"{tuple(offset_pred.shape)}"
            )
        target_size = tuple(gt_mask.shape[-2:])
        if tuple(center_logits.shape[-2:]) != target_size:
            center_logits = F.interpolate(
                center_logits.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        if tuple(offset_pred.shape[-2:]) != target_size:
            source_h, source_w = offset_pred.shape[-2:]
            offset_pred = F.interpolate(
                offset_pred.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            offset_pred = offset_pred.clone()
            offset_pred[:, 0].mul_(target_size[0] / float(source_h))
            offset_pred[:, 1].mul_(target_size[1] / float(source_w))

        center_target, offset_target, foreground = build_instance_targets(
            gt_mask,
            sigma=self.sigma,
            use_watershed=self.use_watershed,
            min_peak_distance=self.min_peak_distance,
            min_peak_height=self.min_peak_height,
            min_instance_area=self.min_instance_area,
        )
        center_target = center_target.to(dtype=center_logits.dtype)
        offset_target = offset_target.to(dtype=offset_pred.dtype)
        foreground = foreground.to(dtype=offset_pred.dtype)

        center_loss = F.mse_loss(torch.sigmoid(center_logits), center_target)
        denominator = (2.0 * foreground.sum()).clamp_min(1.0)
        offset_loss = (torch.abs(offset_pred - offset_target) * foreground).sum() / denominator
        total = self.w_center * center_loss + self.w_offset * offset_loss
        return {
            "loss_instance": total,
            "loss_center": center_loss.detach(),
            "loss_offset": offset_loss.detach(),
        }
