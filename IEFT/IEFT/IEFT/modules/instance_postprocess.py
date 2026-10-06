"""Post-stitch decoding of changed-building pseudo instances.

Scientific semantic metrics must use the stitched semantic probability map
directly.  This module consumes that already-thresholded mask as an immutable
support constraint; instance grouping is a separate product and can never add
or remove semantic foreground pixels.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from IEFT.modules.instance_targets import SCIPY_AVAILABLE, label_connected_components

if SCIPY_AVAILABLE:
    from scipy import ndimage as _ndimage
else:  # pragma: no cover - exercised only without SciPy installed.
    _ndimage = None


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def _single_scene_map(value, *, name: str, channels: int) -> np.ndarray:
    array = _to_numpy(value)
    if channels == 1:
        if array.ndim == 4 and array.shape[:2] == (1, 1):
            array = array[0, 0]
        elif array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        elif array.ndim != 2:
            raise ValueError(f"{name} must describe exactly one scene, got {array.shape}")
    else:
        if array.ndim == 4 and array.shape[:2] == (1, channels):
            array = array[0]
        if array.ndim != 3 or array.shape[0] != channels:
            raise ValueError(f"{name} must be [{channels},H,W], got {array.shape}")
    return np.asarray(array, dtype=np.float32)


def _component_peaks(
    center_probability: np.ndarray,
    component: np.ndarray,
    threshold: float,
    min_distance: int,
) -> List[Tuple[int, int]]:
    """Find separated local maxima inside one semantic component."""

    candidate_values = np.where(component, center_probability, -np.inf)
    if SCIPY_AVAILABLE:
        size = 2 * max(1, int(min_distance)) + 1
        maxima = _ndimage.maximum_filter(
            candidate_values,
            size=size,
            mode="constant",
            cval=-np.inf,
        )
        candidates = component & (center_probability >= float(threshold))
        candidates &= np.isclose(center_probability, maxima, rtol=0.0, atol=1e-7)
        plateaus, count = _ndimage.label(
            candidates,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        ranked = []
        for plateau_id in range(1, int(count) + 1):
            ys, xs = np.where(plateaus == plateau_id)
            if ys.size:
                best = int(np.argmax(center_probability[ys, xs]))
                ranked.append(
                    (float(center_probability[ys[best], xs[best]]), int(ys[best]), int(xs[best]))
                )
    else:
        ys, xs = np.where(component & (center_probability >= float(threshold)))
        ranked = [
            (float(center_probability[y, x]), int(y), int(x))
            for y, x in zip(ys, xs)
        ]

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = []
    radius2 = float(max(1, int(min_distance)) ** 2)
    for _, y, x in ranked:
        if all((y - py) ** 2 + (x - px) ** 2 >= radius2 for py, px in selected):
            selected.append((y, x))

    # A low-confidence component is still semantic foreground.  Give it one
    # fallback center instead of silently erasing it from the instance product.
    if not selected:
        ys, xs = np.where(component)
        best = int(np.argmax(center_probability[ys, xs]))
        selected = [(int(ys[best]), int(xs[best]))]
    return selected


@torch.no_grad()
def decode_fullscene_instances(
    stitched_center,
    stitched_offset,
    semantic_change_mask,
    *,
    center_threshold: float = 0.3,
    min_distance: int = 4,
    center_is_logits: bool = False,
) -> np.ndarray:
    """Decode instances only after full-scene Hann stitching.

    Args:
        stitched_center: one stitched center probability map (or logits when
            ``center_is_logits=True``).
        stitched_offset: one stitched two-channel ``(dy, dx)`` map.
        semantic_change_mask: the independently thresholded scientific semantic
            mask.  It is copied and never modified.

    Returns:
        ``int32[H,W]`` IDs whose non-zero support exactly equals the supplied
        semantic mask.
    """

    center = _single_scene_map(stitched_center, name="stitched_center", channels=1)
    offsets = _single_scene_map(stitched_offset, name="stitched_offset", channels=2)
    semantic_array = _single_scene_map(
        semantic_change_mask,
        name="semantic_change_mask",
        channels=1,
    )
    if center_is_logits:
        center = 1.0 / (1.0 + np.exp(-np.clip(center, -80.0, 80.0)))
    elif np.any((center < 0.0) | (center > 1.0)):
        raise ValueError(
            "stitched_center must contain probabilities in [0,1] when "
            "center_is_logits=False"
        )
    if not 0.0 <= float(center_threshold) <= 1.0:
        raise ValueError("center_threshold must be in [0,1]")
    if int(min_distance) < 1:
        raise ValueError("min_distance must be at least 1 pixel")
    if offsets.shape[-2:] != center.shape or semantic_array.shape != center.shape:
        raise ValueError(
            "Center, offset, and semantic maps must share HxW; got "
            f"{center.shape}, {offsets.shape}, and {semantic_array.shape}"
        )
    if not np.isfinite(center).all() or not np.isfinite(offsets).all():
        raise ValueError("Stitched instance predictions contain NaN or infinity")

    if not np.isfinite(semantic_array).all():
        raise ValueError("semantic_change_mask contains NaN or infinity")
    is_binary = np.isclose(semantic_array, 0.0) | np.isclose(semantic_array, 1.0)
    if not bool(np.all(is_binary)):
        raise ValueError(
            "semantic_change_mask must already be thresholded with the frozen "
            "validation threshold; instance decoding never chooses a semantic threshold"
        )

    semantic = np.array(semantic_array > 0.5, dtype=bool, copy=True)
    components, component_count = label_connected_components(semantic)
    instance_map = np.zeros(semantic.shape, dtype=np.int32)
    next_instance = 0

    for component_id in range(1, component_count + 1):
        component = components == component_id
        ys, xs = np.where(component)
        if ys.size == 0:
            continue
        centers = _component_peaks(
            center,
            component,
            threshold=center_threshold,
            min_distance=min_distance,
        )
        if len(centers) == 1:
            next_instance += 1
            instance_map[component] = next_instance
            continue

        centers_array = np.asarray(centers, dtype=np.float32)
        voted_y = np.clip(ys + offsets[0, ys, xs], 0, semantic.shape[0] - 1)
        voted_x = np.clip(xs + offsets[1, ys, xs], 0, semantic.shape[1] - 1)
        votes = np.stack((voted_y, voted_x), axis=1)
        squared_distance = ((votes[:, None, :] - centers_array[None, :, :]) ** 2).sum(axis=2)
        assignments = np.argmin(squared_distance, axis=1)
        for local_id in range(len(centers)):
            selected = assignments == local_id
            if not np.any(selected):
                continue
            next_instance += 1
            instance_map[ys[selected], xs[selected]] = next_instance

        # Numerical or degenerate assignment must not drop semantic pixels.
        unassigned = component & (instance_map == 0)
        if unassigned.any():
            next_instance += 1
            instance_map[unassigned] = next_instance

    if not np.array_equal(instance_map > 0, semantic):
        raise RuntimeError("Instance grouping changed the scientific semantic mask")
    return instance_map


@torch.no_grad()
def decode_instances(
    center_logits,
    offset,
    change_mask,
    center_threshold: float = 0.3,
    min_distance: int = 4,
) -> np.ndarray:
    """Compatibility wrapper; inputs must already be full-scene stitched."""

    return decode_fullscene_instances(
        center_logits,
        offset,
        change_mask,
        center_threshold=center_threshold,
        min_distance=min_distance,
        center_is_logits=True,
    )


def count_instances(instance_map) -> int:
    array = np.asarray(instance_map)
    return int(array.max()) if array.size else 0
