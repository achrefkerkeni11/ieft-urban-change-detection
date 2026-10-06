"""Canonical full-scene reconstruction and evaluation for LEVIR-CD.

This module deliberately contains no model, dataset, or visualization code.  A
caller supplies *probabilities* for every tile, this module reconstructs one
source scene, and thresholding happens only after reconstruction.  In
particular, connected-component filters and no-change vetoes do not participate
in the scientific semantic metrics implemented here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np


CANONICAL_TILE_SIZE = 256
CANONICAL_TILE_STRIDE = 128
CANONICAL_HANN_FLOOR = 0.20
PROTOCOL_NAME = "ieft-levir-full-scene-hann-v1"
THRESHOLD_ARTIFACT_SCHEMA = "ieft.full_scene_threshold.v1"


class FullSceneProtocolError(ValueError):
    """Raised when inputs violate the canonical scene protocol."""


class IncompleteSceneError(RuntimeError):
    """Raised when one or more source pixels were not reconstructed."""


class SplitDisciplineError(RuntimeError):
    """Raised when validation-selection and final-test roles are mixed."""


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise FullSceneProtocolError(f"{name} must be positive, got {value}")
    return value


def _probability_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise FullSceneProtocolError(
            f"{name} must be a 2D probability map after squeeze, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FullSceneProtocolError(f"{name} contains non-finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise FullSceneProtocolError(f"{name} must contain probabilities in [0, 1]")
    return array


def _binary_mask(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise FullSceneProtocolError(
            f"{name} must be a 2D mask after squeeze, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FullSceneProtocolError(f"{name} contains non-finite values")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=False)
    is_zero = array == 0
    if np.all(is_zero | (array == 1)):
        return array.astype(bool, copy=False)
    if np.all(is_zero | (array == 255)):
        return array == 255
    raise FullSceneProtocolError(
        f"{name} must be an explicitly binary 0/1 or 0/255 mask"
    )


def validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise FullSceneProtocolError(
            f"threshold must be a finite value in [0, 1], got {threshold}"
        )
    return threshold


def modified_hann_window(size: int = CANONICAL_TILE_SIZE) -> np.ndarray:
    """Return the exact retained ``0.20 + 0.80 * outer(hanning, hanning)`` window."""

    size = _positive_int(size, "size")
    one_dimensional = np.hanning(size).astype(np.float32)
    window = np.outer(one_dimensional, one_dimensional).astype(np.float32)
    window /= max(float(window.max()), 1e-6)
    return (CANONICAL_HANN_FLOOR + (1.0 - CANONICAL_HANN_FLOOR) * window).astype(
        np.float32
    )


def tile_starts(length: int, tile_size: int, tile_stride: int) -> Tuple[int, ...]:
    """Return boundary-covering starts without padding or duplicate final tiles."""

    length = _positive_int(length, "length")
    tile_size = _positive_int(tile_size, "tile_size")
    tile_stride = _positive_int(tile_stride, "tile_stride")
    if tile_stride > tile_size:
        raise FullSceneProtocolError(
            "tile_stride cannot exceed tile_size because that would leave uncovered pixels"
        )
    if length <= tile_size:
        return (0,)
    starts = list(range(0, length - tile_size + 1, tile_stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


@dataclass(frozen=True, order=True)
class TileWindow:
    """One source-coordinate tile window."""

    x: int
    y: int
    width: int
    height: int


def scene_tile_windows(
    height: int,
    width: int,
    tile_size: int = CANONICAL_TILE_SIZE,
    tile_stride: int = CANONICAL_TILE_STRIDE,
) -> Tuple[TileWindow, ...]:
    """Enumerate tiles in row-major order, including final boundary-aligned tiles."""

    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    tile_size = _positive_int(tile_size, "tile_size")
    tile_stride = _positive_int(tile_stride, "tile_stride")
    ys = tile_starts(height, tile_size, tile_stride)
    xs = tile_starts(width, tile_size, tile_stride)
    return tuple(
        TileWindow(
            x=x,
            y=y,
            width=min(tile_size, width - x),
            height=min(tile_size, height - y),
        )
        for y in ys
        for x in xs
    )


def canonical_levir_windows(height: int = 1024, width: int = 1024) -> Tuple[TileWindow, ...]:
    """Convenience wrapper for the retained 256/128 LEVIR-CD protocol."""

    return scene_tile_windows(
        height,
        width,
        tile_size=CANONICAL_TILE_SIZE,
        tile_stride=CANONICAL_TILE_STRIDE,
    )


@dataclass(frozen=True)
class DecodedInstances:
    """Instance IDs paired with the unchanged scientific semantic mask."""

    semantic_mask: np.ndarray
    instance_map: np.ndarray


@dataclass(frozen=True)
class StitchedScene:
    """Finalized scene tensors before any semantic threshold is applied."""

    probability: np.ndarray
    weight_sum: np.ndarray
    coverage_count: np.ndarray
    center_probability: Optional[np.ndarray] = None
    offset: Optional[np.ndarray] = None
    tile_size: int = CANONICAL_TILE_SIZE
    tile_stride: int = CANONICAL_TILE_STRIDE

    def semantic_mask(self, threshold: float) -> np.ndarray:
        """Threshold the reconstructed probability exactly once per source pixel."""

        threshold = validate_threshold(threshold)
        return self.probability >= threshold

    def decode_instances(
        self,
        decoder: Callable[..., Any],
        threshold: float,
        **decoder_kwargs: Any,
    ) -> DecodedInstances:
        """Decode IDs after stitching without allowing semantic-mask mutation.

        The callback receives ``(center_probability, offset, semantic_mask)``.
        The semantic mask passed to it is a private copy, and mutation is treated
        as an error.  Instance IDs outside the scientific semantic foreground are
        also rejected.
        """

        if self.center_probability is None or self.offset is None:
            raise FullSceneProtocolError(
                "instance decoding requires stitched center probabilities and offsets"
            )
        semantic_mask = self.semantic_mask(threshold)
        decoder_mask = semantic_mask.copy()
        immutable_reference = decoder_mask.copy()
        instance_map = np.asarray(
            decoder(
                self.center_probability,
                self.offset,
                decoder_mask,
                **decoder_kwargs,
            )
        )
        if not np.array_equal(decoder_mask, immutable_reference):
            raise FullSceneProtocolError(
                "instance decoder mutated the scientific semantic mask"
            )
        if instance_map.shape != semantic_mask.shape:
            raise FullSceneProtocolError(
                f"instance map shape {instance_map.shape} != semantic shape {semantic_mask.shape}"
            )
        if not np.all(np.isfinite(instance_map)) or np.any(instance_map < 0):
            raise FullSceneProtocolError(
                "instance decoder returned negative, NaN, or infinite IDs"
            )
        if not np.all(instance_map == np.floor(instance_map)):
            raise FullSceneProtocolError("instance decoder returned non-integer IDs")
        if np.any(instance_map[~semantic_mask] != 0):
            raise FullSceneProtocolError(
                "instance decoder assigned IDs outside the scientific semantic mask"
            )
        if not np.array_equal(instance_map > 0, semantic_mask):
            raise FullSceneProtocolError(
                "instance IDs must preserve every scientific semantic foreground pixel"
            )
        return DecodedInstances(
            semantic_mask=semantic_mask.copy(),
            instance_map=instance_map.astype(np.int32, copy=False),
        )


@dataclass(frozen=True)
class StitchedSceneArchive:
    """One canonical source-scene archive, including optional instance fields."""

    scene_id: str
    split: str
    stitched: StitchedScene
    ground_truth: np.ndarray

    def semantic_prediction(self) -> "ScenePrediction":
        """Return the semantic-only view used by scientific pixel metrics."""

        return ScenePrediction(
            scene_id=self.scene_id,
            split=self.split,
            probability=self.stitched.probability,
            ground_truth=self.ground_truth,
            weight_sum=self.stitched.weight_sum,
            coverage_count=self.stitched.coverage_count,
        ).validated()


class FullSceneAccumulator:
    """Accumulate overlapping probability tiles for exactly one source scene."""

    def __init__(
        self,
        height: int,
        width: int,
        *,
        tile_size: int = CANONICAL_TILE_SIZE,
        tile_stride: int = CANONICAL_TILE_STRIDE,
        stitch_instances: bool = False,
        enforce_complete_tile_set: bool = True,
    ) -> None:
        self.height = _positive_int(height, "height")
        self.width = _positive_int(width, "width")
        self.tile_size = _positive_int(tile_size, "tile_size")
        self.tile_stride = _positive_int(tile_stride, "tile_stride")
        if self.tile_stride > self.tile_size:
            raise FullSceneProtocolError("tile_stride cannot exceed tile_size")
        self.stitch_instances = bool(stitch_instances)
        self.enforce_complete_tile_set = bool(enforce_complete_tile_set)
        self._window = modified_hann_window(self.tile_size)
        self._expected_windows = frozenset(
            scene_tile_windows(
                self.height,
                self.width,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            )
        )
        self._seen_windows: set[TileWindow] = set()
        self._finalized = False

        shape = (self.height, self.width)
        self.prob_sum = np.zeros(shape, dtype=np.float32)
        self.weight_sum = np.zeros(shape, dtype=np.float32)
        self.coverage_count = np.zeros(shape, dtype=np.uint16)
        self.center_sum = np.zeros(shape, dtype=np.float32) if self.stitch_instances else None
        self.offset_sum = (
            np.zeros((2, self.height, self.width), dtype=np.float32)
            if self.stitch_instances
            else None
        )

    @property
    def expected_tile_count(self) -> int:
        return len(self._expected_windows)

    @property
    def received_tile_count(self) -> int:
        return len(self._seen_windows)

    def add_tile(
        self,
        *,
        x: int,
        y: int,
        probability: Any,
        center_probability: Any = None,
        offset: Any = None,
    ) -> None:
        """Blend one unthresholded tile in source coordinates."""

        if self._finalized:
            raise FullSceneProtocolError("cannot add a tile after finalization")
        x, y = int(x), int(y)
        probability_array = _probability_map(probability, "probability")
        tile_height, tile_width = probability_array.shape
        window = TileWindow(x=x, y=y, width=tile_width, height=tile_height)
        if window not in self._expected_windows:
            raise FullSceneProtocolError(
                f"tile window {window} is not part of the configured source grid"
            )
        if window in self._seen_windows:
            raise FullSceneProtocolError(f"duplicate tile window: {window}")

        if self.stitch_instances:
            if center_probability is None or offset is None:
                raise FullSceneProtocolError(
                    "every tile must provide center_probability and offset when instance stitching is enabled"
                )
            center_array = _probability_map(center_probability, "center_probability")
            if center_array.shape != probability_array.shape:
                raise FullSceneProtocolError(
                    "center_probability and semantic probability shapes differ"
                )
            offset_array = np.asarray(offset, dtype=np.float32)
            if offset_array.shape == (tile_height, tile_width, 2):
                offset_array = np.moveaxis(offset_array, -1, 0)
            if offset_array.shape != (2, tile_height, tile_width):
                raise FullSceneProtocolError(
                    f"offset must be [2,H,W] or [H,W,2], got {offset_array.shape}"
                )
            if not np.all(np.isfinite(offset_array)):
                raise FullSceneProtocolError("offset contains non-finite values")
        elif center_probability is not None or offset is not None:
            raise FullSceneProtocolError(
                "construct with stitch_instances=True before supplying instance outputs"
            )

        y_slice = slice(y, y + tile_height)
        x_slice = slice(x, x + tile_width)
        tile_weight = self._window[:tile_height, :tile_width]
        self.prob_sum[y_slice, x_slice] += probability_array * tile_weight
        self.weight_sum[y_slice, x_slice] += tile_weight
        self.coverage_count[y_slice, x_slice] += 1
        if self.stitch_instances:
            assert self.center_sum is not None and self.offset_sum is not None
            self.center_sum[y_slice, x_slice] += center_array * tile_weight
            self.offset_sum[:, y_slice, x_slice] += offset_array * tile_weight[None, :, :]
        self._seen_windows.add(window)

    def finalize(self) -> StitchedScene:
        """Validate complete coverage and return blended full-scene tensors."""

        if self._finalized:
            raise FullSceneProtocolError("scene accumulator was already finalized")
        if self.enforce_complete_tile_set and self._seen_windows != self._expected_windows:
            missing = sorted(self._expected_windows.difference(self._seen_windows))
            extra = sorted(self._seen_windows.difference(self._expected_windows))
            raise IncompleteSceneError(
                f"incomplete tile set: missing={missing[:8]}"
                f"{' ...' if len(missing) > 8 else ''}, extra={extra[:8]}"
            )
        if np.any(self.coverage_count <= 0):
            uncovered = int(np.count_nonzero(self.coverage_count <= 0))
            raise IncompleteSceneError(f"coverage_count is zero for {uncovered} source pixels")
        if np.any(self.weight_sum <= 0.0):
            unweighted = int(np.count_nonzero(self.weight_sum <= 0.0))
            raise IncompleteSceneError(f"weight_sum is zero for {unweighted} source pixels")

        probability = self.prob_sum / np.maximum(self.weight_sum, 1e-6)
        center_probability = None
        offset = None
        if self.stitch_instances:
            assert self.center_sum is not None and self.offset_sum is not None
            center_probability = self.center_sum / np.maximum(self.weight_sum, 1e-6)
            offset = self.offset_sum / np.maximum(self.weight_sum[None, :, :], 1e-6)
        self._finalized = True
        return StitchedScene(
            probability=probability.astype(np.float32, copy=False),
            weight_sum=self.weight_sum.copy(),
            coverage_count=self.coverage_count.copy(),
            center_probability=(
                center_probability.astype(np.float32, copy=False)
                if center_probability is not None
                else None
            ),
            offset=offset.astype(np.float32, copy=False) if offset is not None else None,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )


@dataclass(frozen=True)
class ScenePrediction:
    """One reconstructed scene and its original-resolution binary GT."""

    scene_id: str
    split: str
    probability: np.ndarray
    ground_truth: np.ndarray
    weight_sum: Optional[np.ndarray] = None
    coverage_count: Optional[np.ndarray] = None

    def validated(self) -> "ScenePrediction":
        scene_id = str(self.scene_id).strip()
        split = str(self.split).strip().lower()
        if not scene_id:
            raise FullSceneProtocolError("scene_id cannot be empty")
        if split not in {"train", "val", "test"}:
            raise SplitDisciplineError(f"scene {scene_id!r} has invalid split {split!r}")
        probability = _probability_map(self.probability, f"{scene_id}.probability")
        ground_truth = _binary_mask(self.ground_truth, f"{scene_id}.ground_truth")
        if probability.shape != ground_truth.shape:
            raise FullSceneProtocolError(
                f"scene {scene_id!r} probability/GT shapes differ: "
                f"{probability.shape} != {ground_truth.shape}"
            )
        if (self.weight_sum is None) != (self.coverage_count is None):
            raise FullSceneProtocolError(
                f"scene {scene_id!r} must provide both weight_sum and coverage_count"
            )
        weight_sum = None
        coverage_count = None
        if self.weight_sum is not None and self.coverage_count is not None:
            weight_sum = np.squeeze(np.asarray(self.weight_sum, dtype=np.float32))
            coverage_count = np.squeeze(np.asarray(self.coverage_count))
            if weight_sum.shape != probability.shape or coverage_count.shape != probability.shape:
                raise FullSceneProtocolError(
                    f"scene {scene_id!r} coverage tensors must match {probability.shape}"
                )
            if not np.all(np.isfinite(weight_sum)) or np.any(weight_sum <= 0.0):
                raise IncompleteSceneError(
                    f"scene {scene_id!r} contains non-positive or non-finite weight_sum"
                )
            if not np.all(np.isfinite(coverage_count)) or np.any(coverage_count <= 0):
                raise IncompleteSceneError(
                    f"scene {scene_id!r} contains non-positive or non-finite coverage_count"
                )
            if not np.all(coverage_count == np.floor(coverage_count)):
                raise FullSceneProtocolError(
                    f"scene {scene_id!r} coverage_count must contain integers"
                )
        return ScenePrediction(
            scene_id,
            split,
            probability,
            ground_truth,
            weight_sum,
            coverage_count,
        )


@dataclass(frozen=True)
class ConfusionCounts:
    """Integer global confusion counts at original source-pixel resolution."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def pixel_count(self) -> int:
        return int(self.tp + self.fp + self.fn + self.tn)

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )

    @classmethod
    def from_masks(cls, prediction: Any, ground_truth: Any) -> "ConfusionCounts":
        prediction_array = _binary_mask(prediction, "prediction")
        ground_truth_array = _binary_mask(ground_truth, "ground_truth")
        if prediction_array.shape != ground_truth_array.shape:
            raise FullSceneProtocolError(
                f"prediction/GT shapes differ: {prediction_array.shape} != {ground_truth_array.shape}"
            )
        counts = cls(
            tp=int(np.count_nonzero(prediction_array & ground_truth_array)),
            fp=int(np.count_nonzero(prediction_array & ~ground_truth_array)),
            fn=int(np.count_nonzero(~prediction_array & ground_truth_array)),
            tn=int(np.count_nonzero(~prediction_array & ~ground_truth_array)),
        )
        expected_pixels = int(ground_truth_array.size)
        if counts.pixel_count != expected_pixels:
            raise FullSceneProtocolError(
                f"confusion pixels {counts.pixel_count} != source pixels {expected_pixels}"
            )
        return counts

    def metrics(self) -> Dict[str, float]:
        def safe_div(numerator: int, denominator: int) -> float:
            return float(numerator) / float(denominator) if denominator > 0 else 0.0

        return {
            "precision": safe_div(self.tp, self.tp + self.fp),
            "recall": safe_div(self.tp, self.tp + self.fn),
            "f1": safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "iou": safe_div(self.tp, self.tp + self.fp + self.fn),
            "oa": safe_div(self.tp + self.tn, self.pixel_count),
        }

    def as_dict(self) -> Dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    threshold: float
    scene_count: int
    source_pixel_count: int
    confusion: ConfusionCounts
    metrics: Mapping[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "threshold": self.threshold,
            "scene_count": self.scene_count,
            "source_pixel_count": self.source_pixel_count,
            "confusion": self.confusion.as_dict(),
            "metrics": dict(self.metrics),
        }


def _validated_scenes(
    scenes: Iterable[ScenePrediction], expected_split: str
) -> Tuple[ScenePrediction, ...]:
    expected_split = str(expected_split).strip().lower()
    validated = tuple(scene.validated() for scene in scenes)
    if not validated:
        raise FullSceneProtocolError("at least one scene is required")
    scene_ids = [scene.scene_id for scene in validated]
    if len(set(scene_ids)) != len(scene_ids):
        raise FullSceneProtocolError("scene IDs must be unique")
    wrong = [scene.scene_id for scene in validated if scene.split != expected_split]
    if wrong:
        raise SplitDisciplineError(
            f"expected only split={expected_split!r}; mismatched scenes: {wrong[:8]}"
        )
    return validated


def evaluate_scenes(
    scenes: Iterable[ScenePrediction],
    *,
    threshold: float,
    expected_split: str,
) -> EvaluationResult:
    """Evaluate each original source pixel exactly once."""

    threshold = validate_threshold(threshold)
    validated = _validated_scenes(scenes, expected_split)
    total = ConfusionCounts()
    source_pixel_count = 0
    for scene in validated:
        semantic_mask = scene.probability >= threshold
        counts = ConfusionCounts.from_masks(semantic_mask, scene.ground_truth)
        total = total + counts
        source_pixel_count += int(scene.ground_truth.size)
    if total.pixel_count != source_pixel_count:
        raise FullSceneProtocolError(
            f"global confusion pixels {total.pixel_count} != source pixels {source_pixel_count}"
        )
    return EvaluationResult(
        split=str(expected_split).lower(),
        threshold=threshold,
        scene_count=len(validated),
        source_pixel_count=source_pixel_count,
        confusion=total,
        metrics=total.metrics(),
    )


def threshold_grid(
    start: float = 0.500,
    stop: float = 0.990,
    step: float = 0.001,
) -> Tuple[float, ...]:
    """Build an inclusive, decimal-stable threshold grid."""

    start = validate_threshold(start)
    stop = validate_threshold(stop)
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise FullSceneProtocolError(f"threshold step must be positive, got {step}")
    if stop < start:
        raise FullSceneProtocolError("threshold stop must be >= start")
    count = int(np.floor(((stop - start) / step) + 1e-9)) + 1
    values = tuple(round(start + index * step, 12) for index in range(count))
    if values[-1] < stop:
        values = values + (round(stop, 12),)
    return values


@dataclass(frozen=True)
class ThresholdSweepResult:
    selected: EvaluationResult
    selection_metric: str
    tie_break: str
    thresholds: Tuple[float, ...]
    metric_values: Tuple[float, ...]

    def as_dict(self, include_curve: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "selected": self.selected.as_dict(),
            "selection_metric": self.selection_metric,
            "tie_break": self.tie_break,
            "grid": {
                "start": self.thresholds[0],
                "stop": self.thresholds[-1],
                "count": len(self.thresholds),
            },
        }
        if include_curve:
            result["curve"] = [
                {"threshold": threshold, self.selection_metric: value}
                for threshold, value in zip(self.thresholds, self.metric_values)
            ]
        return result


def sweep_validation_thresholds(
    scenes: Iterable[ScenePrediction],
    *,
    thresholds: Optional[Sequence[float]] = None,
    selection_metric: str = "iou",
    tie_break: str = "lowest",
) -> ThresholdSweepResult:
    """Select a threshold from validation scenes only.

    Counts for all thresholds are computed from per-scene sorted foreground and
    background probabilities.  This avoids materializing every scene in one
    giant array and preserves the exact ``probability >= threshold`` rule.
    """

    validated = _validated_scenes(scenes, "val")
    if thresholds is None:
        thresholds_tuple = threshold_grid()
    else:
        thresholds_tuple = tuple(validate_threshold(value) for value in thresholds)
    if not thresholds_tuple:
        raise FullSceneProtocolError("threshold grid cannot be empty")
    if any(
        right <= left for left, right in zip(thresholds_tuple, thresholds_tuple[1:])
    ):
        raise FullSceneProtocolError("threshold grid must be strictly increasing")
    if selection_metric not in {"iou", "f1", "precision", "recall", "oa"}:
        raise FullSceneProtocolError(f"unsupported selection metric: {selection_metric}")
    if tie_break not in {"lowest", "highest"}:
        raise FullSceneProtocolError("tie_break must be 'lowest' or 'highest'")

    # NumPy compares a float32 probability map with a scalar threshold in
    # float32 (including on NumPy 1.26 and 2.x).  Search the sorted float32
    # values with float32 thresholds so sweep counts exactly match the final
    # ``probability >= threshold`` evaluation at representational boundaries.
    threshold_array = np.asarray(thresholds_tuple, dtype=np.float32)
    tp = np.zeros(threshold_array.shape, dtype=np.int64)
    fp = np.zeros(threshold_array.shape, dtype=np.int64)
    positive_total = 0
    negative_total = 0
    source_pixel_count = 0
    for scene in validated:
        probability = scene.probability.reshape(-1)
        truth = scene.ground_truth.reshape(-1)
        positive = np.sort(probability[truth].astype(np.float32, copy=False))
        negative = np.sort(probability[~truth].astype(np.float32, copy=False))
        tp += positive.size - np.searchsorted(positive, threshold_array, side="left")
        fp += negative.size - np.searchsorted(negative, threshold_array, side="left")
        positive_total += int(positive.size)
        negative_total += int(negative.size)
        source_pixel_count += int(probability.size)

    fn = positive_total - tp
    tn = negative_total - fp

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        output = np.zeros_like(numerator, dtype=np.float64)
        np.divide(numerator, denominator, out=output, where=denominator > 0)
        return output

    metric_arrays = {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "oa": ratio(tp + tn, tp + fp + fn + tn),
    }
    selected_values = metric_arrays[selection_metric]
    best_value = float(selected_values.max())
    candidate_indices = np.flatnonzero(
        np.isclose(selected_values, best_value, rtol=0.0, atol=1e-15)
    )
    selected_index = int(candidate_indices[0] if tie_break == "lowest" else candidate_indices[-1])
    counts = ConfusionCounts(
        tp=int(tp[selected_index]),
        fp=int(fp[selected_index]),
        fn=int(fn[selected_index]),
        tn=int(tn[selected_index]),
    )
    if counts.pixel_count != source_pixel_count:
        raise FullSceneProtocolError(
            f"validation confusion pixels {counts.pixel_count} != source pixels {source_pixel_count}"
        )
    selected = EvaluationResult(
        split="val",
        threshold=thresholds_tuple[selected_index],
        scene_count=len(validated),
        source_pixel_count=source_pixel_count,
        confusion=counts,
        metrics=counts.metrics(),
    )
    return ThresholdSweepResult(
        selected=selected,
        selection_metric=selection_metric,
        tie_break=tie_break,
        thresholds=thresholds_tuple,
        metric_values=tuple(float(value) for value in selected_values),
    )


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: os.PathLike[str] | str, document: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def freeze_validation_threshold(
    path: os.PathLike[str] | str,
    sweep: ThresholdSweepResult,
    *,
    checkpoint_sha256: str,
    dataset_id: str,
    dataset_manifest_sha256: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write an immutable-protocol artifact selected exclusively on VAL."""

    if sweep.selected.split != "val":
        raise SplitDisciplineError("only validation-selected thresholds may be frozen")
    checkpoint_sha256 = str(checkpoint_sha256).strip().lower()
    dataset_id = str(dataset_id).strip()
    if len(checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in checkpoint_sha256
    ):
        raise FullSceneProtocolError("checkpoint_sha256 must be a 64-character hex digest")
    if not dataset_id:
        raise FullSceneProtocolError("dataset_id is required for a frozen threshold")

    document: Dict[str, Any] = {
        "schema": THRESHOLD_ARTIFACT_SCHEMA,
        "frozen": True,
        "selection_split": "val",
        "selected_threshold": sweep.selected.threshold,
        "selection_metric": sweep.selection_metric,
        "tie_break": sweep.tie_break,
        "protocol": {
            "name": PROTOCOL_NAME,
            "tile_size": CANONICAL_TILE_SIZE,
            "tile_stride": CANONICAL_TILE_STRIDE,
            "blend": "modified_hann",
            "hann_floor": CANONICAL_HANN_FLOOR,
            "threshold_stage": "after_full_scene_reconstruction",
            "semantic_postprocessing": "none",
        },
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": str(dataset_manifest_sha256).strip().lower(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": sweep.selected.as_dict(),
        "sweep": sweep.as_dict(include_curve=True),
        "metadata": dict(metadata or {}),
    }
    _atomic_json_write(path, document)
    return document


def load_frozen_threshold(
    path: os.PathLike[str] | str,
    *,
    checkpoint_sha256: Optional[str] = None,
    dataset_id: Optional[str] = None,
    dataset_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and validate a VAL-selected artifact before any TEST evaluation."""

    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise FullSceneProtocolError("threshold artifact must be a JSON object")
    if document.get("schema") != THRESHOLD_ARTIFACT_SCHEMA:
        raise FullSceneProtocolError("unsupported threshold artifact schema")
    if document.get("frozen") is not True or document.get("selection_split") != "val":
        raise SplitDisciplineError("TEST requires a frozen threshold selected on VAL")
    protocol = document.get("protocol")
    if not isinstance(protocol, dict):
        raise FullSceneProtocolError("threshold artifact is missing protocol metadata")
    expected_protocol = {
        "name": PROTOCOL_NAME,
        "tile_size": CANONICAL_TILE_SIZE,
        "tile_stride": CANONICAL_TILE_STRIDE,
        "blend": "modified_hann",
        "hann_floor": CANONICAL_HANN_FLOOR,
        "threshold_stage": "after_full_scene_reconstruction",
        "semantic_postprocessing": "none",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise FullSceneProtocolError(
                f"threshold artifact protocol mismatch for {key}: "
                f"{protocol.get(key)!r} != {expected!r}"
            )
    document["selected_threshold"] = validate_threshold(document.get("selected_threshold"))

    checks = {
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    for key, expected in checks.items():
        if expected is not None and str(document.get(key, "")) != str(expected):
            raise FullSceneProtocolError(
                f"threshold artifact {key} mismatch: {document.get(key)!r} != {expected!r}"
            )
    return document


def evaluate_test_with_frozen_threshold(
    scenes: Iterable[ScenePrediction],
    threshold_artifact: Mapping[str, Any],
) -> EvaluationResult:
    """Evaluate TEST once using a previously validated frozen artifact."""

    if threshold_artifact.get("frozen") is not True or threshold_artifact.get(
        "selection_split"
    ) != "val":
        raise SplitDisciplineError("TEST threshold was not frozen from VAL")
    return evaluate_scenes(
        scenes,
        threshold=validate_threshold(threshold_artifact.get("selected_threshold")),
        expected_split="test",
    )


def _npz_scalar(archive: Mapping[str, Any], keys: Sequence[str], path: Path) -> str:
    for key in keys:
        if key not in archive:
            continue
        value = np.asarray(archive[key])
        if value.size != 1:
            raise FullSceneProtocolError(f"{path}: {key} must be scalar")
        return str(value.reshape(-1)[0]).strip()
    return ""


def write_stitched_scene_archive(
    path: os.PathLike[str] | str,
    *,
    scene_id: str,
    split: str,
    stitched: StitchedScene,
    ground_truth: Any,
) -> None:
    """Persist one self-describing, unthresholded canonical scene archive."""

    if (
        int(stitched.tile_size) != CANONICAL_TILE_SIZE
        or int(stitched.tile_stride) != CANONICAL_TILE_STRIDE
    ):
        raise FullSceneProtocolError(
            "scientific scene archives require canonical tile_size=256 and tile_stride=128"
        )
    scene = ScenePrediction(
        scene_id=scene_id,
        split=split,
        probability=stitched.probability,
        ground_truth=ground_truth,
        weight_sum=stitched.weight_sum,
        coverage_count=stitched.coverage_count,
    ).validated()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            fields: Dict[str, Any] = {
                "source_pair_id": np.asarray(scene.scene_id),
                "split": np.asarray(scene.split),
                "protocol_name": np.asarray(PROTOCOL_NAME),
                "tile_size": np.int32(CANONICAL_TILE_SIZE),
                "tile_stride": np.int32(CANONICAL_TILE_STRIDE),
                "blend": np.asarray("modified_hann"),
                "hann_floor": np.float32(CANONICAL_HANN_FLOOR),
                "threshold_stage": np.asarray("after_full_scene_reconstruction"),
                "semantic_postprocessing": np.asarray("none"),
                "prob_map": scene.probability.astype(np.float32, copy=False),
                "gt_mask": scene.ground_truth.astype(np.uint8, copy=False),
                "weight_sum": np.asarray(scene.weight_sum, dtype=np.float32),
                "coverage_count": np.asarray(scene.coverage_count, dtype=np.uint16),
            }
            if stitched.center_probability is not None:
                fields["center_probability"] = stitched.center_probability.astype(
                    np.float32, copy=False
                )
            if stitched.offset is not None:
                fields["offset"] = stitched.offset.astype(np.float32, copy=False)
            np.savez_compressed(handle, **fields)
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_stitched_scene_archives(
    directory: os.PathLike[str] | str,
    *,
    expected_split: str,
    probability_key: str = "prob_map",
    ground_truth_key: str = "gt_mask",
    require_instance_outputs: bool = False,
) -> Iterator[StitchedSceneArchive]:
    """Load canonical scene archives without discarding stitched instance maps.

    Split metadata is mandatory.  Refusing unlabeled archives prevents a TEST
    directory from being accidentally reused for threshold selection.  Center
    and offset fields are either both present or both absent.  Set
    ``require_instance_outputs=True`` only for the optional post-stitch instance
    product; semantic evaluation never requires these auxiliary fields.
    """

    root = Path(directory)
    files = sorted(root.glob("*.npz"))
    if not files and (root / "npz").is_dir():
        files = sorted((root / "npz").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no scene .npz files found under {root}")
    expected_split = str(expected_split).strip().lower()
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            split = _npz_scalar(archive, ("split", "dataset_split"), path).lower()
            if not split:
                raise SplitDisciplineError(
                    f"{path} has no split metadata; refusing scientifically ambiguous input"
                )
            if split != expected_split:
                raise SplitDisciplineError(
                    f"{path} declares split={split!r}, expected {expected_split!r}"
                )
            protocol_name = _npz_scalar(archive, ("protocol_name",), path)
            tile_size = _npz_scalar(archive, ("tile_size",), path)
            tile_stride = _npz_scalar(archive, ("tile_stride",), path)
            blend = _npz_scalar(archive, ("blend",), path)
            hann_floor = _npz_scalar(archive, ("hann_floor",), path)
            threshold_stage = _npz_scalar(archive, ("threshold_stage",), path)
            semantic_postprocessing = _npz_scalar(
                archive, ("semantic_postprocessing",), path
            )
            protocol_values = {
                "protocol_name": (protocol_name, PROTOCOL_NAME),
                "tile_size": (tile_size, str(CANONICAL_TILE_SIZE)),
                "tile_stride": (tile_stride, str(CANONICAL_TILE_STRIDE)),
                "blend": (blend, "modified_hann"),
                "hann_floor": (hann_floor, str(CANONICAL_HANN_FLOOR)),
                "threshold_stage": (
                    threshold_stage,
                    "after_full_scene_reconstruction",
                ),
                "semantic_postprocessing": (semantic_postprocessing, "none"),
            }
            for field, (actual, expected) in protocol_values.items():
                if actual != expected:
                    raise FullSceneProtocolError(
                        f"{path}: {field}={actual!r} does not match canonical {expected!r}"
                    )
            scene_id = _npz_scalar(
                archive,
                ("source_pair_id", "scene_id", "patch_id"),
                path,
            ) or path.stem
            required = {
                probability_key,
                ground_truth_key,
                "weight_sum",
                "coverage_count",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise FullSceneProtocolError(
                    f"{path} is missing canonical scene tensors: {missing}"
                )
            probability = np.asarray(archive[probability_key], dtype=np.float32).copy()
            ground_truth = np.asarray(archive[ground_truth_key]).copy()
            weight_sum = np.asarray(archive["weight_sum"], dtype=np.float32).copy()
            coverage_count = np.asarray(archive["coverage_count"]).copy()
            has_center = "center_probability" in archive.files
            has_offset = "offset" in archive.files
            if has_center != has_offset:
                raise FullSceneProtocolError(
                    f"{path} must contain both center_probability and offset, or neither"
                )
            if require_instance_outputs and not has_center:
                raise FullSceneProtocolError(
                    f"{path} has no stitched center_probability/offset instance outputs"
                )
            center_probability = (
                np.asarray(archive["center_probability"], dtype=np.float32).copy()
                if has_center
                else None
            )
            offset = (
                np.asarray(archive["offset"], dtype=np.float32).copy()
                if has_offset
                else None
            )

        semantic = ScenePrediction(
            scene_id=scene_id,
            split=split,
            probability=probability,
            ground_truth=ground_truth,
            weight_sum=weight_sum,
            coverage_count=coverage_count,
        ).validated()
        if center_probability is not None:
            center_probability = _probability_map(
                center_probability,
                f"scene {scene_id!r} center_probability",
            )
            if center_probability.shape != semantic.probability.shape:
                raise FullSceneProtocolError(
                    f"scene {scene_id!r} center shape {center_probability.shape} != "
                    f"semantic shape {semantic.probability.shape}"
                )
            if offset is None:  # pragma: no cover - paired-field check above.
                raise AssertionError("paired offset unexpectedly absent")
            offset = np.asarray(offset, dtype=np.float32)
            if offset.shape != (2, *semantic.probability.shape):
                raise FullSceneProtocolError(
                    f"scene {scene_id!r} offset shape {offset.shape} != "
                    f"{(2, *semantic.probability.shape)}"
                )
            if not np.all(np.isfinite(offset)):
                raise FullSceneProtocolError(
                    f"scene {scene_id!r} offset contains NaN or infinity"
                )
        stitched = StitchedScene(
            probability=semantic.probability,
            weight_sum=np.asarray(semantic.weight_sum, dtype=np.float32),
            coverage_count=np.asarray(semantic.coverage_count, dtype=np.uint16),
            center_probability=center_probability,
            offset=offset,
        )
        yield StitchedSceneArchive(
            scene_id=semantic.scene_id,
            split=semantic.split,
            stitched=stitched,
            ground_truth=semantic.ground_truth,
        )


def load_scene_archives(
    directory: os.PathLike[str] | str,
    *,
    expected_split: str,
    probability_key: str = "prob_map",
    ground_truth_key: str = "gt_mask",
) -> Iterator[ScenePrediction]:
    """Load the semantic-only view of canonical stitched scene archives."""

    for archive in load_stitched_scene_archives(
        directory,
        expected_split=expected_split,
        probability_key=probability_key,
        ground_truth_key=ground_truth_key,
        require_instance_outputs=False,
    ):
        yield archive.semantic_prediction()


def write_evaluation_report(
    path: os.PathLike[str] | str,
    result: EvaluationResult,
    *,
    threshold_artifact_path: Optional[os.PathLike[str] | str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    document: Dict[str, Any] = {
        "schema": "ieft.full_scene_evaluation.v1",
        "protocol": {
            "name": PROTOCOL_NAME,
            "tile_size": CANONICAL_TILE_SIZE,
            "tile_stride": CANONICAL_TILE_STRIDE,
            "blend": "modified_hann",
            "hann_floor": CANONICAL_HANN_FLOOR,
            "threshold_stage": "after_full_scene_reconstruction",
            "semantic_postprocessing": "none",
        },
        "evaluation": result.as_dict(),
        "threshold_artifact": (
            str(Path(threshold_artifact_path).resolve())
            if threshold_artifact_path is not None
            else None
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }
    _atomic_json_write(path, document)
    return document
