"""VAL-only threshold selection and frozen-threshold TEST evaluation.

The input directory must contain one already-stitched ``.npz`` per source
scene, written with ``IEFT.full_scene.write_stitched_scene_archive``.  Tile
inference should feed ``IEFT.full_scene.FullSceneAccumulator``; this CLI never
thresholds tiles and never applies semantic post-processing.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from IEFT.full_scene import (
    CANONICAL_TILE_SIZE,
    CANONICAL_TILE_STRIDE,
    evaluate_test_with_frozen_threshold,
    freeze_validation_threshold,
    load_frozen_threshold,
    load_scene_archives,
    load_stitched_scene_archives,
    PROTOCOL_NAME,
    sha256_file,
    sweep_validation_thresholds,
    threshold_grid,
    write_evaluation_report,
)


def _add_archive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing one self-identifying stitched scene NPZ per source pair.",
    )
    parser.add_argument("--probability-key", default="prob_map")
    parser.add_argument("--ground-truth-key", default="gt_mask")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument(
        "--checkpoint",
        help="Checkpoint file to hash for threshold provenance verification.",
    )
    checkpoint.add_argument(
        "--checkpoint-sha256",
        help="Precomputed SHA256 of the exact evaluated checkpoint.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable identifier shared by the VAL and TEST partitions of this dataset version.",
    )
    parser.add_argument(
        "--dataset-manifest-sha256",
        default="",
        help="Optional SHA256 of the immutable dataset/scene manifest.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical IEFT/LEVIR full-scene scientific evaluation: modified-Hann "
            "probability reconstruction, then one source-pixel threshold."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select-val", help="Sweep thresholds on VAL and freeze the selected threshold."
    )
    _add_archive_arguments(select)
    select.add_argument("--threshold-artifact", required=True)
    select.add_argument("--report", default="")
    select.add_argument("--threshold-start", type=float, default=0.500)
    select.add_argument("--threshold-stop", type=float, default=0.990)
    select.add_argument("--threshold-step", type=float, default=0.001)
    select.add_argument(
        "--selection-metric",
        choices=["iou", "f1", "precision", "recall", "oa"],
        default="iou",
    )
    select.add_argument("--tie-break", choices=["lowest", "highest"], default="lowest")

    test = subparsers.add_parser(
        "evaluate-test",
        help="Evaluate TEST with a frozen threshold artifact previously selected on VAL.",
    )
    _add_archive_arguments(test)
    test.add_argument("--threshold-artifact", required=True)
    test.add_argument("--report", required=True)
    test.add_argument(
        "--instance-output-dir",
        default="",
        help=(
            "Optional new/empty directory for post-stitch instance-ID archives. "
            "This does not alter semantic masks or scientific pixel metrics."
        ),
    )
    test.add_argument("--instance-center-threshold", type=float, default=0.3)
    test.add_argument("--instance-min-distance", type=int, default=4)
    return parser


def _checkpoint_digest(args: argparse.Namespace) -> str:
    if args.checkpoint:
        return sha256_file(args.checkpoint)
    return str(args.checkpoint_sha256).strip().lower()


def _load_prediction_manifest(
    args: argparse.Namespace,
    *,
    split: str,
    checkpoint_sha256: str,
) -> tuple[dict, Path]:
    path = Path(args.input_dir).resolve() / "prediction_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            "canonical evaluation requires prediction_manifest.json in the input "
            f"directory: {path}"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical prediction manifest {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != (
        "ieft.full_scene_prediction_manifest.v1"
    ):
        raise ValueError(f"unsupported canonical prediction manifest: {path}")
    expected = {
        "split": str(split).lower(),
        "dataset_id": str(args.dataset_id),
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(
                f"prediction manifest {key} mismatch: {document.get(key)!r} != {value!r}"
            )
    protocol = document.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("prediction manifest has no protocol object")
    required_protocol = {
        "name": PROTOCOL_NAME,
        "tile_size": CANONICAL_TILE_SIZE,
        "tile_stride": CANONICAL_TILE_STRIDE,
        "tile_batch_size": 1,
        "blend": "modified_hann",
        "hann_floor": 0.20,
        "threshold_applied": False,
        "semantic_postprocessing": "none",
    }
    for key, value in required_protocol.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"prediction manifest protocol mismatch for {key}: "
                f"{protocol.get(key)!r} != {value!r}"
            )
    return document, path


def _load_scenes(
    args: argparse.Namespace,
    split: str,
    checkpoint_sha256: str,
) -> tuple[tuple, dict, Path]:
    producer, manifest_path = _load_prediction_manifest(
        args,
        split=split,
        checkpoint_sha256=checkpoint_sha256,
    )
    scenes = tuple(
        load_scene_archives(
            args.input_dir,
            expected_split=split,
            probability_key=args.probability_key,
            ground_truth_key=args.ground_truth_key,
        )
    )
    declared_count = producer.get("source_scene_count")
    if not isinstance(declared_count, int) or declared_count != len(scenes):
        raise ValueError(
            "prediction manifest source_scene_count mismatch: "
            f"{declared_count!r} != {len(scenes)}"
        )
    return scenes, producer, manifest_path


def _select_validation(args: argparse.Namespace) -> dict:
    checkpoint_sha256 = _checkpoint_digest(args)
    scenes, _producer, producer_manifest_path = _load_scenes(
        args, "val", checkpoint_sha256
    )
    thresholds = threshold_grid(
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
    )
    sweep = sweep_validation_thresholds(
        scenes,
        thresholds=thresholds,
        selection_metric=args.selection_metric,
        tie_break=args.tie_break,
    )
    artifact = freeze_validation_threshold(
        args.threshold_artifact,
        sweep,
        checkpoint_sha256=checkpoint_sha256,
        dataset_id=args.dataset_id,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        metadata={
            "validation_input_dir": str(Path(args.input_dir).resolve()),
            "prediction_manifest": str(producer_manifest_path),
            "prediction_manifest_sha256": sha256_file(producer_manifest_path),
        },
    )
    if args.report:
        write_evaluation_report(
            args.report,
            sweep.selected,
            threshold_artifact_path=args.threshold_artifact,
            metadata={
                "purpose": "validation_threshold_selection",
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_id": args.dataset_id,
            },
        )
    return artifact


def _prepare_instance_output_directory(value: str) -> Path:
    output_dir = Path(value).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"instance output directory is not empty; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _instance_filename(scene_id: str) -> str:
    scene_id = str(scene_id).strip()
    if not scene_id or scene_id in {".", ".."}:
        raise ValueError("scene_id must be non-empty for instance export")
    if Path(scene_id).name != scene_id or "/" in scene_id or "\\" in scene_id:
        raise ValueError(f"unsafe scene_id for instance export: {scene_id!r}")
    return f"{scene_id}.instances.npz"


def _write_instance_archive(
    path: Path,
    *,
    scene_id: str,
    semantic_threshold: float,
    center_threshold: float,
    min_distance: int,
    semantic_mask: np.ndarray,
    instance_map: np.ndarray,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                source_pair_id=np.asarray(scene_id),
                split=np.asarray("test"),
                protocol_name=np.asarray(PROTOCOL_NAME),
                semantic_threshold=np.float32(semantic_threshold),
                center_threshold=np.float32(center_threshold),
                min_distance=np.int32(min_distance),
                semantic_mask=np.asarray(semantic_mask, dtype=np.uint8),
                instance_id_map=np.asarray(instance_map, dtype=np.int32),
                instance_count=np.int32(instance_map.max() if instance_map.size else 0),
                semantic_support_invariant=np.asarray(
                    "instance_id_map>0 equals semantic_mask"
                ),
            )
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _decode_test_instances(
    args: argparse.Namespace,
    threshold_artifact: Mapping[str, Any],
) -> dict:
    """Decode the optional instance product after the VAL threshold is frozen."""

    # Deferred import keeps semantic-only evaluation independent of Torch/SciPy.
    from IEFT.modules.instance_postprocess import (
        count_instances,
        decode_fullscene_instances,
    )

    center_threshold = float(args.instance_center_threshold)
    min_distance = int(args.instance_min_distance)
    if not 0.0 <= center_threshold <= 1.0:
        raise ValueError("--instance-center-threshold must be in [0,1]")
    if min_distance < 1:
        raise ValueError("--instance-min-distance must be at least 1")
    semantic_threshold = float(threshold_artifact["selected_threshold"])
    output_dir = _prepare_instance_output_directory(args.instance_output_dir)
    archives = load_stitched_scene_archives(
        args.input_dir,
        expected_split="test",
        probability_key=args.probability_key,
        ground_truth_key=args.ground_truth_key,
        require_instance_outputs=True,
    )
    rows = []
    for archive in archives:
        decoded = archive.stitched.decode_instances(
            decode_fullscene_instances,
            threshold=semantic_threshold,
            center_threshold=center_threshold,
            min_distance=min_distance,
            center_is_logits=False,
        )
        if not np.array_equal(decoded.instance_map > 0, decoded.semantic_mask):
            raise RuntimeError(
                f"scene {archive.scene_id!r}: instance IDs changed semantic foreground"
            )
        output_path = output_dir / _instance_filename(archive.scene_id)
        _write_instance_archive(
            output_path,
            scene_id=archive.scene_id,
            semantic_threshold=semantic_threshold,
            center_threshold=center_threshold,
            min_distance=min_distance,
            semantic_mask=decoded.semantic_mask,
            instance_map=decoded.instance_map,
        )
        rows.append(
            {
                "scene_id": archive.scene_id,
                "archive": output_path.name,
                "instance_count": count_instances(decoded.instance_map),
                "semantic_foreground_pixels": int(np.count_nonzero(decoded.semantic_mask)),
            }
        )
    return {
        "role": "optional_object_grouping_not_semantic_metric",
        "decode_stage": "after_full_scene_stitch_and_frozen_semantic_threshold",
        "semantic_support_invariant": "instance_id_map>0 equals semantic_mask",
        "semantic_threshold": semantic_threshold,
        "center_threshold": center_threshold,
        "min_distance": min_distance,
        "scene_count": len(rows),
        "total_instance_count": int(sum(row["instance_count"] for row in rows)),
        "output_dir": str(output_dir),
        "scenes": rows,
    }


def _evaluate_test(args: argparse.Namespace) -> dict:
    checkpoint_sha256 = _checkpoint_digest(args)
    artifact = load_frozen_threshold(
        args.threshold_artifact,
        checkpoint_sha256=checkpoint_sha256,
        dataset_id=args.dataset_id,
        dataset_manifest_sha256=(
            args.dataset_manifest_sha256 if args.dataset_manifest_sha256 else None
        ),
    )
    scenes, _producer, producer_manifest_path = _load_scenes(
        args, "test", checkpoint_sha256
    )
    result = evaluate_test_with_frozen_threshold(scenes, artifact)
    instance_analysis = None
    if str(args.instance_output_dir).strip():
        instance_analysis = _decode_test_instances(args, artifact)
    metadata = {
        "purpose": "final_test_after_checkpoint_and_threshold_freeze",
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_id": args.dataset_id,
        "test_input_dir": str(Path(args.input_dir).resolve()),
        "prediction_manifest": str(producer_manifest_path),
        "prediction_manifest_sha256": sha256_file(producer_manifest_path),
    }
    if instance_analysis is not None:
        metadata["instance_analysis"] = instance_analysis
    return write_evaluation_report(
        args.report,
        result,
        threshold_artifact_path=args.threshold_artifact,
        metadata=metadata,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select-val":
        document = _select_validation(args)
        summary = {
            "threshold_artifact": str(Path(args.threshold_artifact).resolve()),
            "selected_threshold": document["selected_threshold"],
            "validation": document["validation"],
        }
    elif args.command == "evaluate-test":
        document = _evaluate_test(args)
        summary = {
            "report": str(Path(args.report).resolve()),
            "evaluation": document["evaluation"],
        }
    else:  # pragma: no cover - argparse prevents this branch.
        raise AssertionError(f"unsupported command {args.command!r}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
