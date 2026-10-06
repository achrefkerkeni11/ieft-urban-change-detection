import json

import numpy as np
import pytest

from evaluate_full_scenes import main
from IEFT.full_scene import (
    FullSceneAccumulator,
    PROTOCOL_NAME,
    write_stitched_scene_archive,
)


def _write_scene(
    directory,
    scene_id,
    split,
    probability,
    ground_truth,
    *,
    center_probability=None,
    offset=None,
):
    stitch_instances = center_probability is not None or offset is not None
    accumulator = FullSceneAccumulator(
        256,
        256,
        stitch_instances=stitch_instances,
    )
    accumulator.add_tile(
        x=0,
        y=0,
        probability=probability,
        center_probability=center_probability,
        offset=offset,
    )
    write_stitched_scene_archive(
        directory / f"{scene_id}.npz",
        scene_id=scene_id,
        split=split,
        stitched=accumulator.finalize(),
        ground_truth=ground_truth,
    )


def _write_prediction_manifest(
    directory,
    *,
    split,
    checkpoint_sha256,
    dataset_id="levir-cd-v1",
    source_scene_count=1,
):
    document = {
        "schema": "ieft.full_scene_prediction_manifest.v1",
        "protocol": {
            "name": PROTOCOL_NAME,
            "tile_size": 256,
            "tile_stride": 128,
            "tile_batch_size": 1,
            "blend": "modified_hann",
            "hann_floor": 0.20,
            "threshold_applied": False,
            "semantic_postprocessing": "none",
        },
        "split": split,
        "dataset_id": dataset_id,
        "checkpoint_sha256": checkpoint_sha256,
        "source_scene_count": source_scene_count,
    }
    (directory / "prediction_manifest.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


def test_validation_freeze_then_test_cli_smoke(tmp_path):
    validation_dir = tmp_path / "val"
    test_dir = tmp_path / "test"
    validation_dir.mkdir()
    test_dir.mkdir()
    ground_truth = np.zeros((256, 256), dtype=np.uint8)
    ground_truth[:128] = 1
    probability = np.full((256, 256), 0.1, dtype=np.float32)
    probability[:128] = 0.9
    _write_scene(validation_dir, "val_1", "val", probability, ground_truth)
    _write_scene(test_dir, "test_1", "test", probability, ground_truth)
    _write_prediction_manifest(
        validation_dir, split="val", checkpoint_sha256="a" * 64
    )
    _write_prediction_manifest(test_dir, split="test", checkpoint_sha256="a" * 64)

    artifact = tmp_path / "frozen_threshold.json"
    validation_report = tmp_path / "validation_report.json"
    assert (
        main(
            [
                "select-val",
                "--input-dir",
                str(validation_dir),
                "--threshold-artifact",
                str(artifact),
                "--report",
                str(validation_report),
                "--checkpoint-sha256",
                "a" * 64,
                "--dataset-id",
                "levir-cd-v1",
                "--threshold-start",
                "0.5",
                "--threshold-stop",
                "0.9",
                "--threshold-step",
                "0.4",
            ]
        )
        == 0
    )
    frozen = json.loads(artifact.read_text(encoding="utf-8"))
    assert frozen["selection_split"] == "val"
    assert frozen["selected_threshold"] == 0.5

    test_report = tmp_path / "test_report.json"
    assert (
        main(
            [
                "evaluate-test",
                "--input-dir",
                str(test_dir),
                "--threshold-artifact",
                str(artifact),
                "--report",
                str(test_report),
                "--checkpoint-sha256",
                "a" * 64,
                "--dataset-id",
                "levir-cd-v1",
            ]
        )
        == 0
    )
    report = json.loads(test_report.read_text(encoding="utf-8"))
    assert report["evaluation"]["split"] == "test"
    assert report["evaluation"]["metrics"]["iou"] == 1.0
    assert report["evaluation"]["source_pixel_count"] == 256 * 256


def test_test_cli_decodes_instances_after_frozen_threshold_without_semantic_change(
    tmp_path,
):
    validation_dir = tmp_path / "val"
    test_dir = tmp_path / "test"
    validation_dir.mkdir()
    test_dir.mkdir()
    ground_truth = np.zeros((256, 256), dtype=np.uint8)
    ground_truth[24:80, 24:80] = 1
    ground_truth[140:220, 150:230] = 1
    probability = np.full((256, 256), 0.1, dtype=np.float32)
    probability[ground_truth > 0] = 0.9
    center_probability = np.zeros((256, 256), dtype=np.float32)
    center_probability[52, 52] = 1.0
    center_probability[180, 190] = 1.0
    offset = np.zeros((2, 256, 256), dtype=np.float32)
    _write_scene(validation_dir, "val_1", "val", probability, ground_truth)
    _write_scene(
        test_dir,
        "test_1",
        "test",
        probability,
        ground_truth,
        center_probability=center_probability,
        offset=offset,
    )
    _write_prediction_manifest(
        validation_dir, split="val", checkpoint_sha256="b" * 64
    )
    _write_prediction_manifest(test_dir, split="test", checkpoint_sha256="b" * 64)

    artifact = tmp_path / "frozen_threshold.json"
    assert (
        main(
            [
                "select-val",
                "--input-dir",
                str(validation_dir),
                "--threshold-artifact",
                str(artifact),
                "--checkpoint-sha256",
                "b" * 64,
                "--dataset-id",
                "levir-cd-v1",
                "--threshold-start",
                "0.5",
                "--threshold-stop",
                "0.5",
                "--threshold-step",
                "0.001",
            ]
        )
        == 0
    )

    instance_dir = tmp_path / "instances"
    test_report = tmp_path / "test_report.json"
    assert (
        main(
            [
                "evaluate-test",
                "--input-dir",
                str(test_dir),
                "--threshold-artifact",
                str(artifact),
                "--report",
                str(test_report),
                "--checkpoint-sha256",
                "b" * 64,
                "--dataset-id",
                "levir-cd-v1",
                "--instance-output-dir",
                str(instance_dir),
            ]
        )
        == 0
    )

    report = json.loads(test_report.read_text(encoding="utf-8"))
    assert report["evaluation"]["metrics"]["iou"] == 1.0
    analysis = report["metadata"]["instance_analysis"]
    assert analysis["decode_stage"] == (
        "after_full_scene_stitch_and_frozen_semantic_threshold"
    )
    assert analysis["scene_count"] == 1
    with np.load(instance_dir / "test_1.instances.npz", allow_pickle=False) as archive:
        semantic_mask = np.asarray(archive["semantic_mask"], dtype=bool)
        instance_map = np.asarray(archive["instance_id_map"])
    np.testing.assert_array_equal(semantic_mask, probability >= 0.5)
    np.testing.assert_array_equal(instance_map > 0, semantic_mask)


def test_cli_rejects_predictions_from_a_different_checkpoint(tmp_path):
    validation_dir = tmp_path / "val"
    validation_dir.mkdir()
    probability = np.full((256, 256), 0.5, dtype=np.float32)
    ground_truth = np.zeros((256, 256), dtype=np.uint8)
    _write_scene(validation_dir, "val_1", "val", probability, ground_truth)
    _write_prediction_manifest(
        validation_dir,
        split="val",
        checkpoint_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        main(
            [
                "select-val",
                "--input-dir",
                str(validation_dir),
                "--threshold-artifact",
                str(tmp_path / "frozen.json"),
                "--checkpoint-sha256",
                "d" * 64,
                "--dataset-id",
                "levir-cd-v1",
            ]
        )
