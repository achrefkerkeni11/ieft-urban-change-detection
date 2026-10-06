import json

import numpy as np
import pytest

from IEFT.full_scene import (
    CANONICAL_TILE_SIZE,
    CANONICAL_TILE_STRIDE,
    ConfusionCounts,
    FullSceneAccumulator,
    FullSceneProtocolError,
    IncompleteSceneError,
    ScenePrediction,
    SplitDisciplineError,
    canonical_levir_windows,
    evaluate_scenes,
    evaluate_test_with_frozen_threshold,
    freeze_validation_threshold,
    load_frozen_threshold,
    load_scene_archives,
    modified_hann_window,
    scene_tile_windows,
    sweep_validation_thresholds,
    threshold_grid,
    write_stitched_scene_archive,
)


def test_modified_hann_matches_protocol_formula_exactly():
    one_dimensional = np.hanning(CANONICAL_TILE_SIZE).astype(np.float32)
    expected = np.outer(one_dimensional, one_dimensional).astype(np.float32)
    expected /= max(float(expected.max()), 1e-6)
    expected = (0.20 + 0.80 * expected).astype(np.float32)

    actual = modified_hann_window()

    np.testing.assert_array_equal(actual, expected)
    assert float(actual.min()) == pytest.approx(0.20)
    assert float(actual.max()) == pytest.approx(1.0)


def test_canonical_1024_scene_has_49_boundary_covering_tiles():
    windows = canonical_levir_windows()

    assert len(windows) == 49
    assert sorted({window.x for window in windows}) == [0, 128, 256, 384, 512, 640, 768]
    assert sorted({window.y for window in windows}) == [0, 128, 256, 384, 512, 640, 768]
    assert all(window.width == 256 and window.height == 256 for window in windows)


def test_probability_reconstruction_preserves_full_scene_and_tracks_coverage():
    yy, xx = np.mgrid[:1024, :1024]
    source_probability = ((3 * yy + 5 * xx) % 997).astype(np.float32) / 996.0
    accumulator = FullSceneAccumulator(1024, 1024)

    for window in canonical_levir_windows():
        tile = source_probability[
            window.y : window.y + window.height,
            window.x : window.x + window.width,
        ]
        accumulator.add_tile(x=window.x, y=window.y, probability=tile)

    stitched = accumulator.finalize()

    assert accumulator.expected_tile_count == 49
    assert accumulator.received_tile_count == 49
    assert np.all(stitched.weight_sum > 0)
    assert np.all(stitched.coverage_count > 0)
    assert set(np.unique(stitched.coverage_count)) == {1, 2, 4}
    np.testing.assert_allclose(stitched.probability, source_probability, rtol=0, atol=3e-7)


def test_missing_tile_is_rejected_before_scientific_evaluation():
    accumulator = FullSceneAccumulator(512, 512)
    first = scene_tile_windows(512, 512)[0]
    accumulator.add_tile(
        x=first.x,
        y=first.y,
        probability=np.zeros((first.height, first.width), dtype=np.float32),
    )

    with pytest.raises(IncompleteSceneError, match="incomplete tile set"):
        accumulator.finalize()


def test_duplicate_or_off_grid_tiles_are_rejected():
    accumulator = FullSceneAccumulator(512, 512)
    tile = np.zeros((256, 256), dtype=np.float32)
    accumulator.add_tile(x=0, y=0, probability=tile)

    with pytest.raises(FullSceneProtocolError, match="duplicate"):
        accumulator.add_tile(x=0, y=0, probability=tile)
    with pytest.raises(FullSceneProtocolError, match="not part"):
        accumulator.add_tile(x=1, y=0, probability=tile)


def test_confusion_counts_equal_original_source_pixel_count():
    prediction = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    ground_truth = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    counts = ConfusionCounts.from_masks(prediction, ground_truth)

    assert counts == ConfusionCounts(tp=1, fp=1, fn=1, tn=1)
    assert counts.pixel_count == ground_truth.size


def test_evaluate_scenes_counts_each_source_pixel_once():
    scenes = [
        ScenePrediction(
            "val_1",
            "val",
            np.array([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32),
            np.array([[1, 1], [0, 0]], dtype=np.uint8),
        ),
        ScenePrediction(
            "val_2",
            "val",
            np.array([[0.7, 0.6], [0.4, 0.3]], dtype=np.float32),
            np.array([[1, 0], [0, 0]], dtype=np.uint8),
        ),
    ]

    result = evaluate_scenes(scenes, threshold=0.65, expected_split="val")

    assert result.source_pixel_count == 8
    assert result.confusion.pixel_count == 8
    assert result.confusion == ConfusionCounts(tp=3, fp=0, fn=0, tn=5)


def test_validation_sweep_selects_threshold_and_rejects_test_data():
    validation_scene = ScenePrediction(
        "val_1",
        "val",
        np.array([[0.90, 0.80], [0.70, 0.10]], dtype=np.float32),
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
    )

    sweep = sweep_validation_thresholds(
        [validation_scene], thresholds=(0.50, 0.75, 0.85)
    )

    assert sweep.selected.threshold == pytest.approx(0.75)
    assert sweep.selected.metrics["iou"] == pytest.approx(1.0)

    test_scene = ScenePrediction(
        "test_1",
        "test",
        validation_scene.probability,
        validation_scene.ground_truth,
    )
    with pytest.raises(SplitDisciplineError, match="expected only split='val'"):
        sweep_validation_thresholds([test_scene], thresholds=(0.5, 0.75))


def test_sweep_counts_match_float32_final_threshold_at_decimal_boundary():
    scene = ScenePrediction(
        "val_boundary",
        "val",
        np.array([[np.float32(0.89), 0.10], [0.20, 0.30]], dtype=np.float32),
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
    )

    sweep = sweep_validation_thresholds([scene], thresholds=(0.89,))
    direct = evaluate_scenes([scene], threshold=0.89, expected_split="val")

    assert sweep.selected.confusion == direct.confusion
    assert sweep.selected.confusion.tp == 1


def test_threshold_grid_is_inclusive_and_decimal_stable():
    grid = threshold_grid(0.500, 0.990, 0.001)

    assert len(grid) == 491
    assert grid[0] == 0.500
    assert grid[-1] == 0.990
    assert grid[390] == 0.890


def test_frozen_val_artifact_is_required_and_verified_for_test(tmp_path):
    validation_scene = ScenePrediction(
        "val_1",
        "val",
        np.array([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32),
        np.array([[1, 1], [0, 0]], dtype=np.uint8),
    )
    sweep = sweep_validation_thresholds([validation_scene], thresholds=(0.5, 0.75))
    artifact_path = tmp_path / "threshold.json"
    freeze_validation_threshold(
        artifact_path,
        sweep,
        checkpoint_sha256="a" * 64,
        dataset_id="levir-cd-v1",
        dataset_manifest_sha256="b" * 64,
    )
    artifact = load_frozen_threshold(
        artifact_path,
        checkpoint_sha256="a" * 64,
        dataset_id="levir-cd-v1",
        dataset_manifest_sha256="b" * 64,
    )
    test_scene = ScenePrediction(
        "test_1",
        "test",
        validation_scene.probability,
        validation_scene.ground_truth,
    )

    result = evaluate_test_with_frozen_threshold([test_scene], artifact)

    assert result.split == "test"
    assert result.threshold == sweep.selected.threshold
    assert result.confusion.pixel_count == 4

    artifact["selection_split"] = "test"
    with pytest.raises(SplitDisciplineError, match="not frozen from VAL"):
        evaluate_test_with_frozen_threshold([test_scene], artifact)


def test_threshold_artifact_rejects_protocol_or_checkpoint_mismatch(tmp_path):
    scene = ScenePrediction(
        "val_1",
        "val",
        np.full((2, 2), 0.9, dtype=np.float32),
        np.ones((2, 2), dtype=np.uint8),
    )
    sweep = sweep_validation_thresholds([scene], thresholds=(0.5,))
    artifact_path = tmp_path / "threshold.json"
    artifact = freeze_validation_threshold(
        artifact_path,
        sweep,
        checkpoint_sha256="c" * 64,
        dataset_id="levir-cd-v1",
    )

    with pytest.raises(FullSceneProtocolError, match="checkpoint_sha256 mismatch"):
        load_frozen_threshold(artifact_path, checkpoint_sha256="d" * 64)

    artifact["protocol"]["tile_stride"] = 256
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(FullSceneProtocolError, match="protocol mismatch"):
        load_frozen_threshold(artifact_path)


def test_instance_outputs_are_stitched_before_decode_without_semantic_mutation():
    height = width = 6
    semantic_probability = np.array(
        [
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.8, 0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8, 0.8, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.7, 0.7],
            [0.1, 0.1, 0.1, 0.1, 0.7, 0.7],
        ],
        dtype=np.float32,
    )
    center_probability = semantic_probability.copy()
    full_offset = np.stack(
        [np.full((height, width), 1.25), np.full((height, width), -0.75)]
    ).astype(np.float32)
    accumulator = FullSceneAccumulator(
        height,
        width,
        tile_size=4,
        tile_stride=2,
        stitch_instances=True,
    )
    for window in scene_tile_windows(height, width, 4, 2):
        ys = slice(window.y, window.y + window.height)
        xs = slice(window.x, window.x + window.width)
        accumulator.add_tile(
            x=window.x,
            y=window.y,
            probability=semantic_probability[ys, xs],
            center_probability=center_probability[ys, xs],
            offset=full_offset[:, ys, xs],
        )
    stitched = accumulator.finalize()

    np.testing.assert_allclose(stitched.center_probability, center_probability, atol=1e-7)
    np.testing.assert_allclose(stitched.offset, full_offset, atol=1e-7)

    def decoder(center, offset, change_mask):
        del center, offset
        return change_mask.astype(np.int32)

    decoded = stitched.decode_instances(decoder, threshold=0.5)
    expected_semantic = semantic_probability >= 0.5
    np.testing.assert_array_equal(decoded.semantic_mask, expected_semantic)
    np.testing.assert_array_equal(decoded.instance_map > 0, expected_semantic)


def test_instance_decoder_cannot_mutate_or_expand_semantic_mask():
    accumulator = FullSceneAccumulator(
        4, 4, tile_size=4, tile_stride=2, stitch_instances=True
    )
    accumulator.add_tile(
        x=0,
        y=0,
        probability=np.full((4, 4), 0.8, dtype=np.float32),
        center_probability=np.full((4, 4), 0.8, dtype=np.float32),
        offset=np.zeros((2, 4, 4), dtype=np.float32),
    )
    stitched = accumulator.finalize()

    def mutating_decoder(center, offset, change_mask):
        del center, offset
        change_mask[:] = False
        return np.zeros_like(change_mask, dtype=np.int32)

    with pytest.raises(FullSceneProtocolError, match="mutated"):
        stitched.decode_instances(mutating_decoder, threshold=0.5)

    mixed_probability = np.full((4, 4), 0.8, dtype=np.float32)
    mixed_probability[0, 0] = 0.1
    accumulator = FullSceneAccumulator(
        4, 4, tile_size=4, tile_stride=2, stitch_instances=True
    )
    accumulator.add_tile(
        x=0,
        y=0,
        probability=mixed_probability,
        center_probability=np.full((4, 4), 0.8, dtype=np.float32),
        offset=np.zeros((2, 4, 4), dtype=np.float32),
    )
    stitched = accumulator.finalize()

    def expanding_decoder(center, offset, change_mask):
        del center, offset, change_mask
        return np.ones((4, 4), dtype=np.int32)

    with pytest.raises(FullSceneProtocolError, match="outside"):
        stitched.decode_instances(expanding_decoder, threshold=0.5)

    def dropping_decoder(center, offset, change_mask):
        del center, offset, change_mask
        return np.zeros((4, 4), dtype=np.int32)

    with pytest.raises(FullSceneProtocolError, match="preserve every"):
        stitched.decode_instances(dropping_decoder, threshold=0.5)


def test_scene_archive_loader_requires_embedded_split_metadata(tmp_path):
    accumulator = FullSceneAccumulator(256, 256)
    accumulator.add_tile(
        x=0,
        y=0,
        probability=np.full((256, 256), 0.7, dtype=np.float32),
    )
    write_stitched_scene_archive(
        tmp_path / "val_1.npz",
        scene_id="val_1",
        split="val",
        stitched=accumulator.finalize(),
        ground_truth=np.ones((256, 256), dtype=np.uint8),
    )
    scenes = tuple(load_scene_archives(tmp_path, expected_split="val"))
    assert len(scenes) == 1
    assert scenes[0].scene_id == "val_1"

    unlabeled = tmp_path / "unlabeled"
    unlabeled.mkdir()
    np.savez_compressed(
        unlabeled / "mystery.npz",
        prob_map=np.full((2, 2), 0.7, dtype=np.float32),
        gt_mask=np.ones((2, 2), dtype=np.uint8),
    )
    with pytest.raises(SplitDisciplineError, match="no split metadata"):
        tuple(load_scene_archives(unlabeled, expected_split="val"))
