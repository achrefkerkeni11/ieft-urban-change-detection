import argparse

import pytest

from IEFT.full_scene import FullSceneProtocolError
from predict_full_scenes import (
    _output_directory,
    _safe_scene_filename,
    apply_temporal_osm_mode,
    build_parser,
    validate_offline_configuration,
)


def _minimal_cli():
    return [
        "--ckpt-path",
        "model.ckpt",
        "--data-root",
        "LEVIR CD",
        "--split",
        "val",
        "--output-dir",
        "predictions",
        "--dataset-id",
        "levir-cd-v1",
    ]


def test_parser_fixes_geometry_and_exposes_no_threshold_or_heuristic_flags():
    parser = build_parser()
    args = parser.parse_args(_minimal_cli())

    assert args.split == "val"
    assert "tile_size" not in vars(args)
    assert "tile_stride" not in vars(args)
    assert "threshold" not in vars(args)
    assert "min_region_size" not in vars(args)
    assert "veto" not in vars(args)

    with pytest.raises(SystemExit):
        parser.parse_args(_minimal_cli() + ["--fixed-threshold", "0.89"])


def test_parser_accepts_only_validation_or_test():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [value if value != "val" else "train" for value in _minimal_cli()]
        )


def test_t2_mode_forbids_t1_requirement_and_disables_pair_failure_gate():
    cfg = {
        "change_osm_mode": "t2",
        "levir_require_osm_t1": True,
        "levir_require_osm_t2": True,
        "levir_filter_failed_osm": True,
    }
    args = build_parser().parse_args(_minimal_cli())

    apply_temporal_osm_mode(cfg, args)

    assert cfg["levir_require_osm_t1"] is False
    assert cfg["levir_require_osm_t2"] is True
    assert cfg["levir_filter_failed_osm"] is False

    args.levir_require_osm_t1 = True
    with pytest.raises(FullSceneProtocolError, match="forbids"):
        apply_temporal_osm_mode(cfg, args)


def test_offline_validation_rejects_remote_or_missing_assets(tmp_path):
    cfg = {"vit": "https://example.invalid/model", "levir_use_osm": False}
    with pytest.raises(FullSceneProtocolError, match="remote ViT"):
        validate_offline_configuration(cfg, project_dir=tmp_path)

    cfg = {
        "vit": "vit_base_patch16_224",
        "levir_use_osm": True,
        "levir_osm_texts_json": "missing.json",
    }
    with pytest.raises(FileNotFoundError, match="legacy OSM JSON"):
        validate_offline_configuration(cfg, project_dir=tmp_path)


def test_output_directory_refuses_overwrite_and_scene_names_are_safe(tmp_path):
    output = _output_directory(str(tmp_path / "new"))
    assert output.is_dir()
    (output / "existing.npz").write_bytes(b"comparison output")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _output_directory(str(output))
    assert _safe_scene_filename("test_1") == "test_1.npz"
    with pytest.raises(FullSceneProtocolError, match="unsafe"):
        _safe_scene_filename("../test_1")
