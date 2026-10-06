import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from IEFT.datasets.levir_auxiliary import (
    ManifestIndex,
    assess_temporal_source,
    load_index_normalization,
    load_spectral_date,
)
from IEFT.datasets.levir_cd_dataset import LEVIRCDDataset
from IEFT.modules.cross_modal_fusion import BidirectionalFIEResidual
from IEFT.modules.temporal_auxiliary import OSMT2EarlyAdapter, TemporalIndexAdapter
from IEFT.modules.vilt_module import _canonical_dense_logits
from IEFT.osm_geometry import OSM_SPATIAL_CHANNEL_ORDER, rasterize_feature_collection


def _write_rgb(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(path)


def test_legacy_indices_are_validity_aware_and_do_not_fabricate_reflectance(tmp_path):
    cache_root = tmp_path / "data_spectral_v2"
    raster = cache_root / "rasters" / "val_1_spectral.npz"
    raster.parent.mkdir(parents=True)
    ndvi = np.asarray([[0.2, 999.0], [0.4, 0.6]], dtype=np.float32)
    ndwi = np.asarray([[0.1, -999.0], [0.3, 0.5]], dtype=np.float32)
    valid = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
    np.savez_compressed(
        raster,
        ndvi_t1=ndvi,
        ndwi_t1=ndwi,
        valid_mask_t1=valid,
        ndvi_t2=ndvi,
        ndwi_t2=ndwi,
        valid_mask_t2=valid,
    )
    manifest_path = cache_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "val_1": {
                    "source_id": "val_1",
                    "raster_path": "data_spectral_v2/rasters/val_1_spectral.npz",
                    "t1": {"status": "ok", "valid_pixel_ratio": 0.75},
                    "t2": {"status": "ok", "valid_pixel_ratio": 0.75},
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = ManifestIndex(str(manifest_path), kind="spectral")
    assessment = assess_temporal_source(
        "val_1",
        manifest,
        "spectral",
        allow_legacy_spectral_indices=True,
    )
    assert assessment["available"] == {"t1": True, "t2": True}
    channels, resized_valid = load_spectral_date(
        assessment["sample"]["t1"],
        manifest,
        (8, 8),
        sample_record=assessment["sample"],
        temporal_key="t1",
    )
    assert channels.shape == (8, 8, 5)
    assert resized_valid.shape == (8, 8)
    assert np.count_nonzero(channels[..., :3]) == 0
    assert np.max(np.abs(channels[..., 3:5])) <= 1.0
    assert np.all(channels[resized_valid == 0] == 0)


def test_t2_mode_never_opens_or_falls_back_to_t1_osm(tmp_path, monkeypatch):
    # Runtime loading must remain completely cache-local.
    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("runtime dataset attempted a network call")

    monkeypatch.setattr("socket.create_connection", forbidden_network)
    root = tmp_path / "levir"
    for part in ("A", "B"):
        _write_rgb(root / "test" / part / "test_1.png")
    _write_mask(root / "test" / "label" / "test_1.png")

    osm_root = tmp_path / "data_osm_t2_v2"
    t2_cache = osm_root / "raw" / "test" / "test_1" / "t2.geojson"
    t2_cache.parent.mkdir(parents=True)
    t2_cache.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"tags": {"building": "yes"}},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "levir-temporal-osm-v3",
        "metadata": {"requested_temporal_keys": ["t2"]},
        "samples": [
            {
                "filename": "test_1.png",
                "source_pair_id": "test_1",
                "split": "test",
                "region_id": 1,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                # If T1 were assessed or opened, construction would fail.
                "t1": {"status": "ok", "raw_geojson": "must_not_open.geojson"},
                "t2": {
                    "status": "ok",
                    "raw_geojson": "raw/test/test_1/t2.geojson",
                    "feature_count": 1,
                    "extraction_geometry": "geometry",
                    "osm_reliability": 0.7,
                },
            }
        ],
    }
    manifest_path = osm_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = LEVIRCDDataset(
        root=str(root),
        split="test",
        crop_size=16,
        tile_stride=16,
        osm_manifest_json=str(manifest_path),
        temporal_osm_mode="t2",
        use_temporal_osm=True,
        require_osm_t1=False,
        require_osm_t2=True,
        tokenizer_mode="simple",
    )
    item = dataset[0]
    assert int(item["has_osm_t1"]) == 0
    assert torch.count_nonzero(item["osm_struct_t1"]) == 0
    assert int(item["has_osm_t2"]) == 1
    assert tuple(item["osm_maps"].shape) == (5, 16, 16)
    assert torch.count_nonzero(item["osm_maps"][0]) > 0
    assert np.isclose(float(item["osm_reliability"]), 0.7)

    # An available, timestamped but empty OSM snapshot is still valid context;
    # it is not converted into a missing sample or a hard no-change label.
    t2_cache.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )
    empty_dataset = LEVIRCDDataset(
        root=str(root),
        split="test",
        crop_size=16,
        tile_stride=16,
        osm_manifest_json=str(manifest_path),
        temporal_osm_mode="t2",
        use_temporal_osm=True,
        require_osm_t1=False,
        require_osm_t2=True,
        tokenizer_mode="simple",
    )
    empty_item = empty_dataset[0]
    assert int(empty_item["has_osm_t2"]) == 1
    assert np.isclose(float(empty_item["osm_reliability"]), 0.7)
    assert torch.count_nonzero(empty_item["osm_maps"]) == 0


def test_osm_raster_channel_order_and_empty_is_valid_context():
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"tags": {"highway": "residential"}},
                "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
            }
        ],
    }
    raster = rasterize_feature_collection(document, [0, 0, 1, 1], 32, 32)
    assert OSM_SPATIAL_CHANNEL_ORDER == (
        "building",
        "transport",
        "water",
        "vegetation",
        "land_use",
    )
    assert raster.shape == (32, 32, 5)
    assert np.count_nonzero(raster[..., 1]) > 0
    assert np.count_nonzero(raster[..., 0]) == 0
    empty = rasterize_feature_collection(
        {"type": "FeatureCollection", "features": []}, [0, 0, 1, 1], 32, 32
    )
    assert not np.any(empty)


def test_all_new_residual_paths_are_exact_zero_at_initialization():
    torch.manual_seed(7)
    spectral = TemporalIndexAdapter(24, 4, hidden_dim=8, input_channels=5)
    channels = torch.rand(2, 5, 12, 10)
    valid = torch.ones(2, 1, 12, 10)
    output = spectral(
        channels,
        channels,
        valid,
        valid,
        torch.ones(2),
        torch.ones(2),
        torch.tensor([1, 2]),
        torch.tensor([2, 3]),
        torch.tensor([0.8, 0.5]),
    )
    assert torch.count_nonzero(output["spatial_t1"]) == 0
    assert torch.count_nonzero(output["spatial_t2"]) == 0
    assert torch.count_nonzero(output["temporal_spatial"]) == 0

    osm = OSMT2EarlyAdapter(24, 4, hidden_dim=8)
    osm_residual = osm(
        torch.rand(2, 5, 32, 32),
        torch.rand(2, 16),
        torch.tensor([0.9, 0.0]),
        torch.ones(2),
    )
    assert torch.count_nonzero(osm_residual) == 0

    bidirectional = BidirectionalFIEResidual(24, 4, dropout=0.0)
    residual = bidirectional(torch.rand(2, 18, 24), torch.rand(2, 5, 24))
    assert torch.count_nonzero(residual) == 0


def test_dense_logits_batch_axis_cannot_broadcast():
    dense = _canonical_dense_logits(torch.zeros(2, 1, 32, 32), batch_size=2)
    coarse = torch.zeros(2, 32, 32)
    fused = dense + coarse
    assert dense.shape == (2, 32, 32)
    assert fused.shape == (2, 32, 32)


def test_spatial_augmentation_is_identical_for_every_modality(monkeypatch):
    dataset = object.__new__(LEVIRCDDataset)
    dataset.random_aug = True
    random_values = iter((0.1, 0.1))  # horizontal flip, then vertical flip
    monkeypatch.setattr(
        "IEFT.datasets.levir_cd_dataset.random.random",
        lambda: next(random_values),
    )
    monkeypatch.setattr(
        "IEFT.datasets.levir_cd_dataset.random.randint",
        lambda _low, _high: 1,
    )
    grid = np.arange(12, dtype=np.float32).reshape(3, 4)
    rgb_t1 = np.repeat(grid[..., None], 3, axis=-1)
    rgb_t2 = rgb_t1 + 100.0
    spectral_t1 = np.repeat(grid[..., None], 5, axis=-1)
    spectral_t2 = spectral_t1 + 200.0
    valid_t1 = grid.copy()
    valid_t2 = grid + 300.0
    osm = np.repeat(grid[..., None], 5, axis=-1) + 400.0

    transformed = dataset._augment(
        rgb_t1,
        rgb_t2,
        grid.copy(),
        spectral_t1,
        spectral_t2,
        valid_t1,
        valid_t2,
        osm,
    )
    t1, t2, label, s1, s2, v1, v2, osm_maps = transformed
    assert np.array_equal(t1[..., 0], label)
    assert np.array_equal(t2[..., 0] - 100.0, label)
    assert np.array_equal(s1[..., 0], label)
    assert np.array_equal(s2[..., 0] - 200.0, label)
    assert np.array_equal(v1, label)
    assert np.array_equal(v2 - 300.0, label)
    assert np.array_equal(osm_maps[..., 0] - 400.0, label)


def test_spectral_normalization_accepts_train_only_metadata(tmp_path):
    stats = tmp_path / "stats.json"
    valid_document = {
        "split_used": "train",
        "validation_or_test_used": False,
        "source_manifest_sha256": "abc123",
        "channel_order": ["green", "red", "nir", "ndvi", "ndwi_mcfeeters"],
        "mean": [0.1, 0.1, 0.1, 0.0, 0.0],
        "std": [0.2, 0.2, 0.2, 0.3, 0.3],
    }
    stats.write_text(json.dumps(valid_document), encoding="utf-8")
    loaded = load_index_normalization(
        "train_stats",
        stats_path=str(stats),
        spectral_manifest_sha256="abc123",
    )
    assert loaded["mode"] == "train_stats"

    valid_document["split_used"] = "val"
    valid_document["validation_or_test_used"] = True
    stats.write_text(json.dumps(valid_document), encoding="utf-8")
    with np.testing.assert_raises(ValueError):
        load_index_normalization(
            "train_stats",
            stats_path=str(stats),
            spectral_manifest_sha256="abc123",
        )
