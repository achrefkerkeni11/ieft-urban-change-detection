"""Produce canonical, unthresholded IEFT full-scene prediction archives.

This is the model-inference half of the scientific protocol.  It always uses
256x256 tiles at stride 128, processes one tile at a time, reconstructs source
probabilities with the modified Hann window, and writes no semantic mask.  Use
``evaluate_full_scenes.py select-val`` and its frozen artifact for thresholding.

Use ``export_change_maps.py`` after official evaluation to create verified
visualizations. No component filter or veto is called by either canonical path.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from IEFT.full_scene import (
    CANONICAL_TILE_SIZE,
    CANONICAL_TILE_STRIDE,
    FullSceneAccumulator,
    FullSceneProtocolError,
    PROTOCOL_NAME,
    canonical_levir_windows,
    sha256_file,
    write_stitched_scene_archive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint inference into canonical unthresholded LEVIR full-scene archives."
        )
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable identifier for this immutable LEVIR dataset version.",
    )
    parser.add_argument(
        "--stitch-instances",
        action="store_true",
        help="Also stitch center probabilities and offsets; never changes the semantic map.",
    )

    # Offline legacy-OSM compatibility.  If the checkpoint hparams already
    # contain a path it is preserved; this flag provides an explicit relocated
    # copy without allowing a service call.
    parser.add_argument("--levir-osm-texts-json", default="")
    parser.add_argument(
        "--disable-osm",
        action="store_true",
        help="Explicit image-only ablation; recorded in producer provenance.",
    )

    # Source-backed cached auxiliary manifests.  Supplying a manifest enables
    # that modality through export_change_maps.build_cfg's existing policy.
    parser.add_argument("--levir-temporal-osm-manifest", default="")
    parser.add_argument(
        "--levir-temporal-osm-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--levir-temporal-osm-mode", choices=["t2", "paired"], default=None
    )
    parser.add_argument(
        "--levir-require-osm-t1", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--levir-require-osm-t2", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--levir-require-nonempty-osm", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--levir-filter-failed-osm", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--levir-osm-timestamp-policy",
        choices=["month_start", "month_end"],
        default=None,
    )
    parser.add_argument("--levir-osm-cache-size", type=int, default=None)

    parser.add_argument("--levir-spectral-manifest", default="")
    parser.add_argument(
        "--levir-spectral-indices-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--levir-require-indices-t1", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--levir-require-indices-t2", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--levir-accept-partial-indices",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--levir-min-index-valid-fraction", type=float, default=None)
    parser.add_argument(
        "--levir-spectral-missing-policy", choices=["filter", "mask"], default=None
    )
    parser.add_argument(
        "--levir-index-normalization", choices=["natural", "train_stats", "sensor_train_stats"], default=None
    )
    parser.add_argument("--levir-index-normalization-stats", default="")
    parser.add_argument("--levir-spectral-cache-size", type=int, default=None)

    parser.add_argument("--levir-tokenizer-mode", choices=["simple", "hf"], default="simple")
    parser.add_argument("--checkpoint-min-parameter-coverage", type=float, default=None)
    parser.add_argument(
        "--checkpoint-strict-compatibility",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _atomic_json_write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _resolve_local_path(value: str, *, base: Path, label: str) -> Optional[Path]:
    value = str(value or "").strip()
    if not value:
        return None
    if value.lower().startswith(("http://", "https://", "hf-hub:", "hf_hub:")):
        raise FullSceneProtocolError(f"{label} must be a local offline path, got {value!r}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured {label} does not exist: {path}")
    return path


def validate_offline_configuration(cfg: Dict[str, Any], *, project_dir: Path) -> Dict[str, Any]:
    """Resolve configured assets and fail instead of silently dropping a modality."""

    path_fields = []
    if bool(cfg.get("levir_use_osm", False)):
        path_fields.append(("levir_osm_texts_json", "legacy OSM JSON"))
    if bool(cfg.get("levir_temporal_osm_enabled", False)):
        path_fields.append(("levir_temporal_osm_manifest", "temporal OSM manifest"))
    if bool(cfg.get("levir_spectral_indices_enabled", False)):
        path_fields.append(("levir_spectral_manifest", "spectral manifest"))
    if str(cfg.get("vit_encoder_ckpt_path", "") or "").strip():
        path_fields.append(("vit_encoder_ckpt_path", "visual encoder checkpoint"))

    resolved: Dict[str, Any] = {}
    for field, label in path_fields:
        value = str(cfg.get(field, "") or "").strip()
        if not value:
            raise FileNotFoundError(f"{label} is enabled but {field} is empty")
        path = _resolve_local_path(value, base=project_dir, label=label)
        assert path is not None
        cfg[field] = str(path)
        resolved[field] = str(path)

    vit_name = str(cfg.get("vit", "") or "").strip()
    if vit_name.lower().startswith(("http://", "https://", "hf-hub:", "hf_hub:")):
        raise FullSceneProtocolError(f"remote ViT identifiers are forbidden: {vit_name!r}")
    cfg["levir_tokenizer_local_only"] = True
    cfg["levir_tokenizer_fallback_to_simple"] = False
    return resolved


def apply_temporal_osm_mode(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply the explicit T2-only contract without ever making T1 an eligibility gate."""

    if args.levir_temporal_osm_mode is not None:
        cfg["levir_temporal_osm_mode"] = args.levir_temporal_osm_mode
        cfg["change_osm_mode"] = args.levir_temporal_osm_mode
    mode = str(
        cfg.get("change_osm_mode", cfg.get("levir_temporal_osm_mode", "")) or ""
    ).strip().lower()
    if mode == "t2":
        if args.levir_require_osm_t1 is True:
            raise FullSceneProtocolError("T2-only OSM mode forbids requiring or loading OSM T1")
        cfg["levir_require_osm_t1"] = False
        if args.levir_require_osm_t2 is None:
            cfg["levir_require_osm_t2"] = True
        # The explicit T2 requirement handles T2 failures.  A legacy pair-wide
        # failure filter can otherwise reject sources solely because T1 is absent.
        if args.levir_filter_failed_osm is None:
            cfg["levir_filter_failed_osm"] = False


def _output_directory(path: str) -> Path:
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory must be absent or empty; refusing to overwrite {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _safe_scene_filename(scene_id: str) -> str:
    scene_id = str(scene_id).strip()
    if not scene_id or Path(scene_id).name != scene_id or any(
        separator in scene_id for separator in ("/", "\\")
    ):
        raise FullSceneProtocolError(f"unsafe source scene identifier: {scene_id!r}")
    return f"{scene_id}.npz"


def _as_2d_probability(tensor, *, name: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise FullSceneProtocolError(f"{name} must reduce to [H,W], got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise FullSceneProtocolError(f"{name} contains invalid probabilities")
    return array


def _as_offset(tensor, *, expected_shape: tuple[int, int]) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (*expected_shape, 2):
        array = np.moveaxis(array, -1, 0)
    if array.shape != (2, *expected_shape):
        raise FullSceneProtocolError(
            f"instance offset must reduce to [2,H,W], got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FullSceneProtocolError("instance offset contains non-finite values")
    return array


def _runtime_imports():
    # Keep parser/tests lightweight; model dependencies are imported only for an
    # actual producer run.
    import torch
    from PIL import Image
    from tqdm import tqdm

    from export_change_maps import (
        build_cfg,
        build_export_dataset,
        load_export_checkpoint,
        move_batch_to_device,
        resize_canonical_batch,
    )
    from IEFT.datasets.levir_cd_dataset import LEVIRCDDataset
    from IEFT.modules.vilt_module import ViLTransformerSS

    return {
        "torch": torch,
        "Image": Image,
        "tqdm": tqdm,
        "build_cfg": build_cfg,
        "build_export_dataset": build_export_dataset,
        "load_export_checkpoint": load_export_checkpoint,
        "move_batch_to_device": move_batch_to_device,
        "resize_canonical_batch": resize_canonical_batch,
        "LEVIRCDDataset": LEVIRCDDataset,
        "ViLTransformerSS": ViLTransformerSS,
    }


def run_prediction(args: argparse.Namespace) -> Dict[str, Any]:
    runtime = _runtime_imports()
    torch = runtime["torch"]
    Image = runtime["Image"]
    tqdm = runtime["tqdm"]

    project_dir = Path(__file__).resolve().parent
    checkpoint_path = _resolve_local_path(
        args.ckpt_path, base=Path.cwd(), label="model checkpoint"
    )
    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"LEVIR data root does not exist: {data_root}")
    output_dir = _output_directory(args.output_dir)

    # Existing config/dataset helpers are reused, but geometry and memory policy
    # are fixed here and are not exposed as CLI options.
    args.ckpt_path = str(checkpoint_path)
    args.data_root = str(data_root)
    args.model_input_size = CANONICAL_TILE_SIZE
    args.tile_size = CANONICAL_TILE_SIZE
    args.tile_stride = CANONICAL_TILE_STRIDE
    cfg = runtime["build_cfg"](args)
    cfg.update(
        {
            "batch_size": 1,
            "per_gpu_batchsize": 1,
            "num_workers": 0,
            "image_size": CANONICAL_TILE_SIZE,
            "model_input_size": CANONICAL_TILE_SIZE,
            "levir_tile_size": CANONICAL_TILE_SIZE,
            "levir_tile_stride": CANONICAL_TILE_STRIDE,
            "levir_tokenizer_local_only": True,
        }
    )
    apply_temporal_osm_mode(cfg, args)
    resolved_assets = validate_offline_configuration(cfg, project_dir=project_dir)

    dataset = runtime["build_export_dataset"](args, cfg)
    indices_by_source: Dict[str, list[int]] = defaultdict(list)
    for tile_index, metadata in enumerate(dataset.samples):
        indices_by_source[str(metadata["stem"])].append(tile_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = runtime["ViLTransformerSS"](cfg)
    checkpoint_report = runtime["load_export_checkpoint"](
        model,
        str(checkpoint_path),
        cfg,
        output_dir / "checkpoint_compatibility.json",
    )
    model.eval().to(device)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    started = time.perf_counter()
    archive_rows = []

    with torch.inference_mode():
        for scene_id in tqdm(dataset.retained_sources, desc=f"Canonical {args.split}"):
            scene_id = str(scene_id)
            tile_indices = indices_by_source.get(scene_id, [])
            if not tile_indices:
                raise FullSceneProtocolError(f"source {scene_id!r} has no tiles")
            first_meta = dataset.samples[tile_indices[0]]
            source_height = int(first_meta["source_height"])
            source_width = int(first_meta["source_width"])
            accumulator = FullSceneAccumulator(
                source_height,
                source_width,
                stitch_instances=bool(args.stitch_instances),
            )
            expected_windows = canonical_levir_windows(source_height, source_width)
            if len(tile_indices) != len(expected_windows):
                raise FullSceneProtocolError(
                    f"source {scene_id!r} has {len(tile_indices)} tiles; "
                    f"canonical grid requires {len(expected_windows)}"
                )

            for tile_index in tile_indices:
                metadata = dataset.samples[tile_index]
                item = dataset[tile_index]
                batch = runtime["LEVIRCDDataset"].collate([item])
                batch = runtime["resize_canonical_batch"](batch, CANONICAL_TILE_SIZE)
                batch = runtime["move_batch_to_device"](batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    output = model.infer(batch, mask_text=False, mask_image=False)

                probability = _as_2d_probability(
                    output["change_refined_map_up"], name="change_refined_map_up"
                )
                if probability.shape != (CANONICAL_TILE_SIZE, CANONICAL_TILE_SIZE):
                    raise FullSceneProtocolError(
                        f"model probability shape {probability.shape} is not 256x256"
                    )
                center_probability = None
                offset = None
                if args.stitch_instances:
                    center_logits = output.get("change_instance_center_logits")
                    offset_tensor = output.get("change_instance_offset")
                    if center_logits is None or offset_tensor is None:
                        raise FullSceneProtocolError(
                            "--stitch-instances requires model center-logit and offset outputs"
                        )
                    center_probability = _as_2d_probability(
                        torch.sigmoid(center_logits),
                        name="change_instance_center_probability",
                    )
                    offset = _as_offset(
                        offset_tensor,
                        expected_shape=(CANONICAL_TILE_SIZE, CANONICAL_TILE_SIZE),
                    )
                accumulator.add_tile(
                    x=int(metadata["x"]),
                    y=int(metadata["y"]),
                    probability=probability,
                    center_probability=center_probability,
                    offset=offset,
                )

            stitched = accumulator.finalize()
            with Image.open(first_meta["path_l"]) as label_image:
                ground_truth = (
                    np.asarray(label_image.convert("L"), dtype=np.uint8)
                    > int(cfg.get("levir_mask_threshold", 127))
                ).astype(np.uint8)
            if ground_truth.shape != (source_height, source_width):
                raise FullSceneProtocolError(
                    f"source {scene_id!r} GT shape {ground_truth.shape} != "
                    f"{(source_height, source_width)}"
                )
            archive_path = output_dir / _safe_scene_filename(scene_id)
            write_stitched_scene_archive(
                archive_path,
                scene_id=scene_id,
                split=args.split,
                stitched=stitched,
                ground_truth=ground_truth,
            )
            archive_rows.append(
                {
                    "scene_id": scene_id,
                    "archive": archive_path.name,
                    "source_height": source_height,
                    "source_width": source_width,
                    "tile_count": accumulator.received_tile_count,
                    "coverage_min": int(stitched.coverage_count.min()),
                    "coverage_max": int(stitched.coverage_count.max()),
                    "weight_min": float(stitched.weight_sum.min()),
                    "weight_max": float(stitched.weight_sum.max()),
                }
            )

    elapsed = float(time.perf_counter() - started)
    manifest = {
        "schema": "ieft.full_scene_prediction_manifest.v1",
        "protocol": {
            "name": PROTOCOL_NAME,
            "tile_size": CANONICAL_TILE_SIZE,
            "tile_stride": CANONICAL_TILE_STRIDE,
            "tile_batch_size": 1,
            "blend": "modified_hann",
            "hann_floor": 0.20,
            "threshold_applied": False,
            "semantic_postprocessing": "none",
        },
        "split": args.split,
        "dataset_id": args.dataset_id,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_parameter_coverage": float(checkpoint_report["parameter_coverage"]),
        "resolved_offline_assets": resolved_assets,
        "stitches_instance_outputs": bool(args.stitch_instances),
        "source_scene_count": len(archive_rows),
        "tile_count": int(sum(row["tile_count"] for row in archive_rows)),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "dataset_filtering_stats": dataset.filtering_stats,
        "archives": archive_rows,
    }
    _atomic_json_write(output_dir / "prediction_manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_prediction(args)
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "split": manifest["split"],
                "source_scene_count": manifest["source_scene_count"],
                "tile_count": manifest["tile_count"],
                "threshold_applied": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
