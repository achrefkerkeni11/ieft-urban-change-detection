# IEFT for LEVIR-CD urban change detection

This is the maintained, offline-at-runtime implementation that produced the
final 118,000-step LEVIR-CD result. It combines paired RGB imagery with cached
historical T2 OpenStreetMap context, cached bitemporal spectral indices, and an
auxiliary instance-separation head. Training and inference never call a remote
service; network access is confined to the explicit data-preparation tools.

Run every command in this document from this directory (`IEFT/IEFT`).

## Final result

The final checkpoint was trained from random initialization for 118,000
optimizer steps. The decision threshold was selected once on all 64 validation
scenes and then frozen for the 128 test scenes.

| Split | Scenes | Threshold | IoU | F1 | Precision | Recall | OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VAL | 64 | 0.747 | 73.6069% | 84.7972% | 83.4283% | 86.2118% | 98.7027% |
| TEST | 128 | 0.747 | **73.8947%** | **84.9879%** | 84.2069% | 85.7835% | 98.4562% |

The machine-readable result, confusion matrix, protocol, and artifact hashes
are stored in [`docs/final_result.json`](docs/final_result.json).

The exact checkpoint is intentionally not committed because it is 1.56 GB.
Its identity is:

```text
size:   1,564,322,778 bytes
sha256: 00836fe490c1e9e291bc477414d4d3e8314281186c438cb8c086302371c2e11d
```

Place a copy anywhere outside Git's tracked source tree and pass its path to
the inference commands below. A newly trained checkpoint is written under
`result/<experiment>_seed0_from_none/version_0/checkpoints/last.ckpt`.

## Maintained files

- `run_final_stable_118k_auto.py`: recommended end-to-end training launcher;
- `run_final_stable_118k.py`: training implementation used by the launcher;
- `predict_full_scenes.py`: canonical unthresholded full-scene inference;
- `evaluate_full_scenes.py`: VAL threshold selection and frozen TEST metrics;
- `export_change_maps.py`: visualization-only export verified against the
  official TEST confusion matrix;
- `prepare_levir_temporal_osm.py`: historical T2 OSM cache preparation;
- `prepare_levir_spectral_indices.py`: physical spectral cache preparation;
- `prepare_legacy_spectral_train_stats.py`: TRAIN-only normalization for the
  legacy indices cache used by the reported run;
- `IEFT/`: model, data pipeline, full-scene reconstruction, and post-processing;
- `tests/`: lightweight regression tests plus an opt-in final-checkpoint test.

Old milestone runners, debug scripts, duplicated configurations, targeted-veto
post-processing, and intermediate stage notes were removed. They are not part
of the reported result.

## Installation

Create an isolated Python environment and install a PyTorch build suitable for
your CPU/CUDA platform, then install the project requirements:

```powershell
python -m pip install -r requirements-levir.txt
```

For development and tests:

```powershell
python -m pip install -r requirements-dev.txt
```

## Required local data

LEVIR-CD, generated caches, checkpoints, and outputs are not stored in Git.
The expected layout is:

```text
data_levir_cd/
  raw/
    LEVIR CD/
      train/{A,B,label}/
      val/{A,B,label}/
      test/{A,B,label}/
data_osm_t2_v2/
  manifest.json
  ... cached GeoJSON files ...
data_spectral_v2/
  manifest.json
  train_sensor_stats.json
  ... cached spectral files ...
```

The reported run used all 445 TRAIN, 64 VAL, and 128 TEST source pairs. Its
auxiliary manifests have these hashes:

```text
OSM manifest:      7a2345439a64e438efc50e178e156142702b57b298b63a37594536c23911616b
spectral manifest: 2311e86a36f09c80348d632af09fd64b28314996edee612a784824e09b4b3841
```

Different source snapshots can be valid inputs, but they are not bit-for-bit
the data used for the metrics above.

## Prepare the auxiliary caches

The OSM preparer downloads only historical T2 context through the ohsome API,
records provenance, and supports resume:

```powershell
python prepare_levir_temporal_osm.py preflight --coords-json LEVIR_CD_name_coords.json --data-root "data_levir_cd/raw/LEVIR CD" --temporal-key t2 --output-root data_osm_t2_v2 --manifest-name manifest.json
python prepare_levir_temporal_osm.py generate --coords-json LEVIR_CD_name_coords.json --data-root "data_levir_cd/raw/LEVIR CD" --temporal-key t2 --output-root data_osm_t2_v2 --manifest-name manifest.json
python prepare_levir_temporal_osm.py validate --manifest data_osm_t2_v2/manifest.json --require-complete
```

The spectral preparer writes Green/Red/NIR surface reflectance, NDVI,
McFeeters NDWI, validity masks, and provenance. Earth Engine requires a valid
project and prior authentication:

```powershell
python prepare_levir_spectral_indices.py preflight --coords-json LEVIR_CD_name_coords.json --data-root "data_levir_cd/raw/LEVIR CD" --provider earth_engine_landsat --ee-project <PROJECT_ID> --output-root data_spectral_v2
python prepare_levir_spectral_indices.py generate --coords-json LEVIR_CD_name_coords.json --data-root "data_levir_cd/raw/LEVIR CD" --provider earth_engine_landsat --ee-project <PROJECT_ID> --output-root data_spectral_v2
python prepare_levir_spectral_indices.py validate --manifest data_spectral_v2/manifest.json --require-complete
python prepare_levir_spectral_indices.py stats --manifest data_spectral_v2/manifest.json --output data_spectral_v2/train_sensor_stats.json
```

If you already have the legacy indices-only cache used by the reported run,
keep its `manifest.json` and generate normalization statistics without touching
VAL or TEST pixels:

```powershell
python prepare_legacy_spectral_train_stats.py --manifest data_spectral_v2/manifest.json --output data_spectral_v2/train_sensor_stats.json --project-root .
```

## Train the final model

The recommended launcher performs strict data, disk, CUDA, and source checks;
starts one scratch run; writes full-state recovery checkpoints at 40k and 80k;
and writes the final checkpoint at 118k. Automatic recovery is limited to
verified native Windows/CUDA crashes.

```powershell
python run_final_stable_118k_auto.py --run-tag my_final_run
```

The exact configuration composition used by that launcher is:

```text
task_levir_cd_v20_cliprank_vitb16_levir_dense_completion
levir_scratch_full_base
levir_scratch_rgb
levir_final_rgb_osm_instance_safe_spectral
```

Important fixed settings include batch size 1, mixed precision, 500 training
batches per short epoch, no in-training validation, and a single 118k learning
rate schedule. Do not tune against the TEST split.

## Reproduce full-scene evaluation

Set the three local paths first:

```powershell
$checkpoint = "path/to/final_118k.ckpt"
$dataRoot = "data_levir_cd/raw/LEVIR CD"
$datasetId = "LEVIR-CD-official-637"
```

1. Export unthresholded validation probabilities and select the threshold:

```powershell
python predict_full_scenes.py --ckpt-path $checkpoint --data-root $dataRoot --split val --output-dir result/final_118k_val --dataset-id $datasetId --levir-temporal-osm-manifest data_osm_t2_v2/manifest.json --levir-temporal-osm-enabled --levir-temporal-osm-mode t2 --no-levir-require-osm-t1 --no-levir-require-osm-t2 --no-levir-filter-failed-osm --levir-spectral-manifest data_spectral_v2/manifest.json --levir-spectral-indices-enabled --no-levir-require-indices-t1 --no-levir-require-indices-t2 --levir-accept-partial-indices --levir-min-index-valid-fraction 0 --levir-spectral-missing-policy mask --levir-index-normalization sensor_train_stats --levir-index-normalization-stats data_spectral_v2/train_sensor_stats.json

python evaluate_full_scenes.py select-val --input-dir result/final_118k_val --checkpoint $checkpoint --dataset-id $datasetId --threshold-artifact result/final_118k_threshold.json --report result/final_118k_val_report.json
```

2. Export TEST probabilities and evaluate only with the frozen VAL artifact:

```powershell
python predict_full_scenes.py --ckpt-path $checkpoint --data-root $dataRoot --split test --output-dir result/final_118k_test --dataset-id $datasetId --stitch-instances --levir-temporal-osm-manifest data_osm_t2_v2/manifest.json --levir-temporal-osm-enabled --levir-temporal-osm-mode t2 --no-levir-require-osm-t1 --no-levir-require-osm-t2 --no-levir-filter-failed-osm --levir-spectral-manifest data_spectral_v2/manifest.json --levir-spectral-indices-enabled --no-levir-require-indices-t1 --no-levir-require-indices-t2 --levir-accept-partial-indices --levir-min-index-valid-fraction 0 --levir-spectral-missing-policy mask --levir-index-normalization sensor_train_stats --levir-index-normalization-stats data_spectral_v2/train_sensor_stats.json

python evaluate_full_scenes.py evaluate-test --input-dir result/final_118k_test --checkpoint $checkpoint --dataset-id $datasetId --threshold-artifact result/final_118k_threshold.json --report result/final_118k_test_report.json --instance-output-dir result/final_118k_test_instances --instance-center-threshold 0.3 --instance-min-distance 4
```

The canonical protocol reconstructs each 1024x1024 scene from 256x256 tiles at
stride 128 with a modified Hann window (`hann_floor=0.2`). Thresholding occurs
only after reconstruction; no component filter or veto modifies the semantic
mask.

## Export verified visualizations

This step performs no inference and no threshold search. It refuses to export
unless the selected probability array reproduces the official TEST confusion
matrix exactly:

```powershell
python export_change_maps.py --prediction-dir result/final_118k_test --data-root $dataRoot --report result/final_118k_test_report.json --output-dir result/final_118k_visuals --top-k 20
```

## Verification

Run the lightweight suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -B -m pytest tests -q
```

The final-checkpoint integrity/compatibility test is skipped by default because
it hashes and loads the 1.56 GB artifact:

```powershell
$env:IEFT_RUN_FINAL_CHECKPOINT_TEST = "1"
$env:IEFT_FINAL_CHECKPOINT = $checkpoint
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -B -m pytest tests/test_final_checkpoint.py -q -s
```

## Artifact policy

Git contains the code, coordinate metadata, requirements, tests, and the exact
result summary. Datasets, API caches, checkpoints, tensorboard files,
predictions, and visual exports remain local and are ignored. This keeps the
repository transferable while preserving exact hashes for artifact checking.
