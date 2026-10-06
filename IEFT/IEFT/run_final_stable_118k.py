"""Exact training runner used for the final 118,000-step IEFT model."""

import copy
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import pytorch_lightning as pl
import torch

from IEFT.config import ex
from IEFT.datamodules.multitask_datamodule import MTDataModule
from IEFT.modules import ViLTransformerSS, objectives


# Final stable scientific schedule. A normal run is ONE uninterrupted fit.
# Automatic recovery is allowed only after a real process failure.
FINAL_SCHEDULE_TOTAL_STEPS = 118000


def _as_int_num_gpus(num_gpus_cfg):
    if num_gpus_cfg is None:
        return 0
    if isinstance(num_gpus_cfg, int):
        return num_gpus_cfg
    if isinstance(num_gpus_cfg, (list, tuple)):
        return len(num_gpus_cfg)
    try:
        return int(num_gpus_cfg)
    except Exception:
        return 0


def _callable_parameters(callable_obj):
    try:
        return inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return {}


def _supports_parameter(callable_obj, name):
    parameters = _callable_parameters(callable_obj)
    return name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _filter_supported_kwargs(callable_obj, kwargs):
    parameters = _callable_parameters(callable_obj)
    if not parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _atomic_write_json(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hashes_from_config(config):
    hashes = {}
    for key, value in sorted(config.items()):
        if "manifest" not in str(key).lower() or not isinstance(value, (str, os.PathLike)):
            continue
        raw_path = os.fspath(value).strip()
        if not raw_path:
            continue
        resolved = os.path.abspath(raw_path)
        record = {"configured_path": raw_path, "resolved_path": resolved}
        if os.path.isfile(resolved):
            record["sha256"] = _sha256_file(resolved)
        else:
            record["sha256"] = None
            record["status"] = "not_found"
        hashes[str(key)] = record
    return hashes


def _iter_leaf_datamodules(datamodule):
    children = getattr(datamodule, "dms", None)
    if isinstance(children, (list, tuple)) and children:
        for child in children:
            yield from _iter_leaf_datamodules(child)
        return
    yield datamodule


def _collect_data_provenance(datamodule, config):
    datasets = {}
    seen = set()
    for leaf in _iter_leaf_datamodules(datamodule):
        for attribute in ("train_dataset", "val_dataset", "test_dataset"):
            dataset = getattr(leaf, attribute, None)
            if dataset is None or id(dataset) in seen:
                continue
            seen.add(id(dataset))
            split = str(getattr(dataset, "split", attribute.replace("_dataset", "")))
            record = {}
            filtering_stats = getattr(dataset, "filtering_stats", None)
            if filtering_stats is not None:
                record["filtering_stats"] = filtering_stats
            retained = getattr(dataset, "retained_sources", None)
            if retained is not None:
                record["retained_source_ids"] = list(retained)
            samples = getattr(dataset, "samples", None)
            if isinstance(samples, list):
                record["tile_count"] = len(samples)
                record["tile_ids"] = [
                    str(sample.get("patch_id", sample.get("tile_id", "")))
                    for sample in samples
                    if isinstance(sample, dict)
                ]
            datasets[split] = record

        auxiliary_stats = getattr(leaf, "auxiliary_filtering_stats", None)
        if isinstance(auxiliary_stats, dict):
            for split, stats in auxiliary_stats.items():
                datasets.setdefault(str(split), {}).setdefault("filtering_stats", stats)

    return {
        "datasets": datasets,
        "configured_manifest_hashes": _manifest_hashes_from_config(config),
        "evaluation_protocol": {
            "fixed_threshold": float(config.get("change_eval_threshold", 0.5)),
            "threshold_policy": (
                "fixed configuration value; select on retained validation data only and "
                "reuse unchanged for final test"
            ),
            "eval_split": str(config.get("eval_split", "test")),
            "random_seed": int(config.get("seed", 0)),
        },
    }


def _save_run_provenance(datamodule, config, log_dir):
    config_path = os.path.join(log_dir, "config_snapshot.json")
    provenance_path = os.path.join(log_dir, "data_provenance.json")
    _atomic_write_json(config_path, dict(config))
    _atomic_write_json(provenance_path, _collect_data_provenance(datamodule, config))
    print(f"[INFO] Config snapshot saved to: {config_path}")
    print(f"[INFO] Data provenance saved to: {provenance_path}")


def _free_bytes(path):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(path, exist_ok=True)
    return int(shutil.disk_usage(path).free)


def _remove_partial_checkpoint(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _save_full_state_checkpoint(
    trainer,
    path,
    *,
    stage,
    estimated_checkpoint_bytes=4 * 1024 ** 3,
    reserve_bytes=2 * 1024 ** 3,
):
    """Save one full Trainer-state checkpoint with an explicit disk guard.

    The checkpoint contains model weights, optimizer state, LR scheduler state,
    AMP/scaler state when provided by Lightning, epoch and global_step.
    """

    path = os.path.abspath(os.fspath(path))
    checkpoint_dir = os.path.dirname(path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    estimated_checkpoint_bytes = max(1, int(estimated_checkpoint_bytes))
    reserve_bytes = max(0, int(reserve_bytes))
    free = _free_bytes(checkpoint_dir)
    required = estimated_checkpoint_bytes + reserve_bytes

    print(
        f"[CHECKPOINT] {stage}: free={free / 1024**3:.2f} GB | "
        f"required_guard={required / 1024**3:.2f} GB | path={path}"
    )

    if free < required:
        raise RuntimeError(
            f"Refusing checkpoint save at {stage}: only {free / 1024**3:.2f} GB "
            f"free, while the safety guard requires {required / 1024**3:.2f} GB. "
            "Free disk space before continuing; no partial checkpoint was written."
        )

    _remove_partial_checkpoint(path)
    try:
        trainer.save_checkpoint(path, weights_only=False)
    except OSError as exc:
        _remove_partial_checkpoint(path)
        if getattr(exc, "errno", None) == 28:
            raise RuntimeError(
                f"Disk became full while saving the full-state checkpoint at {stage}. "
                "The partial checkpoint was removed."
            ) from exc
        raise

    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise RuntimeError(
            f"Checkpoint save reported success but the file is missing/empty: {path}"
        )

    actual_size = int(os.path.getsize(path))
    print(
        f"[CHECKPOINT] FULL-STATE saved at {stage}: "
        f"{actual_size / 1024**3:.2f} GB | {path}"
    )
    return actual_size


def _save_stable_last_checkpoint(trainer, checkpoint_dir):
    """Save the final 118k checkpoint.

    A full-state checkpoint is attempted first.  If disk space is unexpectedly
    exhausted after training has already completed, a weights-only fallback is
    written so the final trained model itself is not lost.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    stable_last = os.path.join(checkpoint_dir, "last.ckpt")

    try:
        _save_full_state_checkpoint(
            trainer,
            stable_last,
            stage="final global_step=118000",
        )
        return stable_last
    except RuntimeError as exc:
        message = str(exc).lower()
        if "disk" not in message and "free" not in message:
            raise

        print(
            "[CHECKPOINT][WARN] Full-state final checkpoint could not be stored "
            "because of disk space. Saving WEIGHTS-ONLY last.ckpt as an emergency "
            "fallback so the final model parameters are preserved."
        )
        _remove_partial_checkpoint(stable_last)
        trainer.save_checkpoint(stable_last, weights_only=True)
        if not os.path.isfile(stable_last) or os.path.getsize(stable_last) <= 0:
            raise RuntimeError(
                "Emergency weights-only final checkpoint also failed."
            ) from exc
        print(
            "[CHECKPOINT][WARN] Final last.ckpt is WEIGHTS-ONLY. "
            "It is valid for inference/evaluation but not optimizer-state resume."
        )
        return stable_last


class PeriodicAuxCheckpointCallback(pl.Callback):
    """Save deterministic auxiliary-training checkpoints at fixed optimizer steps.

    This callback is intentionally independent of the validation metric.  The
    scientific S1/S2 protocol selects candidates later with canonical
    full-scene VAL reconstruction, so intermediate optimizer states must not be
    discarded just because patch-level ``val/loss`` is not the best one.
    """

    def __init__(self, checkpoint_dir, every_n_steps=200):
        super().__init__()
        self.checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))
        self.every_n_steps = max(1, int(every_n_steps))
        self._last_saved_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0 or step % self.every_n_steps != 0:
            return
        if step == self._last_saved_step:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"step_{step:06d}.ckpt")
        trainer.save_checkpoint(path)
        self._last_saved_step = step
        print(f"[INFO] Periodic auxiliary checkpoint saved: {path}")


class ResumeStatusCallback(pl.Callback):
    """Print the authoritative Trainer state at the beginning of fit."""

    def on_train_start(self, trainer, pl_module):
        step = int(getattr(trainer, "global_step", 0))
        epoch = int(getattr(trainer, "current_epoch", 0))
        print(
            "[TRAIN_STATE] "
            f"global_step={step} | current_epoch={epoch} | "
            "tqdm's x/56960 value is only the dataloader batch counter."
        )


class CudaMemoryMaintenanceCallback(pl.Callback):
    """CUDA cleanup/telemetry for the RTX 2050 4 GB training protocol.

    The previous failing run showed live allocated CUDA memory growing almost
    linearly by ~0.26 GiB / 1000 steps.  This protocol therefore:
      * uses 500-batch epochs,
      * disables Lightning train metric logging,
      * runs gc.collect()+empty_cache every 500 optimizer steps,
      * measures memory at every new epoch,
      * aborts CLEANLY before a catastrophic CUDA/cuBLAS failure if live
        allocated memory after cleanup exceeds the safety ceiling.

    The safety check is performed after a completed optimizer step and does not
    modify gradients, weights, the LR schedule, or the 118k step budget.
    """

    def __init__(
        self,
        every_n_steps=500,
        live_allocated_abort_gib=2.60,
    ):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.live_allocated_abort_gib = float(live_allocated_abort_gib)
        self._last_periodic_step = -1
        self._last_epoch_start_step = None

    @staticmethod
    def _gib(value):
        return float(value) / float(1024 ** 3)

    @staticmethod
    def _current_lrs(trainer):
        values = []
        for optimizer in list(getattr(trainer, "optimizers", []) or []):
            for group in optimizer.param_groups:
                try:
                    values.append(float(group.get("lr", 0.0)))
                except Exception:
                    pass
        return values

    def _cleanup_measure(self, trainer, step, prefix, enforce_guard):
        if not torch.cuda.is_available():
            return

        gc.collect()
        torch.cuda.empty_cache()

        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        allocated_gib = self._gib(allocated)
        lrs = self._current_lrs(trainer)

        print(
            f"[CUDA_MEM] {prefix} step={step} | "
            f"allocated={allocated_gib:.2f} GB | "
            f"reserved={self._gib(reserved):.2f} GB | "
            f"peak={self._gib(peak):.2f} GB | "
            f"free={self._gib(free_bytes):.2f}/{self._gib(total_bytes):.2f} GB | "
            f"lr={lrs}"
        )

        torch.cuda.reset_peak_memory_stats(device)

        if enforce_guard and allocated_gib > self.live_allocated_abort_gib:
            raise RuntimeError(
                "[CUDA_MEM_GUARD] Live CUDA allocation is still growing after "
                f"cleanup: {allocated_gib:.2f} GiB > "
                f"{self.live_allocated_abort_gib:.2f} GiB safety ceiling. "
                "Training is stopped cleanly before another CUDA/cuBLAS OOM. "
                "Do not automatically restart this configuration."
            )

    def on_train_start(self, trainer, pl_module):
        self._cleanup_measure(
            trainer,
            int(getattr(trainer, "global_step", 0)),
            "train_start",
            enforce_guard=False,
        )

    def on_train_epoch_start(self, trainer, pl_module):
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        step = int(getattr(trainer, "global_step", 0))
        # Avoid a duplicate train_start report at step zero.
        if step == 0:
            return
        if step == self._last_epoch_start_step:
            return

        self._cleanup_measure(
            trainer,
            step,
            "epoch_start",
            enforce_guard=True,
        )
        self._last_epoch_start_step = step

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        *args,
        **kwargs,
    ):
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0 or step % self.every_n_steps != 0:
            return
        if step == self._last_periodic_step:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        self._cleanup_measure(
            trainer,
            step,
            "periodic",
            enforce_guard=True,
        )
        self._last_periodic_step = step


class FullStateMilestoneCheckpointCallback(pl.Callback):
    """Save full Trainer-state checkpoints at 40k and 80k without stopping fit.

    There is NO checkpoint at 8k, 20k, 60k or 100k.  The two milestone files
    are directly resumable because weights_only=False is used.
    """

    def __init__(self, checkpoint_dir, milestones=(40000, 80000)):
        super().__init__()
        self.checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))
        self.milestones = tuple(
            sorted({int(step) for step in milestones if int(step) > 0})
        )
        self._saved = set()
        self._last_checkpoint_size = 4 * 1024 ** 3

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        *args,
        **kwargs,
    ):
        step = int(getattr(trainer, "global_step", 0))
        if step not in self.milestones or step in self._saved:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return

        path = os.path.join(
            self.checkpoint_dir,
            f"milestone_{step}.ckpt",
        )
        actual_size = _save_full_state_checkpoint(
            trainer,
            path,
            stage=f"milestone global_step={step}",
            estimated_checkpoint_bytes=self._last_checkpoint_size,
        )
        self._last_checkpoint_size = max(
            int(actual_size),
            int(self._last_checkpoint_size),
        )
        self._saved.add(step)

        _atomic_write_json(
            os.path.join(self.checkpoint_dir, f"milestone_{step}.json"),
            {
                "global_step": step,
                "checkpoint": os.path.abspath(path),
                "weights_only": False,
                "kind": "full_trainer_state",
                "resumable": True,
                "size_bytes": int(actual_size),
            },
        )
        print(
            f"[MILESTONE] global_step={step} FULL-STATE checkpoint complete. "
            "Trainer.fit continues without validation or restart."
        )


class ChangeMetricsCallback(pl.Callback):
    """Aggregate dense validation/test metrics from global confusion counts."""

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = float(threshold)
        # Validate once during startup rather than on the first evaluation batch.
        objectives.BinaryChangeMetricAccumulator(self.threshold)

    def _start(self, pl_module, stage):
        pl_module._change_metric_accumulator = objectives.BinaryChangeMetricAccumulator(
            self.threshold
        )
        pl_module._change_metric_stage = stage

    def _finish(self, trainer, pl_module, stage):
        accumulator = getattr(pl_module, "_change_metric_accumulator", None)
        active_stage = getattr(pl_module, "_change_metric_stage", None)
        if accumulator is None or active_stage != stage:
            return
        metrics = accumulator.compute(synchronize=True)
        scalar_metrics = {
            f"{stage}/{name}": float(metrics[name].detach().cpu().item())
            for name in objectives.BinaryChangeMetricAccumulator.METRIC_NAMES
        }
        scalar_metrics.update(
            {
                f"{stage}/{name}": int(metrics[name].detach().cpu().item())
                for name in ("tp", "fp", "fn", "tn")
            }
        )
        scalar_metrics[f"{stage}/threshold"] = self.threshold
        pl_module._last_change_metrics = getattr(pl_module, "_last_change_metrics", {})
        pl_module._last_change_metrics[stage] = scalar_metrics

        is_global_zero = bool(getattr(trainer, "is_global_zero", True))
        logger = getattr(trainer, "logger", None)
        if is_global_zero and logger is not None and hasattr(logger, "log_metrics"):
            logger.log_metrics(scalar_metrics, step=int(getattr(trainer, "global_step", 0)))
        if is_global_zero:
            summary = " | ".join(
                f"{name}={scalar_metrics[f'{stage}/{name}']:.6f}"
                for name in objectives.BinaryChangeMetricAccumulator.METRIC_NAMES
            )
            print(f"[METRICS] {stage} threshold={self.threshold:.6f} | {summary}")

        pl_module._change_metric_stage = None
        pl_module._change_metric_accumulator = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._start(pl_module, "val")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._finish(trainer, pl_module, "val")

    def on_test_epoch_start(self, trainer, pl_module):
        self._start(pl_module, "test")

    def on_test_epoch_end(self, trainer, pl_module):
        self._finish(trainer, pl_module, "test")


def _build_model_checkpoint(checkpoint_dir):
    checkpoint_cls = pl.callbacks.ModelCheckpoint
    parameters = _callable_parameters(checkpoint_cls.__init__)
    kwargs = {
        "monitor": "val/loss",
        "mode": "min",
        "save_top_k": 1,
        "save_last": False,
        "verbose": True,
    }
    if "dirpath" in parameters:
        kwargs["dirpath"] = checkpoint_dir
        kwargs["filename"] = "best-{epoch:02d}"
    elif "filepath" in parameters:
        kwargs["filepath"] = os.path.join(checkpoint_dir, "best-{epoch:02d}")
    if "every_n_epochs" in parameters:
        kwargs["every_n_epochs"] = 1
    elif "period" in parameters:
        kwargs["period"] = 1
    return checkpoint_cls(**_filter_supported_kwargs(checkpoint_cls.__init__, kwargs))


def _normalize_max_steps(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "auto"}:
            return None
        value = int(value)
    value = int(value)
    return value if value > 0 else None


def _estimate_training_steps(datamodule, max_epoch, grad_steps):
    train_loader = datamodule.train_dataloader()
    batches_per_epoch = len(train_loader)
    if batches_per_epoch <= 0:
        raise RuntimeError("The training dataloader is empty after auxiliary filtering")
    return int(math.ceil((batches_per_epoch * int(max_epoch)) / max(1, int(grad_steps))))


def _build_trainer_kwargs(
    config,
    logger,
    callbacks,
    checkpoint_callback,
    grad_steps,
    max_steps,
    use_ddp,
):
    trainer_init = pl.Trainer.__init__
    parameters = _callable_parameters(trainer_init)
    num_gpus_int = _as_int_num_gpus(config.get("num_gpus", 0))
    max_epochs = int(config.get("max_epoch", 1)) if max_steps is None else 1000
    scratch_training = bool(config.get("scratch_training", False))

    kwargs = {
        "num_nodes": int(config.get("num_nodes", 1)),
        "precision": config.get("precision", 32),
        "benchmark": False,
        "deterministic": True,
        "max_epochs": max_epochs,
        "callbacks": list(callbacks),
        "logger": logger,
        "prepare_data_per_node": False,
        "replace_sampler_ddp": False,
        "use_distributed_sampler": False,
        "accumulate_grad_batches": int(grad_steps),
        "log_every_n_steps": 10,
        "flush_logs_every_n_steps": 10,
        "weights_summary": "top",
        "enable_model_summary": True,
        "fast_dev_run": bool(config.get("fast_dev_run", False)),
    }

    # Scratch runs requested by the user are deliberately one uninterrupted
    # training phase: no sanity validation, no validation during fit, and no
    # validation-driven checkpointing.  Validation is performed later in a
    # separate process with the canonical full-scene evaluator.
    if scratch_training:
        kwargs["num_sanity_val_steps"] = 0
        kwargs["limit_val_batches"] = 0.0

        # Hard-coded on purpose: this is NOT a Sacred configuration entry.
        # Using an unknown command-line config key caused ConfigAddedError in
        # Sacred.  500 batches/epoch is only a memory-lifetime boundary; the
        # scientific optimization budget still ends at global_step=118000.
        short_epoch_batches = 500
        kwargs["limit_train_batches"] = short_epoch_batches
    else:
        kwargs["val_check_interval"] = config.get("val_check_interval", 1.0)

    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)

    if "gpus" in parameters:
        kwargs["gpus"] = config.get("num_gpus", 0)
        if use_ddp:
            kwargs["accelerator"] = "ddp"
    else:
        if num_gpus_int > 0:
            kwargs["accelerator"] = "gpu"
            kwargs["devices"] = config.get("num_gpus", num_gpus_int)
        else:
            kwargs["accelerator"] = "cpu"
            kwargs["devices"] = 1
        if use_ddp:
            kwargs["strategy"] = "ddp"

    # Old Lightning versions expose checkpoint_callback; newer versions expose
    # enable_checkpointing/callback lists.  For scratch single-phase training,
    # explicitly disable Lightning's automatic checkpoint callback so the only
    # model file is the manually saved final last.ckpt.
    if scratch_training:
        if "checkpoint_callback" in parameters:
            kwargs["checkpoint_callback"] = False
        if "enable_checkpointing" in parameters:
            kwargs["enable_checkpointing"] = False
    else:
        if "checkpoint_callback" in parameters:
            kwargs["checkpoint_callback"] = checkpoint_callback
        elif checkpoint_callback is not None:
            kwargs["callbacks"].append(checkpoint_callback)

    resume_path = config.get("resume_from", None)
    resume_via_method = False
    if resume_path:
        if "resume_from_checkpoint" in parameters:
            kwargs["resume_from_checkpoint"] = resume_path
        else:
            resume_via_method = True

    return _filter_supported_kwargs(trainer_init, kwargs), resume_via_method


def _invoke_trainer_method(method, model, datamodule, checkpoint_path=None):
    kwargs = {"datamodule": datamodule}
    if checkpoint_path and _supports_parameter(method, "ckpt_path"):
        kwargs["ckpt_path"] = checkpoint_path
    return method(model, **_filter_supported_kwargs(method, kwargs))


def _load_warm_start(model, load_path, log_dir, config):
    if not load_path:
        return None
    load_path = os.path.abspath(os.fspath(load_path))
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {load_path}")
    if not hasattr(model, "load_compatible_checkpoint"):
        raise AttributeError("Model does not expose load_compatible_checkpoint")
    report = model.load_compatible_checkpoint(
        load_path,
        map_location="cpu",
        minimum_parameter_coverage=float(
            config.get("checkpoint_min_parameter_coverage", 0.75)
        ),
        strict_compatibility=bool(config.get("checkpoint_strict_compatibility", True)),
    )
    report_path = os.path.join(log_dir, "checkpoint_load_report.json")
    _atomic_write_json(report_path, report)
    print(
        "[INFO] Warm-start checkpoint loaded: "
        f"coverage={float(report.get('parameter_coverage', 0.0)):.2%}, "
        f"missing={len(report.get('missing_keys', []))}, "
        f"unexpected={len(report.get('unexpected_keys', []))}, "
        f"shape_mismatches={len(report.get('shape_mismatches', {}))}."
    )
    print(f"[INFO] Detailed checkpoint report saved to: {report_path}")
    return report


def _validate_scratch_contract(config):
    """Validate a true scratch start or a FULL-STATE crash continuation.

    ``load_path`` and every pretrained entry point are forbidden. ``resume_from``
    is permitted only for automatic recovery from this runner's own 40k/80k
    full-state milestone checkpoints.
    """

    if not bool(config.get("scratch_training", False)):
        return

    violations = []
    load_path = str(config.get("load_path", "") or "").strip()
    resume_from = str(config.get("resume_from", "") or "").strip()
    encoder_ckpt = str(config.get("vit_encoder_ckpt_path", "") or "").strip()

    if load_path:
        violations.append(f"load_path={load_path!r}")
    if bool(config.get("vit_pretrained", False)):
        violations.append("vit_pretrained=True")
    if encoder_ckpt:
        violations.append(f"vit_encoder_ckpt_path={encoder_ckpt!r}")
    if int(config.get("vit_encoder_freeze_steps", 0) or 0) != 0:
        violations.append(
            f"vit_encoder_freeze_steps={config.get('vit_encoder_freeze_steps')!r}"
        )

    if violations:
        raise RuntimeError(
            "Final scratch-training contract violated; refusing pretrained or "
            "weights-only state: " + ", ".join(violations)
        )

    if resume_from:
        print("[STABLE118K] FULL-STATE crash continuation requested.")
        print("[STABLE118K] Model + optimizer + scheduler + global_step will resume.")
    else:
        print("[STABLE118K] True random-initialization start verified.")
        print("[STABLE118K] No warm-start, resume checkpoint, or pretrained encoder will be loaded.")

    print(f"[STABLE118K] LR schedule horizon fixed at {FINAL_SCHEDULE_TOTAL_STEPS} steps.")
    print("[STABLE118K] In-training validation disabled; canonical full-scene VAL is external.")


def _seed_everything(seed):
    try:
        pl.seed_everything(int(seed), workers=True)
    except TypeError:
        pl.seed_everything(int(seed))


@ex.automain
def main(_config):
    config = copy.deepcopy(dict(_config))
    _seed_everything(config.get("seed", 0))
    _validate_scratch_contract(config)

    load_path = config.get("load_path", "")
    resume_path = config.get("resume_from", None)
    if load_path and resume_path:
        # Historical v20d runs stored both values.  The Trainer resume is the
        # authoritative state (model, optimizer, scheduler and global step);
        # ``load_path`` remains only as lineage metadata and for the historical
        # logger name.  Loading it first would be redundant and misleading.
        print(
            "[INFO] Both load_path and resume_from are set; resume_from takes "
            "precedence and the weights-only warm start is skipped."
        )
    if resume_path and not os.path.isfile(os.path.abspath(os.fspath(resume_path))):
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    eval_split = str(config.get("eval_split", "test")).strip().lower()
    if eval_split not in {"val", "test"}:
        raise ValueError(f"eval_split must be val or test, got {eval_split!r}")
    eval_threshold = float(config.get("change_eval_threshold", 0.5))
    if not 0.0 <= eval_threshold <= 1.0:
        raise ValueError(
            f"change_eval_threshold must be a fixed value in [0,1], got {eval_threshold}. "
            "Select thresholds on validation data only; test-time tuning is not supported."
        )

    num_gpus_int = _as_int_num_gpus(config.get("num_gpus", 0))
    num_nodes = int(config.get("num_nodes", 1))
    use_ddp = (num_gpus_int > 1) or (num_nodes > 1)

    per_gpu = int(config.get("per_gpu_batchsize", 1))
    effective_gpus = max(1, num_gpus_int)
    denominator = max(1, per_gpu * effective_gpus * max(1, num_nodes))
    grad_steps = max(1, int(config.get("batch_size", per_gpu)) // denominator)

    datamodule = MTDataModule(config, dist=use_ddp)
    test_only = bool(config.get("test_only", False))
    setup_stage = "fit" if not test_only or eval_split == "val" else "test"
    datamodule.setup(setup_stage)

    # This continuous runner should be launched with max_steps=118000.
    # The same value is used by Trainer and by the cosine scheduler, so the
    # entire scientific run is one uninterrupted optimization trajectory.
    trainer_stop_step = _normalize_max_steps(config.get("max_steps", None))
    if test_only:
        trainer_stop_step = None
    elif trainer_stop_step is None:
        trainer_stop_step = FINAL_SCHEDULE_TOTAL_STEPS
    if trainer_stop_step is not None and int(trainer_stop_step) > FINAL_SCHEDULE_TOTAL_STEPS:
        raise ValueError(
            f"Requested milestone {trainer_stop_step} exceeds final horizon "
            f"{FINAL_SCHEDULE_TOTAL_STEPS}."
        )
    if trainer_stop_step is not None and int(trainer_stop_step) <= 1:
        config["warmup_steps"] = 0

    config["max_steps"] = FINAL_SCHEDULE_TOTAL_STEPS
    config["final_schedule_total_steps"] = FINAL_SCHEDULE_TOTAL_STEPS
    config["milestone_stop_step"] = trainer_stop_step

    model = ViLTransformerSS(config)
    experiment_name = str(config.get("exp_name", "vilt"))
    os.makedirs(config.get("log_dir", "result"), exist_ok=True)
    source_path = load_path or resume_path or ""
    load_name = os.path.basename(os.fspath(source_path)) if source_path else "none"
    if load_name.endswith(".ckpt"):
        load_name = load_name[:-5]

    logger = pl.loggers.TensorBoardLogger(
        config.get("log_dir", "result"),
        name=f"{experiment_name}_seed{config.get('seed', 0)}_from_{load_name}",
    )
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    _save_run_provenance(datamodule, config, logger.log_dir)
    _load_warm_start(model, load_path if not resume_path else "", logger.log_dir, config)
    if hasattr(model, "assert_training_scope_safe"):
        scope_report = model.assert_training_scope_safe()
        _atomic_write_json(
            os.path.join(logger.log_dir, "training_scope.json"),
            scope_report,
        )
        print(
            "[TRAIN_SCOPE] "
            f"scope={scope_report['scope']} | "
            f"trainable={scope_report['trainable_parameters']:,}/"
            f"{scope_report['total_parameters']:,}"
        )
        for name in scope_report.get("trainable_names", []):
            print(f"[TRAIN_SCOPE]   {name}")

    scratch_training = bool(config.get("scratch_training", False))
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    if scratch_training and not test_only:
        metric_callback = None
        best_checkpoint = None

        checkpoint_every = int(
            config.get("scratch_checkpoint_every_n_steps", 40000)
        )
        if checkpoint_every != 40000:
            raise RuntimeError(
                "Final stable 118k protocol requires "
                "scratch_checkpoint_every_n_steps=40000 exactly; "
                f"got {checkpoint_every}."
            )

        resume_status_callback = ResumeStatusCallback()
        cuda_memory_callback = CudaMemoryMaintenanceCallback(
            every_n_steps=500,
            live_allocated_abort_gib=2.60,
        )
        milestone_callback = FullStateMilestoneCheckpointCallback(
            checkpoint_dir,
            milestones=(40000, 80000),
        )
        callbacks = [
            resume_status_callback,
            cuda_memory_callback,
            milestone_callback,
        ]
        print(
            "[STABLE118K] Manual FULL-STATE checkpoints enabled ONLY at "
            "global_step 40000 and 80000. Final last.ckpt is saved at 118000."
        )
        print(
            "[STABLE118K] No 8k/20k/60k/100k checkpoint and no in-training VAL."
        )
        print(
            "[STABLE118K] CUDA memory maintenance/telemetry every 500 optimizer steps."
        )
        print(
            "[STABLE118K] Memory-safe short epochs: 500 train batches/epoch "
            "(instead of 56960). max_steps and LR schedule remain 118000."
        )
        print(
            "[STABLE118K] Lightning train metric logging and scratch LR monitor are disabled. CUDA live-allocation guard=2.60 GiB after cleanup."
        )
    else:
        metric_callback = ChangeMetricsCallback(threshold=eval_threshold)
        best_checkpoint = None if test_only else _build_model_checkpoint(checkpoint_dir)
        callbacks = [lr_callback, metric_callback]

        if str(config.get("change_train_scope", "legacy")).strip().lower() == "aux_only":
            aux_checkpoint_every = int(
                config.get("change_aux_checkpoint_every_n_steps", 200)
            )
            if aux_checkpoint_every > 0:
                callbacks.append(
                    PeriodicAuxCheckpointCallback(
                        checkpoint_dir,
                        every_n_steps=aux_checkpoint_every,
                    )
                )
                print(
                    "[INFO] Auxiliary periodic checkpoints enabled every "
                    f"{aux_checkpoint_every} optimizer step(s)."
                )
    trainer_kwargs, resume_via_method = _build_trainer_kwargs(
        config,
        logger,
        callbacks,
        best_checkpoint,
        grad_steps,
        trainer_stop_step,
        use_ddp,
    )
    if scratch_training and not test_only:
        print(
            "[STABLE118K] Trainer fit contract: "
            f"stop_at_global_step={trainer_kwargs.get('max_steps')} | "
            f"scheduler_total_steps={config.get('max_steps')} | "
            f"num_sanity_val_steps={trainer_kwargs.get('num_sanity_val_steps')} | "
            f"limit_val_batches={trainer_kwargs.get('limit_val_batches')} | "
            f"limit_train_batches={trainer_kwargs.get('limit_train_batches')} | "
            f"checkpointing={'disabled' if (trainer_kwargs.get('checkpoint_callback') is False or trainer_kwargs.get('enable_checkpointing') is False) else 'no-callback'}"
        )
    trainer = pl.Trainer(**trainer_kwargs)
    method_checkpoint = resume_path if resume_via_method else None

    if not test_only:
        _invoke_trainer_method(
            trainer.fit,
            model,
            datamodule,
            checkpoint_path=method_checkpoint,
        )
        final_global_step = int(getattr(trainer, "global_step", 0))
        print(f"[INFO] Training fit finished at global_step={final_global_step}.")
        if (
            scratch_training
            and trainer_stop_step is not None
            and final_global_step < int(trainer_stop_step)
        ):
            raise RuntimeError(
                f"Final long training stopped early at global_step={final_global_step}; "
                f"expected milestone={int(trainer_stop_step)}. Final checkpoint will not be written."
            )
        _save_stable_last_checkpoint(trainer, checkpoint_dir)
        best_path = getattr(best_checkpoint, "best_model_path", "") if best_checkpoint else ""
        if best_path:
            print(f"[INFO] Best validation checkpoint: {best_path}")
    elif eval_split == "val":
        validate = getattr(trainer, "validate", None)
        if validate is None:
            raise RuntimeError(
                "This PyTorch Lightning version does not expose Trainer.validate; "
                "validation-only evaluation cannot be labeled safely."
            )
        _invoke_trainer_method(
            validate,
            model,
            datamodule,
            checkpoint_path=method_checkpoint,
        )
    else:
        _invoke_trainer_method(
            trainer.test,
            model,
            datamodule,
            checkpoint_path=method_checkpoint,
        )
