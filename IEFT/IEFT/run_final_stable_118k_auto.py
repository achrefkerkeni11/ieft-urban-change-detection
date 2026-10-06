"""Preflight, launch, and safely recover the final 118k training run."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

TOTAL_STEPS = 118000
CHECKPOINT_STEPS = (40000, 80000)
DEFAULT_MIN_FREE_GB = 20.0

TASK_CONFIG = "task_levir_cd_v20_cliprank_vitb16_levir_dense_completion"
SCRATCH_BASE = "levir_scratch_full_base"
RGB_BASE = "levir_scratch_rgb"
FINAL_CONFIG = "levir_final_rgb_osm_instance_safe_spectral"

DATA_ROOT = Path("data_levir_cd") / "raw" / "LEVIR CD"
OSM_MANIFEST = Path("data_osm_t2_v2") / "manifest.json"
SPECTRAL_MANIFEST = Path("data_spectral_v2") / "manifest.json"
SPECTRAL_STATS = Path("data_spectral_v2") / "train_sensor_stats.json"

SAFE_OSM_MODULE = Path("IEFT") / "modules" / "safe_osm_guidance.py"
SAFE_SPECTRAL_MODULE = Path("IEFT") / "modules" / "safe_spectral_residual.py"
VILT_MODULE = Path("IEFT") / "modules" / "vilt_module.py"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def qassign(name: str, value: str) -> str:
    return f'{name}="{str(value).replace(chr(92), "/")}"'


def atomic_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class KeepWindowsAwake:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __enter__(self):
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
                )
                print("[AUTO] Windows system sleep prevention enabled.")
            except Exception as exc:
                print(f"[AUTO][WARN] Could not disable system sleep: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
                print("[AUTO] Windows sleep policy restored.")
            except Exception:
                pass


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def require_free_space(root: Path, minimum_gb: float, stage: str) -> None:
    free = free_gb(root)
    print(f"[PREFLIGHT] Disk at {stage}: {free:.2f} GB free.")
    if free < minimum_gb:
        raise RuntimeError(
            f"Only {free:.2f} GB free at {stage}. "
            f"This stable protocol requires at least {minimum_gb:.2f} GB before launch. "
            "Free disk space first; no training was started."
        )


def load_json_file(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}, got {type(value).__name__}.")
    return value


def assert_amp_safe_sources(root: Path) -> None:
    for relative in (SAFE_OSM_MODULE, SAFE_SPECTRAL_MODULE):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        if "F.binary_cross_entropy(" in text:
            raise RuntimeError(
                f"Unsafe AMP BCE call still present in {path}. "
                "Install the corrected safe auxiliary module before training."
            )
        if "binary_cross_entropy_with_logits" not in text:
            raise RuntimeError(
                f"Expected AMP-safe binary_cross_entropy_with_logits not found in {path}."
            )
    print("[PREFLIGHT] AMP-safe OSM/spectral BCE implementation: OK.")


def assert_memory_safe_training_source(root: Path) -> None:
    path = root / VILT_MODULE
    text = path.read_text(encoding="utf-8")

    required_fragments = [
        'if stage != "train":',
        'return self._shared_eval(batch, "train")',
        'foreach=False',
        'There is no Lightning train',
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise RuntimeError(
            f"Memory-safe vilt_module.py is not installed at {path}. "
            f"Missing markers: {missing}"
        )

    training_step_region = text[
        text.index("    def training_step"):
        text.index("    def validation_step")
    ]
    if re.search(r"^\s*self\.log(?:_dict)?\(", training_step_region, flags=re.MULTILINE):
        raise RuntimeError(
            "training_step still contains active Lightning logging."
        )

    print(
        "[PREFLIGHT] Train Lightning logging disabled + AdamW foreach=False: OK."
    )


def cuda_preflight(python_exe: str, minimum_free_vram_gb: float) -> Dict:
    """Probe CUDA in a disposable child process.

    The automation parent deliberately never imports torch / initializes CUDA.
    This prevents the orchestrator from holding a CUDA context and consuming
    part of the 4 GB GPU while the actual training process is running.
    """

    probe_code = r"""
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA_NOT_AVAILABLE")

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
torch.cuda.empty_cache()
free_bytes, total_bytes = torch.cuda.mem_get_info(device)

print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device_index": int(device),
    "device_name": props.name,
    "device_total_bytes": int(props.total_memory),
    "mem_get_info_free_bytes": int(free_bytes),
    "mem_get_info_total_bytes": int(total_bytes),
}))
"""

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,garbage_collection_threshold:0.8"
    )
    env["CUDA_MODULE_LOADING"] = "LAZY"

    proc = subprocess.run(
        [python_exe, "-c", probe_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "CUDA preflight failed in disposable subprocess.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    json_lines = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().startswith("{") and line.strip().endswith("}")
    ]
    if not json_lines:
        raise RuntimeError(
            "CUDA preflight returned no JSON information.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    info = json.loads(json_lines[-1])
    free_gb_value = float(info["mem_get_info_free_bytes"]) / (1024 ** 3)
    total_gb_value = float(info["mem_get_info_total_bytes"]) / (1024 ** 3)

    print(
        "[PREFLIGHT] CUDA disposable probe: "
        f"torch={info['torch']} | runtime={info['cuda_runtime']} | "
        f"GPU={info['device_name']} | free={free_gb_value:.2f}/{total_gb_value:.2f} GB."
    )

    if free_gb_value < float(minimum_free_vram_gb):
        raise RuntimeError(
            f"Only {free_gb_value:.2f} GB GPU memory is free before training. "
            f"Require at least {minimum_free_vram_gb:.2f} GB on this 4 GB card. "
            "Close other GPU-using applications/processes and relaunch."
        )

    info["free_vram_gb"] = free_gb_value
    info["total_vram_gb"] = total_gb_value
    info["probe_process_exited_before_training"] = True
    return info


def preflight(
    root: Path,
    runner: Path,
    minimum_gb: float,
    python_exe: str,
    minimum_free_vram_gb: float,
) -> Dict:
    required = [
        runner,
        root / DATA_ROOT,
        root / OSM_MANIFEST,
        root / SPECTRAL_MANIFEST,
        root / SPECTRAL_STATS,
        root / SAFE_OSM_MODULE,
        root / SAFE_SPECTRAL_MODULE,
        root / VILT_MODULE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required stable-training paths are missing:\n  - "
            + "\n  - ".join(missing)
        )

    # Validate the structured auxiliary metadata before spending GPU hours.
    osm_manifest = load_json_file(root / OSM_MANIFEST)
    spectral_manifest = load_json_file(root / SPECTRAL_MANIFEST)
    spectral_stats = load_json_file(root / SPECTRAL_STATS)

    assert_amp_safe_sources(root)
    assert_memory_safe_training_source(root)
    require_free_space(root, minimum_gb, "startup")
    cuda_info = cuda_preflight(python_exe, minimum_free_vram_gb)

    return {
        "time": now_iso(),
        "runner": str(runner),
        "data_root": str(root / DATA_ROOT),
        "osm_manifest": str(root / OSM_MANIFEST),
        "spectral_manifest": str(root / SPECTRAL_MANIFEST),
        "spectral_stats": str(root / SPECTRAL_STATS),
        "osm_manifest_keys": sorted(osm_manifest.keys()),
        "spectral_manifest_keys": sorted(spectral_manifest.keys()),
        "spectral_stats_keys": sorted(spectral_stats.keys()),
        "cuda": cuda_info,
        "minimum_free_gb": minimum_gb,
    }


def run_process(
    command: list[str],
    *,
    cwd: Path,
    log_file: Path,
    title: str,
) -> int:
    header = (
        "\n" + "=" * 100 + "\n"
        f"[AUTO] {title}\n"
        f"[AUTO] Time: {now_iso()}\n"
        f"[AUTO] Command:\n  {' '.join(command)}\n"
        + "=" * 100 + "\n"
    )
    print(header, flush=True)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", errors="replace") as log:
        log.write(header)
        log.flush()
        child_env = os.environ.copy()
        child_env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,garbage_collection_threshold:0.8"
        )
        child_env["CUDA_MODULE_LOADING"] = "LAZY"

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        return_code = process.wait()
        footer = (
            f"\n[AUTO] Finished: {title}\n"
            f"[AUTO] Return code: {return_code}\n"
            f"[AUTO] Time: {now_iso()}\n"
        )
        print(footer, flush=True)
        log.write(footer)
        log.flush()
    return return_code


def checkpoint_step_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.startswith("milestone_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except (TypeError, ValueError):
        return None


def latest_full_state_milestone(
    result_root: Path,
    exp_name: str,
) -> Tuple[Optional[int], Optional[Path]]:
    candidates = []
    pattern = f"{exp_name}_seed0_from_*/version_*/checkpoints/milestone_*.ckpt"

    for path in result_root.glob(pattern):
        if not path.is_file():
            continue
        step = checkpoint_step_from_name(path)
        if step not in CHECKPOINT_STEPS:
            continue
        # The stable runner writes a sidecar only after the full-state save
        # has completed successfully.
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            continue
        try:
            metadata = load_json_file(sidecar)
        except Exception:
            continue
        if (
            int(metadata.get("global_step", -1)) != step
            or metadata.get("weights_only") is not False
            or metadata.get("resumable") is not True
        ):
            continue
        candidates.append((step, path))

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda pair: (pair[0], pair[1].stat().st_mtime),
        reverse=True,
    )
    return candidates[0]


def training_command(
    python_exe: str,
    runner: Path,
    exp_name: str,
    resume_from: Optional[Path],
) -> list[str]:
    cmd = [
        python_exe,
        "-u",
        str(runner),
        "with",
        TASK_CONFIG,
        SCRATCH_BASE,
        RGB_BASE,
        FINAL_CONFIG,
        qassign("data_root", str(DATA_ROOT)),
        qassign("exp_name", exp_name),
        "seed=0",
        "batch_size=1",
        "per_gpu_batchsize=1",
        "num_workers=0",
        "num_gpus=1",
        "num_nodes=1",
        "precision=16",
        "warmup_steps=300",
        "max_steps=118000",
        "scratch_checkpoint_every_n_steps=40000",
    ]
    if resume_from is not None:
        cmd.append(qassign("resume_from", str(resume_from)))
    return cmd


def is_native_windows_crash(return_code: int) -> bool:
    # 0xC0000005 access violation was observed previously.
    return return_code in {
        3221225477,  # 0xC0000005
        3221226505,  # 0xC0000409 stack buffer overrun / fast-fail
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stable 118k scratch run: one fit, full-state checkpoints at 40k/80k, "
            "final at 118k, strict preflight, automatic recovery only after real crashes."
        )
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help="Minimum free disk required before training starts.",
    )
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=2.8,
        help=(
            "Minimum GPU memory that must be free before the training child starts. "
            "The CUDA probe runs in a separate process and exits first."
        ),
    )
    parser.add_argument(
        "--max-native-retries",
        type=int,
        default=3,
        help="Maximum automatic retries for native Windows/CUDA process crashes.",
    )
    parser.add_argument("--retry-wait-seconds", type=int, default=60)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    result_root = root / "result"
    runner = root / "run_final_stable_118k.py"
    python_exe = sys.executable

    # IMPORTANT: the automation parent itself never initializes CUDA.
    preflight_report = preflight(
        root,
        runner,
        float(args.min_free_gb),
        python_exe,
        float(args.min_free_vram_gb),
    )

    tag = args.run_tag.strip() or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"FINAL_STABLE_118K_{tag}"
    auto_root = result_root / exp_name
    auto_root.mkdir(parents=True, exist_ok=False)

    log_file = auto_root / "automatic_run.log"
    state_file = auto_root / "state.json"
    preflight_file = auto_root / "preflight.json"

    atomic_json(preflight_file, preflight_report)

    state = {
        "run_tag": tag,
        "exp_name": exp_name,
        "started_at": now_iso(),
        "training_protocol": {
            "scratch_from_zero": True,
            "scheduler_total_steps": TOTAL_STEPS,
            "trainer_max_steps": TOTAL_STEPS,
            "full_state_checkpoints": list(CHECKPOINT_STEPS),
            "final_checkpoint": TOTAL_STEPS,
            "batch_size": 1,
            "per_gpu_batchsize": 1,
            "num_workers": 0,
            "precision": 16,
            "in_training_validation": False,
            "train_batches_per_epoch": 500,
            "original_train_loader_batches": 56960,
            "lightning_train_logging": False,
            "lightning_lr_monitor_during_scratch": False,
            "pytorch_cuda_alloc_conf": "expandable_segments:True,garbage_collection_threshold:0.8",
            "cuda_module_loading": "LAZY",
            "memory_logging": "train step-only; no train epoch aggregation",
            "adamw_foreach": False,
            "cuda_memory_maintenance_every_steps": 500,
        },
        "attempts": [],
    }
    atomic_json(state_file, state)

    print("=" * 100)
    print("[AUTO] FINAL STABLE 118K")
    print("[AUTO] First attempt: TRUE SCRATCH FROM ZERO.")
    print("[AUTO] One uninterrupted Trainer.fit if the OS/hardware remains healthy.")
    print("[AUTO] FULL-STATE saves: 40,000 and 80,000. Final save: 118,000.")
    print("[AUTO] NO checkpoint at 8k, 20k, 60k or 100k.")
    print("[AUTO] num_workers=0, batch=1, precision=16, scheduler horizon=118k.")
    print("[AUTO] Parent orchestrator does NOT keep a CUDA context.")
    print("[AUTO] CUDA allocator: expandable_segments + garbage collection threshold 0.8.")
    print("[AUTO] Lightning train logging is OFF; AdamW foreach=False.")
    print("[AUTO] Training epoch length is capped at 500 batches; total max_steps stays 118000.")
    print("[AUTO] CUDA cache maintenance/telemetry runs every 500 optimizer steps.")
    print("[AUTO] 500-batch epoch limit is hard-coded in the runner; no unknown Sacred override is used.")
    print("[AUTO] No in-training validation and no validation-driven checkpointing.")
    print("=" * 100)

    resume_from: Optional[Path] = None
    native_retry_count = 0

    with KeepWindowsAwake():
        while True:
            # Do not require the original 20GB on a crash resume because the
            # already-created milestone files legitimately consume part of it.
            # We still require a hard 8GB safety floor before any retry.
            if resume_from is not None:
                require_free_space(root, 8.0, "crash-recovery restart")

            attempt_index = len(state["attempts"])
            state["attempts"].append(
                {
                    "attempt": attempt_index,
                    "started_at": now_iso(),
                    "resume_from": None if resume_from is None else str(resume_from),
                }
            )
            atomic_json(state_file, state)

            title = (
                "SCRATCH TRAINING 0 -> 118,000"
                if resume_from is None
                else f"FULL-STATE RECOVERY from {resume_from.name} -> 118,000"
            )
            rc = run_process(
                training_command(
                    python_exe,
                    runner,
                    exp_name,
                    resume_from,
                ),
                cwd=root,
                log_file=log_file,
                title=title,
            )

            state["attempts"][-1]["finished_at"] = now_iso()
            state["attempts"][-1]["return_code"] = rc
            atomic_json(state_file, state)

            if rc == 0:
                break

            step, milestone = latest_full_state_milestone(result_root, exp_name)

            if not is_native_windows_crash(rc):
                state["status"] = "failed_non_native"
                state["failed_at"] = now_iso()
                atomic_json(state_file, state)
                raise RuntimeError(
                    f"Training stopped with return code {rc}. "
                    "This is not classified as a native Windows/CUDA crash, so the "
                    "automation will NOT blindly retry a likely configuration/data error. "
                    f"See {log_file}."
                )

            native_retry_count += 1
            if native_retry_count > int(args.max_native_retries):
                state["status"] = "failed_native_retry_limit"
                atomic_json(state_file, state)
                raise RuntimeError(
                    "Native Windows/CUDA crash retry limit reached. "
                    f"See {log_file}."
                )

            if milestone is None:
                state["status"] = "failed_before_first_full_state_checkpoint"
                state["failed_at"] = now_iso()
                atomic_json(state_file, state)
                raise RuntimeError(
                    "Native Windows/CUDA crash occurred BEFORE the first 40k "
                    "FULL-STATE checkpoint. Automatic restart from zero is DISABLED "
                    "to prevent silent loss of progress. Inspect the log before relaunching."
                )
            else:
                print(
                    "[AUTO][RECOVERY] Native crash detected. "
                    f"Latest verified FULL-STATE checkpoint: step={step} | {milestone}"
                )
                resume_from = milestone

            wait = max(0, int(args.retry_wait_seconds))
            print(
                f"[AUTO][RECOVERY] Waiting {wait}s before starting a fresh "
                "Python/CUDA process..."
            )
            time.sleep(wait)

    final_candidates = [
        p
        for p in result_root.glob(
            f"{exp_name}_seed0_from_*/version_*/checkpoints/last.ckpt"
        )
        if p.is_file() and p.stat().st_size > 0
    ]
    if not final_candidates:
        raise FileNotFoundError(
            "Training returned success but no non-empty final last.ckpt was found."
        )
    final_ckpt = max(final_candidates, key=lambda p: p.stat().st_mtime)

    state["completed_at"] = now_iso()
    state["status"] = "completed"
    state["final_checkpoint"] = str(final_ckpt)
    state["final_checkpoint_size_bytes"] = final_ckpt.stat().st_size

    step40, ckpt40 = latest_full_state_milestone(result_root, exp_name)
    # Collect both milestone paths explicitly.
    milestones = {}
    for wanted in CHECKPOINT_STEPS:
        candidates = []
        for p in result_root.glob(
            f"{exp_name}_seed0_from_*/version_*/checkpoints/milestone_{wanted}.ckpt"
        ):
            if p.is_file() and p.with_suffix(".json").is_file():
                candidates.append(p)
        if candidates:
            selected = max(candidates, key=lambda p: p.stat().st_mtime)
            milestones[str(wanted)] = str(selected)
    state["milestones"] = milestones
    atomic_json(state_file, state)

    print("\n" + "=" * 100)
    print("[AUTO] TRAINING COMPLETE")
    print(f"[AUTO] Final checkpoint: {final_ckpt}")
    for step in CHECKPOINT_STEPS:
        print(f"[AUTO] {step:,}: {milestones.get(str(step), 'not found')}")
    print(f"[AUTO] State: {state_file}")
    print(f"[AUTO] Log: {log_file}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[AUTO] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[AUTO][FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
