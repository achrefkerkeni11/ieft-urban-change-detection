"""Opt-in integrity and compatibility check for the final 118k checkpoint."""

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from IEFT.modules.vilt_module import ViLTransformerSS


EXPECTED_SHA256 = "00836fe490c1e9e291bc477414d4d3e8314281186c438cb8c086302371c2e11d"
EXPECTED_SIZE = 1_564_322_778


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.skipif(
    os.environ.get("IEFT_RUN_FINAL_CHECKPOINT_TEST") != "1",
    reason="set IEFT_RUN_FINAL_CHECKPOINT_TEST=1 for the heavyweight test",
)
def test_final_checkpoint_hash_and_model_compatibility():
    configured_path = os.environ.get("IEFT_FINAL_CHECKPOINT", "").strip()
    assert configured_path, "set IEFT_FINAL_CHECKPOINT to the final last.ckpt"

    checkpoint = Path(configured_path).expanduser().resolve()
    assert checkpoint.is_file(), checkpoint
    assert checkpoint.stat().st_size == EXPECTED_SIZE
    assert _sha256(checkpoint) == EXPECTED_SHA256

    hparams_path = checkpoint.parents[1] / "hparams.yaml"
    assert hparams_path.is_file(), hparams_path
    document = yaml.safe_load(hparams_path.read_text(encoding="utf-8"))
    config = dict(document.get("config", document))

    model = ViLTransformerSS(config).eval()
    report = model.load_compatible_checkpoint(
        checkpoint,
        map_location="cpu",
        minimum_parameter_coverage=0.98,
        strict_compatibility=True,
    )
    assert report["parameter_coverage"] >= 0.98
    assert not report["shape_mismatches"]
