from types import SimpleNamespace

import numpy as np
import pytest
import torch

from IEFT.modules.instance_postprocess import (
    count_instances,
    decode_fullscene_instances,
)
from IEFT.modules.instance_targets import (
    InstanceLoss,
    SCIPY_AVAILABLE,
    build_instance_targets,
    build_pseudo_instance_labels,
    label_connected_components,
)
from IEFT.modules.multiscale_change_head import PixelObjectDecoder
from IEFT.modules.objectives import compute_change_supervised


def _touching_lobes(height=96, width=96):
    yy, xx = np.ogrid[:height, :width]
    return (
        ((yy - 48) ** 2 + (xx - 32) ** 2 <= 20 ** 2)
        | ((yy - 48) ** 2 + (xx - 64) ** 2 <= 20 ** 2)
    )


def test_watershed_pseudo_targets_split_touching_lobes_without_changing_support():
    semantic = _touching_lobes()
    _, connected_count = label_connected_components(semantic)
    pseudo = build_pseudo_instance_labels(
        semantic,
        min_peak_distance=8,
        min_peak_height=3.0,
        min_instance_area=40,
    )

    assert connected_count == 1
    assert np.array_equal(pseudo > 0, semantic)
    if SCIPY_AVAILABLE:
        assert int(pseudo.max()) == 2
    else:
        assert int(pseudo.max()) == connected_count


def test_center_offset_targets_have_expected_shapes_and_finite_loss():
    mask = torch.from_numpy(_touching_lobes()).float()[None, None]
    centers, offsets, foreground = build_instance_targets(
        mask,
        sigma=3.0,
        min_peak_distance=8,
        min_peak_height=3.0,
        min_instance_area=40,
    )
    assert centers.shape == (1, 1, 96, 96)
    assert offsets.shape == (1, 2, 96, 96)
    assert torch.equal(foreground.bool(), mask.bool())
    assert torch.isfinite(centers).all()
    assert torch.isfinite(offsets).all()

    center_logits = torch.zeros_like(centers, requires_grad=True)
    offset_pred = torch.zeros_like(offsets, requires_grad=True)
    result = InstanceLoss(
        sigma=3.0,
        min_peak_distance=8,
        min_peak_height=3.0,
        min_instance_area=40,
    )(center_logits, offset_pred, mask)
    result["loss_instance"].backward()
    assert center_logits.grad is not None
    assert offset_pred.grad is not None
    assert torch.isfinite(result["loss_instance"])


def test_fullscene_decode_separates_instances_but_preserves_semantic_mask():
    semantic = _touching_lobes()
    original = semantic.copy()
    center_probability = np.zeros(semantic.shape, dtype=np.float32)
    center_probability[48, 32] = 0.95
    center_probability[48, 64] = 0.90

    yy, xx = np.indices(semantic.shape)
    offsets = np.zeros((2, *semantic.shape), dtype=np.float32)
    left = semantic & (xx < 48)
    right = semantic & ~left
    offsets[0, left] = 48 - yy[left]
    offsets[1, left] = 32 - xx[left]
    offsets[0, right] = 48 - yy[right]
    offsets[1, right] = 64 - xx[right]

    instances = decode_fullscene_instances(
        center_probability,
        offsets,
        semantic,
        center_threshold=0.3,
        min_distance=8,
    )
    assert count_instances(instances) == 2
    assert np.array_equal(instances > 0, semantic)
    assert np.array_equal(semantic, original)


def test_fullscene_decode_refuses_to_choose_a_semantic_threshold():
    semantic_probability = _touching_lobes().astype(np.float32) * 0.89
    with pytest.raises(ValueError, match="already be thresholded"):
        decode_fullscene_instances(
            np.zeros_like(semantic_probability),
            np.zeros((2, *semantic_probability.shape), dtype=np.float32),
            semantic_probability,
        )


def test_instance_disabled_decoder_has_identical_semantic_contract_and_state():
    torch.manual_seed(7)
    baseline = PixelObjectDecoder(token_dim=8, dropout=0.0, use_instance_head=False).eval()
    augmented = PixelObjectDecoder(token_dim=8, dropout=0.0, use_instance_head=True).eval()
    incompatible = augmented.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all("instance_head" in key for key in incompatible.missing_keys)
    assert all("instance_head" not in key for key in baseline.state_dict())

    token_map = torch.randn(2, 8, 2, 2)
    rgb_features = (
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 96, 8, 8),
        torch.randn(2, 128, 4, 4),
    )
    with torch.no_grad():
        baseline_outputs = baseline(token_map, rgb_features)
        augmented_outputs = augmented(token_map, rgb_features)

    assert len(baseline_outputs) == 2
    assert len(augmented_outputs) == 4
    assert torch.equal(baseline_outputs[0], augmented_outputs[0])
    assert torch.equal(baseline_outputs[1], augmented_outputs[1])
    assert augmented_outputs[2].shape == (2, 1, 32, 32)
    assert augmented_outputs[3].shape == (2, 2, 32, 32)


def _objective_inputs(use_instance_head):
    config = {
        "use_instance_head": use_instance_head,
        "instance_loss_weight": 0.3,
        "instance_center_sigma": 3.0,
        "instance_target_min_peak_distance": 4,
        "instance_target_min_peak_height": 1.0,
        "instance_target_min_area": 4,
    }
    module = SimpleNamespace(
        device=torch.device("cpu"),
        hparams=SimpleNamespace(config=config),
    )
    gt = torch.zeros(2, 1, 32, 32)
    gt[0, 0, 8:20, 7:16] = 1
    batch = {"gt_mask": gt}
    infer = {
        "change_refined_logits_up": torch.zeros(2, 32, 32, requires_grad=True),
        "change_logits": torch.zeros(2, 4, requires_grad=True),
        "change_global_logits": torch.zeros(2, requires_grad=True),
        "change_level_logits": [],
    }
    if use_instance_head:
        infer["change_instance_center_logits"] = torch.zeros(
            2, 1, 32, 32, requires_grad=True
        )
        infer["change_instance_offset"] = torch.zeros(
            2, 2, 32, 32, requires_grad=True
        )
    return module, batch, infer


def test_objective_adds_instance_loss_only_when_enabled():
    module_off, batch, infer_off = _objective_inputs(False)
    result_off = compute_change_supervised(module_off, batch, infer=infer_off)
    assert "loss_instance" not in result_off

    module_on, batch, infer_on = _objective_inputs(True)
    result_on = compute_change_supervised(module_on, batch, infer=infer_on)
    assert "loss_instance" in result_on
    expected = result_off["loss"] + 0.3 * result_on["loss_instance"]
    assert torch.allclose(result_on["loss"], expected)
