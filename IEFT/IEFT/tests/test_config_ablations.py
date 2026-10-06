from pathlib import Path

from IEFT.config import ex
from prepare_levir_temporal_osm import parse_args as parse_osm_args
from run_final_stable_118k_auto import training_command


def _named(name: str) -> dict:
    return dict(ex.named_configs[name]())


def test_final_118k_command_uses_only_the_retained_configuration_chain():
    command = training_command(
        "python",
        Path("run_final_stable_118k.py"),
        "FINAL_TEST",
        resume_from=None,
    )
    assert command[3:8] == [
        "with",
        "task_levir_cd_v20_cliprank_vitb16_levir_dense_completion",
        "levir_scratch_full_base",
        "levir_scratch_rgb",
        "levir_final_rgb_osm_instance_safe_spectral",
    ]
    assert "max_steps=118000" in command
    assert not any(value.startswith("resume_from=") for value in command)


def test_final_configuration_keeps_only_bounded_auxiliary_residuals_active():
    cfg = _named("levir_final_rgb_osm_instance_safe_spectral")
    assert cfg["change_use_safe_osm_guidance"] is True
    assert cfg["change_use_safe_spectral_late"] is True
    assert cfg["change_safe_aux_total_max_prob_delta"] == 0.025
    assert cfg["change_use_osm_t2_early"] is False
    assert cfg["change_use_osm_t2_late"] is False
    assert cfg["change_spectral_fusion"] == "none"
    assert cfg["use_instance_head"] is True
    assert cfg["instance_detach_shared_features"] is True


def test_retained_auxiliary_preset_is_t2_only_and_uses_requested_paths():
    cfg = _named("levir_v20d_t2_osm_paired_spectral")
    assert cfg["levir_temporal_osm_mode"] == "t2"
    assert cfg["change_osm_mode"] == "t2"
    assert cfg["levir_require_osm_t1"] is False
    assert cfg["levir_require_osm_t2"] is True
    assert cfg["levir_temporal_osm_manifest"] == "data_osm_t2_v2/manifest.json"
    assert cfg["levir_spectral_manifest"] == "data_spectral_v2/manifest.json"
    assert cfg["load_path"].endswith(
        "v_20d_full_118k_v2_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )
    assert cfg["resume_from"] is None


def test_common_subset_ablation_switches_are_independent():
    expected = {
        "levir_aux_ablation_rgb_common_subset": ("none", False, False),
        "levir_aux_ablation_spectral_common_subset": ("adapter", False, False),
        "levir_aux_ablation_spectral_osm_late": ("adapter", False, True),
        "levir_aux_ablation_spectral_osm_early": ("adapter", True, False),
        "levir_aux_ablation_spectral_osm_early_late": ("adapter", True, True),
    }
    names = set()
    for name, (spectral, early, late) in expected.items():
        cfg = _named(name)
        assert cfg["change_spectral_fusion"] == spectral
        assert cfg["change_use_osm_t2_early"] is early
        assert cfg["change_use_osm_t2_late"] is late
        assert cfg["use_bidirectional_fie"] is False
        assert cfg["use_instance_head"] is False
        names.add(cfg["exp_name"])
    assert len(names) == len(expected)


def test_osm_preparer_defaults_to_t2_only_requested_cache():
    args = parse_osm_args(["generate", "--dry-run", "--history-start", "2007-10-08"])
    assert args.temporal_key == "t2"
    assert args.output_root == "data_osm_t2_v2"
    assert args.manifest_name == "manifest.json"
