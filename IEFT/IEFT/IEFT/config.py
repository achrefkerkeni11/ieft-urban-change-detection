# -*- coding: utf-8 -*-
"""Sacred configuration for the maintained IEFT/LEVIR-CD pipeline."""

from sacred import Experiment

ex = Experiment("ViLT", save_git_info=False)


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["levir_cd"]
    loss_names = _loss_names({"irtr": 0})

    batch_size = 8
    per_gpu_batchsize = 8
    num_workers = 4
    num_gpus = 1
    num_nodes = 1
    precision = 16

    train_transform_keys = ["pixelbert_randaug"]
    val_transform_keys = ["pixelbert"]
    image_size = 256
    model_input_size = 256
    patch_size = 16
    max_image_len = -1
    draw_false_image = 0
    image_only = False

    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    draw_false_text = 0
    whole_word_masking = False
    mlm_prob = 0.15
    levir_fixed_text = "building change detection"

    vit = "vit_small_patch16_224"
    # Offline-only: timm never downloads weights at model construction.  Set
    # vit_pretrained=True only together with a readable vit_encoder_ckpt_path.
    vit_pretrained = False
    vit_encoder_ckpt_path = ""
    vit_encoder_partial_load = True
    vit_encoder_freeze_steps = 0

    hidden_size = 384
    num_heads = 6
    num_layers = 4
    mlp_ratio = 4
    drop_rate = 0.10

    optim_type = "adamw"
    learning_rate = 3e-4
    encoder_learning_rate = 2e-5
    weight_decay = 0.05
    # "legacy" preserves the historical optimizer behavior.  New controlled
    # modality experiments use "aux_only" so the 118k RGB core cannot drift.
    change_train_scope = "legacy"
    change_aux_learning_rate = 1e-4
    change_aux_weight_decay = 0.01

    # Explicit research contract for true random-initialization experiments.
    # When True, the final training runner refuses every warm-start or
    # local-pretrained path; verified full-state crash recovery remains allowed.
    scratch_training = False
    scratch_checkpoint_every_n_steps = 2000
    max_epoch = 40
    max_steps = 6000
    warmup_steps = 200
    end_lr = 0.0
    lr_mult = 1.0

    get_recall_metric = False
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    eval_split = "test"
    change_eval_threshold = 0.5
    checkpoint_min_parameter_coverage = 0.75
    checkpoint_strict_compatibility = True

    data_root = ""
    log_dir = "result"
    load_path = ""
    json = ""

    levir_label_dirname = "label"
    levir_image_a_dirname = "A"
    levir_image_b_dirname = "B"
    levir_mask_threshold = 127

    levir_train_crop_size = 256
    levir_val_crop_size = 256
    levir_tile_size = 256
    levir_tile_stride = 256
    levir_train_repeat = 8
    levir_train_focus_positive = True
    levir_train_positive_focus_prob = 0.62
    levir_train_hard_negative_prob = 0.20
    levir_train_random_aug = True
    levir_label_smoothing = 0.0
    # One authoritative base hard-negative policy.  Named historical configs
    # may override these values without relying on duplicate assignments.
    levir_train_positive_thr = 0.003
    levir_train_negative_mining_mode = "rural_structural"
    levir_train_hard_negative_gamma = 2.0
    levir_train_hard_negative_min_weight = 0.02
    levir_train_extreme_negative_prob = 0.10
    levir_train_extreme_negative_quantile = 0.88

    use_semantic_multiscale = True
    ms_change_num_levels = 4
    ms_change_level_indices = [2, 5, 8, 11]
    ms_change_decoder_dim = 160
    ms_change_dropout = 0.08

    # Optional pseudo-instance side head. LEVIR-CD has semantic labels only;
    # center/offset targets are derived on the fly and never alter semantics.
    use_instance_head = False
    # Final safe default: the instance loss trains only the center/offset head.
    # Shared RGB semantic features are detached before entering that head.
    instance_detach_shared_features = True
    instance_init_seed = 1702
    instance_loss_weight = 0.3
    instance_w_center = 1.0
    instance_w_offset = 0.05
    instance_center_sigma = 6.0
    instance_target_use_watershed = True
    # Conservative defaults at LEVIR's 0.5 m GSD; tune on validation only.
    instance_target_min_peak_distance = 12
    instance_target_min_peak_height = 3.0
    instance_target_min_area = 64
    instance_center_threshold = 0.3
    instance_decode_min_distance = 12

    change_global_gate_floor = 0.80
    change_apply_global_gate_to_dense = False
    change_apply_global_gate_to_coarse = False
    change_global_aux_scale = 0.05
    change_dense_coarse_fuse_weight = 0.08
    change_dense_global_bias_scale = 0.08

    change_dense_pos_weight = 5.0
    change_local_pos_weight = 5.0
    change_boundary_pos_weight = 3.0

    change_dense_bce_weight = 0.52
    change_dense_dice_weight = 0.48
    change_local_loss_weight = 0.10
    change_global_loss_weight = 0.12
    change_boundary_loss_weight = 0.10
    change_boundary_dice_weight = 0.06
    change_color_only_penalty_weight = 0.02
    change_deep_sup_loss_weight = 0.05
    change_no_change_penalty_weight = 0.14
    change_no_change_max_penalty_weight = 0.18
    change_no_change_topk_penalty_weight = 0.14
    change_no_change_topk = 64
    change_force_binary_gt_for_losses = True
    change_force_binary_gt_threshold = 0.5
    change_dense_completion_weight = 0.08
    change_dense_completion_kernel = 5

    dense_export_no_change_global_thr = 0.16
    dense_export_no_change_mean_thr = 0.030

    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_osm_text_mode = "concat"
    levir_osm_max_phrases = 3
    levir_osm_text_key = "text_v21"
    levir_osm_fallback_text = "no_osm_context"
    levir_osm_compose_mode = "signature_compact"
    levir_osm_word_budget = 32
    levir_osm_joiner = " ; "
    levir_osm_include_source_text = False

    # Offline historical OSM.  These fields never trigger a network request;
    # manifests must be prepared before constructing a DataLoader.
    levir_temporal_osm_enabled = False
    # None retains legacy inference; canonical auxiliary configs use either
    # "paired" or the explicit single-snapshot "t2" mode.
    levir_temporal_osm_mode = None
    levir_temporal_osm_manifest = ""
    levir_osm_cache_size = 1
    levir_require_osm_t1 = True
    levir_require_osm_t2 = True
    levir_require_nonempty_osm = False
    levir_filter_failed_osm = True
    levir_osm_timestamp_policy = "month_end"
    # Compatibility aliases retained for historical manifests and checkpoints.
    levir_require_paired_osm = False
    levir_use_temporal_osm = False

    # Genuine bitemporal multispectral indices.  NDWI is always McFeeters in
    # the primary workflow; official LEVIR RGB is never used to derive it.
    levir_spectral_indices_enabled = False
    levir_spectral_provider = "earth_engine_landsat"
    levir_spectral_manifest = ""
    levir_require_indices_t1 = True
    levir_require_indices_t2 = True
    levir_accept_partial_indices = True
    levir_min_index_valid_fraction = 0.80
    levir_spectral_missing_policy = "filter"
    levir_spectral_fusion = "adapter"
    levir_ndwi_variant = "mcfeeters"
    levir_index_normalization = "natural"
    levir_index_normalization_stats = ""
    levir_spectral_cache_size = 2
    # Explicit compatibility tier for the retained low-resolution Landsat
    # NDVI/NDWI cache. False keeps the stricter Green/Red/NIR+index schema.
    levir_allow_legacy_spectral_indices = False
    levir_require_spectral = False
    levir_use_spectral = False
    levir_auxiliary_policy = "mask"

    # LEVIR's active model does not consume tokenized natural language.  The
    # deterministic local tokenizer prevents hidden HuggingFace downloads.
    levir_tokenizer_mode = "simple"
    levir_tokenizer_local_only = True
    levir_tokenizer_fallback_to_simple = False

    change_use_osm_struct = False
    change_osm_struct_dim = 16
    change_osm_struct_hidden = 128
    change_osm_global_weight = 0.10
    change_osm_coarse_weight = 0.08
    change_osm_dense_weight = 0.05
    change_osm_patch_weight = 0.10
    change_osm_gate_floor = 0.10
    change_osm_reliability_bias = 0.0
    change_osm_use_text = False
    # None preserves legacy auto-detection from change_use_osm_struct. New
    # source-backed presets set an explicit temporal mode.
    change_osm_mode = None
    change_osm_temporal_hidden = 128
    change_osm_temporal_residual_scale = 0.01
    change_use_osm_t2_early = False
    # Spatial early fusion is only scientifically trusted for exact historical
    # geometries.  Bbox-only caches remain valid for structured/late context.
    change_osm_t2_early_require_exact_geometry = True
    # None preserves the historical late-OSM auto-detection. T2 ablation
    # presets set this to an explicit True/False independently of early OSM.
    change_use_osm_t2_late = None
    change_osm_t2_early_hidden = 32
    change_osm_t2_early_residual_scale = 0.01
    change_osm_t2_late_residual_scale = 0.0

    # Final conservative OSM guidance.  This is separate from the historical
    # late/early OSM fusion paths.  It consumes only the tile-local 16-D T2 OSM
    # structure and applies a small bounded correction to uncertain RGB pixels.
    change_use_safe_osm_guidance = False
    change_safe_osm_hidden = 32
    change_safe_osm_max_prob_delta = 0.015
    change_safe_osm_uncertainty_power = 2.0
    change_safe_osm_loss_weight = 0.50
    change_safe_osm_safety_weight = 3.0
    change_safe_osm_l1_weight = 0.10
    change_safe_osm_init_seed = 1703
    change_safe_aux_total_max_prob_delta = 0.025

    change_spectral_fusion = "none"
    change_spectral_adapter_hidden = 32
    change_spectral_residual_scale = 0.01

    # Final conservative spectral mode.  The old early adapter remains available
    # for historical reproducibility, but the final architecture keeps it OFF and
    # uses only this bounded late NDVI/NDWI residual.
    change_use_safe_spectral_late = False
    change_safe_spectral_hidden = 24
    change_safe_spectral_max_prob_delta = 0.02
    change_safe_spectral_min_reliability = 0.50
    change_safe_spectral_uncertainty_power = 2.0
    change_safe_spectral_loss_weight = 1.0
    change_safe_spectral_safety_weight = 2.0
    change_safe_spectral_l1_weight = 0.05
    change_safe_spectral_init_seed = 1701

    use_bidirectional_fie = False

    # Canonical scientific reconstruction.  Threshold selection is performed
    # on full VAL scenes and frozen before TEST; this patch-level field remains
    # only for Lightning monitoring.
    full_scene_tile_size = 256
    full_scene_tile_stride = 128
    full_scene_hann_floor = 0.20
    full_scene_threshold_start = 0.500
    full_scene_threshold_stop = 0.990
    full_scene_threshold_step = 0.001
    historical_v20d_full_scene_threshold = 0.89

    # Offline CLIP-filtered OSM controls.
    # These MUST exist in the base Sacred config so named configs may override
    # them without triggering ConfigAddedError.
    change_use_clip_filtered_osm = False
    change_osm_clip_confidence_floor = 0.75

    change_clip_filter_runtime = False
    clip_topk = 2
    clip_min_keep = 1
    clip_use_summary = True
    clip_use_main_text = True

    export_panel_title_bar_h = 40
    export_panel_font_size = 18


@ex.named_config
def task_levir_cd_v20_cliprank_vitb16_levir_dense_completion():
    exp_name = "levir_cd_v20_cliprank_vitb16_levir_dense_completion"
    datasets = ["levir_cd"]
    loss_names = _loss_names({"irtr": 0})

    image_size = 256
    model_input_size = 256
    patch_size = 16
    vit = "vit_base_patch16_224"
    vit_pretrained = False
    vit_encoder_ckpt_path = ""
    vit_encoder_partial_load = True
    vit_encoder_freeze_steps = 300
    hidden_size = 768
    num_heads = 12
    num_layers = 4

    batch_size = 8
    per_gpu_batchsize = 8
    num_workers = 0
    max_epoch = 40
    max_steps = 8000
    warmup_steps = 300

    encoder_learning_rate = 1e-5
    learning_rate = 2e-4
    weight_decay = 0.05

    levir_train_crop_size = 256
    levir_val_crop_size = 256
    levir_tile_size = 256
    levir_tile_stride = 256
    levir_train_repeat = 8
    levir_train_focus_positive = True
    levir_train_positive_focus_prob = 0.66
    levir_train_hard_negative_prob = 0.28
    levir_train_random_aug = True
    levir_label_smoothing = 0.0
    levir_train_positive_thr = 0.003
    levir_train_negative_mining_mode = "rural_structural"
    levir_train_hard_negative_gamma = 2.2
    levir_train_hard_negative_min_weight = 0.02
    levir_train_extreme_negative_prob = 0.10
    levir_train_extreme_negative_quantile = 0.88

    use_semantic_multiscale = True
    ms_change_num_levels = 4
    ms_change_level_indices = [2, 5, 8, 11]
    ms_change_decoder_dim = 128
    ms_change_dropout = 0.08

    change_global_gate_floor = 0.82
    change_apply_global_gate_to_dense = False
    change_apply_global_gate_to_coarse = False
    change_global_aux_scale = 0.05
    change_dense_coarse_fuse_weight = 0.10
    change_dense_global_bias_scale = 0.08

    change_dense_pos_weight = 5.0
    change_local_pos_weight = 5.0
    change_boundary_pos_weight = 3.0

    change_dense_bce_weight = 0.50
    change_dense_dice_weight = 0.50
    change_local_loss_weight = 0.10
    change_global_loss_weight = 0.12
    change_boundary_loss_weight = 0.10
    change_boundary_dice_weight = 0.06
    change_color_only_penalty_weight = 0.03
    change_deep_sup_loss_weight = 0.05
    change_no_change_penalty_weight = 0.18
    change_no_change_max_penalty_weight = 0.20
    change_no_change_topk_penalty_weight = 0.16
    change_no_change_topk = 64
    change_force_binary_gt_for_losses = True
    change_force_binary_gt_threshold = 0.5
    change_dense_completion_weight = 0.10
    change_dense_completion_kernel = 5

    levir_use_osm = True
    levir_osm_texts_json = "data_osm\\osm_texts_by_patch_levir_v21.json"
    levir_osm_text_mode = "concat"
    levir_osm_max_phrases = 4
    levir_osm_text_key = "text_v21"
    levir_osm_fallback_text = "no_osm_context"
    levir_osm_compose_mode = "signature_compact"
    levir_osm_word_budget = 40
    levir_osm_joiner = " ; "
    levir_osm_include_source_text = False

    change_use_osm_struct = True
    change_osm_struct_dim = 16
    change_osm_struct_hidden = 128
    change_osm_global_weight = 0.14
    change_osm_coarse_weight = 0.10
    change_osm_dense_weight = 0.08
    change_osm_patch_weight = 0.14
    change_osm_gate_floor = 0.16
    change_osm_reliability_bias = 0.05
    change_osm_use_text = False

    # Rollback stable v20d: CLIP is not injected inside vilt_module.py.
    # Keep OSM structured guidance; CLIP can be used offline later to build a better OSM JSON.
    change_clip_filter_runtime = False
    clip_topk = 2
    clip_min_keep = 1
    clip_use_summary = True
    clip_use_main_text = True


@ex.named_config
def levir_v20d_118k_exact_lineage():
    # Stored hyperparameters for the historical v20d 118k training lineage.
    # This preset is intentionally documentary as well as executable: bit-exact
    # regeneration still requires the absent 58k trainer checkpoint and legacy
    # per-patch OSM JSON named below. ``load_path`` alone is not a substitute
    # for the missing optimizer/trainer state in ``resume_from``.

    exp_name = "v_20d_full_118k_v2"
    seed = 0
    batch_size = 1
    per_gpu_batchsize = 1
    learning_rate = 1e-4
    max_epoch = 80
    max_steps = 118000
    change_clip_filter_runtime = True
    levir_use_osm = True
    levir_osm_texts_json = "data_osm/osm_texts_by_patch_levir_v21.json"
    load_path = "weights/vilt_200k_mlm_itm.ckpt"
    resume_from = (
        "result/v_20d_full_58k_v1_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )


@ex.named_config
def levir_v20d_t2_osm_paired_spectral():
    # Extend the retained v20d 118k model with T2 OSM and T1/T2 indices.

    exp_name = "v_20d_full_118k_v2_t2_osm_paired_spectral"
    per_gpu_batchsize = 1

    # This is a new warm-started branch, not continuation of the unavailable
    # historical 58k trainer state declared by the exact-lineage preset.
    resume_from = None
    load_path = (
        "result/v_20d_full_118k_v2_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )

    # The manifest is generated exclusively by prepare_levir_temporal_osm.py
    # with --temporal-key t2.  T1 is intentionally not an eligibility gate.
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_require_osm_t1 = False
    levir_require_osm_t2 = True
    levir_require_paired_osm = False
    levir_require_nonempty_osm = False
    # T2's explicit requirement already filters every unavailable/failed T2
    # record.  Disable the legacy pair-wide failure gate so T1 is never used
    # to decide source eligibility if an older paired manifest is supplied.
    levir_filter_failed_osm = False
    levir_use_temporal_osm = True

    levir_spectral_indices_enabled = True
    levir_spectral_manifest = "data_spectral_v2/manifest.json"
    levir_require_indices_t1 = True
    levir_require_indices_t2 = True
    levir_require_spectral = True
    levir_use_spectral = True
    levir_allow_legacy_spectral_indices = True

    # Explicit per-date requirements plus strict/filter policy give the common
    # subset: successful T2 OSM and valid paired T1/T2 NDVI/NDWI.
    levir_auxiliary_policy = "strict"
    levir_spectral_missing_policy = "filter"

    # Never fall back to the legacy generated/current-state OSM JSON.
    levir_use_osm = False
    levir_osm_texts_json = ""
    change_osm_mode = "t2"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = True
    change_use_osm_t2_late = True
    change_osm_t2_early_require_exact_geometry = True
    change_train_scope = "aux_only"


@ex.named_config
def levir_v20d_rgb_only_control():
    """Historical RGB branch; no T1/T2 OSM and no spectral adapter."""

    exp_name = "v20d_rgb_only_control"
    resume_from = None
    load_path = (
        "result/v_20d_full_118k_v2_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )
    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = False
    levir_temporal_osm_mode = None
    levir_temporal_osm_manifest = ""
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False
    levir_spectral_indices_enabled = False
    levir_spectral_manifest = ""
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    change_osm_mode = "none"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_spectral_fusion = "none"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_v20d_spectral_only():
    """RGB plus genuine paired Landsat NDVI/NDWI context, no OSM."""

    exp_name = "v20d_spectral_only"
    resume_from = None
    load_path = (
        "result/v_20d_full_118k_v2_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )
    levir_temporal_osm_enabled = False
    levir_temporal_osm_mode = None
    levir_temporal_osm_manifest = ""
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False
    levir_spectral_indices_enabled = True
    levir_spectral_manifest = "data_spectral_v2/manifest.json"
    levir_allow_legacy_spectral_indices = True
    levir_require_indices_t1 = True
    levir_require_indices_t2 = True
    levir_spectral_missing_policy = "filter"
    change_osm_mode = "none"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_spectral_fusion = "adapter"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_v20d_osm_t2_late_only():
    """RGB plus legacy-compatible late T2 OSM guidance only."""

    exp_name = "v20d_osm_t2_late_only"
    resume_from = None
    load_path = (
        "result/v_20d_full_118k_v2_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )
    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_require_osm_t1 = False
    levir_require_osm_t2 = True
    levir_filter_failed_osm = False
    levir_spectral_indices_enabled = False
    levir_spectral_manifest = ""
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    change_osm_mode = "t2"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = True
    change_spectral_fusion = "none"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_v20d_osm_t2_early_only():
    """RGB plus zero-init early T2 spatial/structured OSM residual only."""

    exp_name = "v20d_osm_t2_early_only"
    resume_from = None
    load_path = (
        "result/v_20d_full_118k_v2_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )
    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_require_osm_t1 = False
    levir_require_osm_t2 = True
    levir_filter_failed_osm = False
    levir_spectral_indices_enabled = False
    levir_spectral_manifest = ""
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    change_osm_mode = "t2"
    change_use_osm_t2_early = True
    change_use_osm_t2_late = False
    change_spectral_fusion = "none"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_ablation_rgb_common_subset():
    """Common T2-OSM/paired-spectral eligible sources, with both branches bypassed.

    Compose this *after* ``levir_v20d_t2_osm_paired_spectral``.  Keeping the
    manifests and strict eligibility enabled makes A/B/C/D compare the same
    source set; these switches only control model fusion.
    """

    exp_name = "v20d_aux_common_rgb"
    levir_spectral_fusion = "none"
    change_spectral_fusion = "none"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_train_scope = "legacy"  # evaluation-only common RGB control
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_ablation_spectral_common_subset():
    """A: paired spectral adapter only, on the common auxiliary source set."""

    exp_name = "v20d_aux_common_spectral"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_ablation_spectral_osm_late():
    """B: paired spectral plus late T2 OSM, on the common source set."""

    exp_name = "v20d_aux_common_spectral_osm_late"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = True
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_ablation_spectral_osm_early():
    """C: paired spectral plus early T2 OSM, on the common source set."""

    exp_name = "v20d_aux_common_spectral_osm_early"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = True
    change_use_osm_t2_late = False
    change_osm_t2_early_require_exact_geometry = True
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_ablation_spectral_osm_early_late():
    """D: paired spectral plus both T2 OSM paths, on the common source set."""

    exp_name = "v20d_aux_common_spectral_osm_early_late"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = True
    change_use_osm_t2_late = True
    change_osm_t2_early_require_exact_geometry = True
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_v20d_aux_full_benchmark():
    """Full 445/64/128 LEVIR auxiliary runtime without source filtering.

    Compose this before one of the ``levir_aux_full_*`` modality switches.
    Partial/missing auxiliary observations are handled by masks/reliability;
    they do not remove an RGB/GT source from the canonical benchmark.
    """

    exp_name = "v20d_aux_full_base"
    resume_from = None
    load_path = (
        "result/v_20d_full_118k_v2_seed0_from_vilt_200k_mlm_itm/"
        "version_0/checkpoints/last.ckpt"
    )
    per_gpu_batchsize = 1

    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False
    levir_require_paired_osm = False
    levir_require_nonempty_osm = False
    levir_filter_failed_osm = False
    levir_use_temporal_osm = True

    levir_spectral_indices_enabled = True
    levir_spectral_manifest = "data_spectral_v2/manifest.json"
    levir_allow_legacy_spectral_indices = True
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    levir_require_spectral = False
    levir_use_spectral = True
    levir_accept_partial_indices = True
    levir_min_index_valid_fraction = 0.0
    levir_spectral_missing_policy = "mask"
    levir_auxiliary_policy = "mask"

    # Reproduce the retained controlled fine-tuning normalization: statistics
    # are recomputed from TRAIN only and stored in this immutable JSON.
    levir_index_normalization = "sensor_train_stats"
    levir_index_normalization_stats = "data_spectral_v2/train_sensor_stats.json"

    change_osm_mode = "t2"
    change_train_scope = "aux_only"
    change_aux_learning_rate = 1e-4
    change_aux_weight_decay = 0.01
    change_osm_t2_early_require_exact_geometry = True
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_full_rgb_control():
    """S0: full-source RGB control; auxiliary data may load but fusion is bypassed."""

    exp_name = "v20d_aux_full_rgb_control"
    levir_spectral_fusion = "none"
    change_spectral_fusion = "none"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_train_scope = "legacy"  # control is evaluation-only, not fine-tuned
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_full_spectral():
    """S1: full-source RGB + NDVI/NDWI, auxiliary-only optimization."""

    exp_name = "v20d_aux_full_spectral"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_full_spectral_osm_late():
    """S2: full-source spectral + late T2 OSM structured guidance."""

    exp_name = "v20d_aux_full_spectral_osm_late"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = True
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_full_spectral_osm_early():
    """S3: spectral + early T2 OSM; exact geometry is required by default."""

    exp_name = "v20d_aux_full_spectral_osm_early"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = True
    change_use_osm_t2_late = False
    change_osm_t2_early_require_exact_geometry = True
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_aux_full_spectral_osm_early_late():
    """S4: spectral + early + late T2 OSM; exact early geometry required."""

    exp_name = "v20d_aux_full_spectral_osm_early_late"
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"
    change_use_osm_t2_early = True
    change_use_osm_t2_late = True
    change_osm_t2_early_require_exact_geometry = True
    change_train_scope = "aux_only"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_scratch_full_base():
    """True random-initialization LEVIR training contract.

    No v20d/ViLT/local encoder checkpoint is loaded.  The complete semantic
    network is optimized from random initialization.  Compose this before one
    of the ``levir_scratch_*`` modality switches below.
    """

    exp_name = "scratch_full_base"
    scratch_training = True
    seed = 0

    # Hard no-pretraining / no-resume contract.
    resume_from = None
    load_path = ""
    vit_pretrained = False
    vit_encoder_ckpt_path = ""
    vit_encoder_partial_load = False
    vit_encoder_freeze_steps = 0

    # Full optimization: unlike aux_only, the complete enabled model learns.
    change_train_scope = "legacy"

    # RTX-2050-safe micro-batch.  Keep logical batch size equal to one for the
    # first scratch study so max_steps means 8,000 real optimizer updates and
    # the R0/R1/R2 comparisons share the exact same optimization budget.
    batch_size = 1
    per_gpu_batchsize = 1
    num_workers = 0
    num_gpus = 1
    num_nodes = 1
    precision = 16

    max_epoch = 1000
    max_steps = 8000
    warmup_steps = 300
    end_lr = 0.0
    encoder_learning_rate = 1e-5
    learning_rate = 2e-4
    weight_decay = 0.05
    val_check_interval = 1000
    scratch_checkpoint_every_n_steps = 2000

    # Shared training/data protocol across every scratch ablation.
    levir_train_random_aug = True
    levir_auxiliary_policy = "mask"
    levir_spectral_missing_policy = "mask"
    levir_accept_partial_indices = True
    levir_min_index_valid_fraction = 0.0
    levir_allow_legacy_spectral_indices = True
    levir_index_normalization = "sensor_train_stats"
    levir_index_normalization_stats = "data_spectral_v2/train_sensor_stats.json"

    # Keep unrelated experimental branches out of R0/R1/R2.
    change_clip_filter_runtime = False
    use_bidirectional_fie = False
    use_instance_head = False
    change_osm_t2_early_require_exact_geometry = True


@ex.named_config
def levir_scratch_rgb():
    """R0-Scratch: RGB-only baseline, every semantic parameter random/trainable."""

    exp_name = "R0_scratch_rgb"

    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = False
    levir_temporal_osm_mode = None
    levir_temporal_osm_manifest = ""
    levir_use_temporal_osm = False
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False

    levir_spectral_indices_enabled = False
    levir_spectral_manifest = ""
    levir_use_spectral = False
    levir_require_spectral = False
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    levir_spectral_fusion = "none"

    change_osm_mode = "none"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_spectral_fusion = "none"
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_scratch_spectral():
    """R1-Scratch: RGB + genuine paired Landsat NDVI/NDWI from random init."""

    exp_name = "R1_scratch_spectral"

    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = False
    levir_temporal_osm_mode = None
    levir_temporal_osm_manifest = ""
    levir_use_temporal_osm = False
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False

    levir_spectral_indices_enabled = True
    levir_spectral_manifest = "data_spectral_v2/manifest.json"
    levir_use_spectral = True
    levir_require_spectral = False
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"

    change_osm_mode = "none"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_scratch_spectral_osm_late():
    """R2-Scratch: R1 plus historical T2 OSM structured/late guidance.

    The current cache is bbox geometry, therefore early spatial OSM remains
    disabled.  Only the scientifically valid T2 structured/late context is used.
    """

    exp_name = "R2_scratch_spectral_osm_late"

    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_use_temporal_osm = True
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False
    levir_require_paired_osm = False
    levir_require_nonempty_osm = False
    levir_filter_failed_osm = False

    levir_spectral_indices_enabled = True
    levir_spectral_manifest = "data_spectral_v2/manifest.json"
    levir_use_spectral = True
    levir_require_spectral = False
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    levir_spectral_fusion = "adapter"
    change_spectral_fusion = "adapter"

    change_osm_mode = "t2"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = True
    change_osm_t2_early_require_exact_geometry = True
    use_bidirectional_fie = False
    use_instance_head = False


@ex.named_config
def levir_v20d_aux_instance_experiment():
    """Optional pseudo-instance side task; semantic mask remains independent."""

    use_instance_head = True


@ex.named_config
def levir_v20d_bidirectional_fie_experiment():
    """Optional zero-init bidirectional residual after each historical FIE."""

    use_bidirectional_fie = True


@ex.named_config
def levir_scratch_rgb_osm_late_instance_bidir():
    """R2-FULL-NO-SPECTRAL: RGB + valid T2 OSM late + building separation.

    This is the "everything useful except spectral" scratch candidate:
      - RGB T1/T2 semantic IEFT core: ON
      - historical T2 OSM structured/late guidance: ON
      - early spatial OSM: OFF because the retained cache uses bbox-approximate
        geometries rather than exact feature geometries
      - pseudo-instance building-separation head: ON
      - bidirectional FIE residual adapters: ON
      - spectral / NDVI / NDWI: COMPLETELY OFF
      - runtime CLIP injection: OFF (stable semantic protocol)
    """

    exp_name = "R2_scratch_rgb_osmT2late_instance_bidir"

    # OSM: only the scientifically valid T2 temporal cache.
    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_use_temporal_osm = True
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False
    levir_require_paired_osm = False
    levir_require_nonempty_osm = False
    levir_filter_failed_osm = False

    change_osm_mode = "t2"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = True
    change_osm_t2_early_require_exact_geometry = True

    # Spectral is disabled everywhere.
    levir_spectral_indices_enabled = False
    levir_spectral_manifest = ""
    levir_use_spectral = False
    levir_require_spectral = False
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    levir_spectral_fusion = "none"
    change_spectral_fusion = "none"

    # Additional valid branches requested for this candidate.
    use_bidirectional_fie = True

    # Building separation: auxiliary pseudo-instance task.
    use_instance_head = True
    instance_loss_weight = 0.3
    instance_w_center = 1.0
    instance_w_offset = 0.05
    instance_center_sigma = 6.0
    instance_target_use_watershed = True
    instance_target_min_peak_distance = 12
    instance_target_min_peak_height = 3.0
    instance_target_min_area = 64
    instance_center_threshold = 0.3
    instance_decode_min_distance = 12

    # Stable semantic protocol.
    change_clip_filter_runtime = False

@ex.named_config
def levir_scratch_instance_supervised_corrected():
    exp_name = "R0I2_scratch_rgb_instance_supervised"

    use_instance_head = True
    instance_loss_weight = 0.30
    instance_w_center = 1.0
    instance_w_offset = 0.05
    instance_center_sigma = 6.0
    instance_target_use_watershed = True
    instance_target_min_peak_distance = 12
    instance_target_min_peak_height = 3.0
    instance_target_min_area = 64
    instance_center_threshold = 0.15
    instance_decode_min_distance = 12

    change_use_clip_filtered_osm = False
    change_clip_filter_runtime = False
    use_bidirectional_fie = False


@ex.named_config
def levir_scratch_clipfiltered_osm_lite_instance():
    exp_name = "R3_scratch_rgb_instance_clipfiltered_osm_lite"

    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_use_temporal_osm = True
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False
    levir_require_paired_osm = False
    levir_require_nonempty_osm = False
    levir_filter_failed_osm = False

    change_osm_mode = "t2"
    change_use_osm_t2_early = False
    change_use_osm_t2_late = True
    change_osm_t2_early_require_exact_geometry = True

    # Lightweight late fusion.
    change_osm_t2_late_residual_scale = 0.10
    change_osm_global_weight = 0.08
    change_osm_coarse_weight = 0.06
    change_osm_patch_weight = 0.06
    change_osm_dense_weight = 0.04
    change_osm_gate_floor = 0.10
    change_osm_reliability_bias = 0.0

    # CLIP filters OSM offline; it never contributes logits directly.
    change_use_clip_filtered_osm = True
    change_osm_clip_confidence_floor = 0.75
    change_clip_filter_runtime = False

    # Spectral and bidirectional branches remain off.
    levir_spectral_indices_enabled = False
    levir_spectral_manifest = ""
    levir_use_spectral = False
    levir_require_spectral = False
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    levir_spectral_fusion = "none"
    change_spectral_fusion = "none"
    use_bidirectional_fie = False

    # Properly supervised building-separation head.
    use_instance_head = True
    instance_loss_weight = 0.30
    instance_w_center = 1.0
    instance_w_offset = 0.05
    instance_center_sigma = 6.0
    instance_target_use_watershed = True
    instance_target_min_peak_distance = 12
    instance_target_min_peak_height = 3.0
    instance_target_min_area = 64
    instance_center_threshold = 0.15
    instance_decode_min_distance = 12

@ex.named_config
def levir_final_rgb_osm_instance_safe_spectral():
    exp_name = "FINAL_rgb_osm_guided_instance_safe_spectral"

    # ------------------------------------------------------------------
    # OSM MUST remain present as semantic guidance.
    # ------------------------------------------------------------------
    # Use the real timestamped T2 OSM cache on the complete LEVIR benchmark.
    # Missing/empty OSM is masked; no source scene is deleted.
    levir_use_osm = False
    levir_osm_texts_json = ""
    levir_temporal_osm_enabled = True
    levir_temporal_osm_mode = "t2"
    levir_temporal_osm_manifest = "data_osm_t2_v2/manifest.json"
    levir_use_temporal_osm = True
    levir_require_osm_t1 = False
    levir_require_osm_t2 = False
    levir_require_paired_osm = False
    levir_require_nonempty_osm = False
    levir_filter_failed_osm = False

    # Disable the previously tested OSM injection paths that changed the RGB
    # learning trajectory.  The final architecture uses only SafeOSMLateGuidance.
    change_osm_mode = "none"
    change_use_osm_struct = False
    change_use_osm_t2_early = False
    change_use_osm_t2_late = False
    change_osm_t2_late_residual_scale = 0.0
    change_use_clip_filtered_osm = False
    change_clip_filter_runtime = False

    # Safe OSM guidance: always part of the architecture when T2 OSM is present,
    # but bounded, reliability-gated and applied only where RGB is uncertain.
    # The OSM auxiliary loss sees a detached RGB baseline and therefore cannot
    # backpropagate into ViT/FIE/semantic decoder.
    change_use_safe_osm_guidance = True
    change_safe_osm_hidden = 32
    change_safe_osm_max_prob_delta = 0.015
    change_safe_osm_uncertainty_power = 2.0
    change_safe_osm_loss_weight = 0.50
    change_safe_osm_safety_weight = 3.0
    change_safe_osm_l1_weight = 0.10
    change_safe_osm_init_seed = 1703

    # ------------------------------------------------------------------
    # Spectral: genuine cached NDVI/NDWI, but only as a safe late residual.
    # ------------------------------------------------------------------
    levir_spectral_indices_enabled = True
    levir_spectral_manifest = "data_spectral_v2/manifest.json"
    levir_use_spectral = True
    levir_require_spectral = False
    levir_require_indices_t1 = False
    levir_require_indices_t2 = False
    levir_accept_partial_indices = True
    levir_min_index_valid_fraction = 0.0
    levir_spectral_missing_policy = "mask"
    levir_auxiliary_policy = "mask"
    levir_allow_legacy_spectral_indices = True
    levir_index_normalization = "sensor_train_stats"
    levir_index_normalization_stats = "data_spectral_v2/train_sensor_stats.json"

    # Old early feature-level spectral fusion is abandoned in the final model.
    levir_spectral_fusion = "none"
    change_spectral_fusion = "none"

    change_use_safe_spectral_late = True
    change_safe_spectral_hidden = 24
    change_safe_spectral_max_prob_delta = 0.02
    change_safe_spectral_min_reliability = 0.50
    change_safe_spectral_uncertainty_power = 2.0
    change_safe_spectral_loss_weight = 1.0
    change_safe_spectral_safety_weight = 2.0
    change_safe_spectral_l1_weight = 0.05
    change_safe_spectral_init_seed = 1701

    # Combined OSM+spectral correction can never exceed 2.5 probability points.
    change_safe_aux_total_max_prob_delta = 0.025

    # ------------------------------------------------------------------
    # Building separation: retained, but isolated from semantic gradients.
    # ------------------------------------------------------------------
    use_instance_head = True
    instance_detach_shared_features = True
    instance_init_seed = 1702
    instance_loss_weight = 0.30
    instance_w_center = 1.0
    instance_w_offset = 0.05
    instance_center_sigma = 6.0
    instance_target_use_watershed = True
    instance_target_min_peak_distance = 12
    instance_target_min_peak_height = 3.0
    instance_target_min_area = 64
    instance_center_threshold = 0.15
    instance_decode_min_distance = 12

    # Branches that degraded/complicated the controlled runs stay inactive.
    use_bidirectional_fie = False
