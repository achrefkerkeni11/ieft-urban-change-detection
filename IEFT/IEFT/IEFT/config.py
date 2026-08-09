# -*- coding: utf-8 -*-
# Rollback stable v20d-compatible config generated from the user's provided config.
# Main change: task_levir_cd_v20_cliprank_vitb16_levir_dense_completion keeps CLIP runtime disabled
# because the stable v20d-compatible vilt_module.py does not inject CLIP into logits.

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
    vit_pretrained = True
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
    levir_train_positive_thr = 0.003
    levir_train_negative_mining_mode = "rural_structural"
    levir_train_hard_negative_gamma = 2.2
    levir_train_hard_negative_min_weight = 0.02
    levir_train_extreme_negative_prob = 0.10
    levir_train_extreme_negative_quantile = 0.88
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

    change_clip_filter_runtime = False
    clip_topk = 2
    clip_min_keep = 1
    clip_use_summary = True
    clip_use_main_text = True

    export_panel_title_bar_h = 40
    export_panel_font_size = 18


@ex.named_config
def task_levir_cd_v15_pixel_object_dense256():
    exp_name = "levir_cd_v15_pixel_object_dense256"
    datasets = ["levir_cd"]
    loss_names = _loss_names({"irtr": 0})

    image_size = 256
    model_input_size = 256
    patch_size = 16
    hidden_size = 384
    num_heads = 6
    num_layers = 4
    vit = "vit_small_patch16_224"

    batch_size = 8
    per_gpu_batchsize = 8
    num_workers = 0
    max_epoch = 40
    max_steps = 6000
    warmup_steps = 200

    encoder_learning_rate = 2e-5
    learning_rate = 3e-4
    weight_decay = 0.05

    levir_train_crop_size = 256
    levir_val_crop_size = 256
    levir_tile_size = 256
    levir_tile_stride = 256
    levir_train_repeat = 8
    levir_train_focus_positive = True
    levir_train_positive_focus_prob = 0.62
    levir_train_hard_negative_prob = 0.20
    levir_train_random_aug = True
    levir_label_smoothing = 0.01

    use_semantic_multiscale = True
    ms_change_num_levels = 4
    ms_change_level_indices = [2, 5, 8, 11]
    ms_change_decoder_dim = 160
    ms_change_dropout = 0.08

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


@ex.named_config
def task_levir_cd_v18_multimodal_clip_dense256_safe():
    exp_name = "levir_cd_v18_multimodal_clip_dense256_safe"
    datasets = ["levir_cd"]
    loss_names = _loss_names({"irtr": 0})

    image_size = 256
    model_input_size = 256
    patch_size = 16
    hidden_size = 384
    num_heads = 6
    num_layers = 4
    vit = "vit_small_patch16_224"
    vit_pretrained = True

    batch_size = 8
    per_gpu_batchsize = 8
    num_workers = 0
    max_epoch = 40
    max_steps = 6000
    warmup_steps = 200

    encoder_learning_rate = 2e-5
    learning_rate = 3e-4
    weight_decay = 0.05

    levir_train_crop_size = 256
    levir_val_crop_size = 256
    levir_tile_size = 256
    levir_tile_stride = 256
    levir_train_repeat = 8
    levir_train_focus_positive = True
    levir_train_positive_focus_prob = 0.62
    levir_train_hard_negative_prob = 0.20
    levir_train_random_aug = True
    levir_label_smoothing = 0.01

    use_semantic_multiscale = True
    ms_change_num_levels = 4
    ms_change_level_indices = [2, 5, 8, 11]
    ms_change_decoder_dim = 160
    ms_change_dropout = 0.08

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

    levir_use_osm = True
    levir_osm_texts_json = "data_osm\\osm_texts_by_patch_levir_v21.json"
    levir_osm_text_mode = "concat"
    levir_osm_max_phrases = 3
    levir_osm_text_key = "text_v21"
    levir_osm_fallback_text = "no_osm_context"
    levir_osm_compose_mode = "signature_compact"
    levir_osm_word_budget = 32
    levir_osm_joiner = " ; "
    levir_osm_include_source_text = False

    change_use_osm_struct = True
    change_osm_struct_dim = 16
    change_osm_struct_hidden = 128
    change_osm_global_weight = 0.18
    change_osm_coarse_weight = 0.14
    change_osm_dense_weight = 0.10
    change_osm_patch_weight = 0.18
    change_osm_gate_floor = 0.18
    change_osm_reliability_bias = 0.05
    change_osm_use_text = False

    change_clip_filter_runtime = False
    clip_topk = 2
    clip_min_keep = 1
    clip_use_summary = True
    clip_use_main_text = True


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
