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
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # desired effective batch size

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # OSM / external text setting
    osm_texts_json = ""
    osm_text_mode = "concat"   # first | random | concat
    osm_max_phrases = 3

    # Transformer setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1

    # Change detection setting
    s2_scale_div = 10000.0
    change_loss_weight = 1.0
    change_global_loss_weight = 0.20
    change_smoothness_loss_weight = 0.05
    change_pos_quantile = 0.90
    change_neg_quantile = 0.35

    # Downstream setting
    get_recall_metric = False

    # PL trainer setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # Environment-dependent params
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    json = "/home/amax/wyj/dataset/RSITMD/dataset_RSITMD.json"


@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 1
    num_nodes = 1


@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k_randaug():
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_sydney_randaug():
    exp_name = "finetune_irtr_sydney_randaug"
    datasets = ["sydney"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 512
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4
    json = "/home/amax/wyj/dataset/Sydney_captions/karpathy/dataset.json"


@ex.named_config
def task_finetune_irtr_ucm_randaug():
    exp_name = "finetune_irtr_ucm_randaug"
    datasets = ["ucm"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4
    json = "/home/amax/wyj/dataset/UCM_captions/karpathy/dataset.json"


@ex.named_config
def task_finetune_irtr_rsicd_randaug():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsicd"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4
    json = "/home/amax/wyj/dataset/RSICD_captions/karpathy/dataset_rsicd.json"


@ex.named_config
def task_finetune_irtr_rsitmd_randaug():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["rsitmd"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4
    json = "/home/amax/wyj/dataset/RSITMD/dataset_RSITMD.json"


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12


@ex.named_config
def task_smoke_s2_npz():
    exp_name = "smoke_s2_npz"
    datasets = ["s2_npz"]
    data_root = "data_npz"

    per_gpu_batchsize = 1
    batch_size = 1
    num_workers = 0
    num_gpus = 1
    num_nodes = 1

    max_epoch = 1
    max_steps = 5
    warmup_steps = 0
    get_recall_metric = False

    load_path = "weights/vilt_200k_mlm_itm.ckpt"

    image_size = 384
    vit = "vit_base_patch32_384"
    patch_size = 32

    train_transform_keys = ["pixelbert_randaug"]
    val_transform_keys = ["pixelbert"]
@ex.named_config
def task_finetune_s2_npz_irtr_osm():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["s2_npz"]
    data_root = "data_npz"

    loss_names = _loss_names({"irtr": 1})
    get_recall_metric = False

    per_gpu_batchsize = 1
    batch_size = 1
    num_workers = 0
    num_gpus = 1
    num_nodes = 1

    max_epoch = 20
    max_steps = 5000
    warmup_steps = 0

    draw_false_image = 1
    draw_false_text = 2

    image_size = 384
    vit = "vit_base_patch32_384"
    patch_size = 32
    train_transform_keys = ["pixelbert_randaug"]
    val_transform_keys = ["pixelbert"]

    max_text_len = 40
@ex.named_config
def task_s2_npz_irtr_v14():
    exp_name = "finetune_irtr_rsicd_randaug"
    datasets = ["s2_npz"]

    train_transform_keys = ["pixelbert_randaug"]
    val_transform_keys = ["pixelbert"]

    loss_names = _loss_names({"irtr": 1})

    data_root = "data_npz"

    per_gpu_batchsize = 1
    batch_size = 1
    num_workers = 0
    num_gpus = 1
    num_nodes = 1

    max_epoch = 1
    max_steps = 100
    warmup_steps = 0

    get_recall_metric = False
    draw_false_text = 2

    image_size = 384
    vit = "vit_base_patch32_384"
    patch_size = 32

    max_text_len = 40
    osm_text_mode = "first"
    osm_max_phrases = 1

    change_loss_weight = 1.0
    