import run
from IEFT.config import ex

if __name__ == "__main__":
    cfg = {
        "exp_name": "finetune_irtr_rsicd_randaug",
        "datasets": ["s2_npz"],
        "loss_names": {
            "itm": 0,
            "mlm": 0,
            "mpp": 0,
            "vqa": 0,
            "nlvr2": 0,
            "irtr": 1,
        },
        "data_root": "data_npz",
        "load_path": r"result\finetune_irtr_rsicd_randaug_seed0_from_last\version_15\checkpoints\last.ckpt",
        "osm_texts_json": r"data_osm\osm_texts_by_patch_v21_compact_signature_refined.json",
        "osm_text_mode": "first",
        "osm_max_phrases": 1,
        "max_text_len": 40,
        "draw_false_text": 2,
        "change_loss_weight": 1.0,
        "batch_size": 1,
        "per_gpu_batchsize": 1,
        "num_workers": 0,
        "num_gpus": 1,
        "num_nodes": 1,
        "max_epoch": 1,
        "max_steps": 100,
        "warmup_steps": 0,
        "get_recall_metric": False,
        "test_only": False,
        "fast_dev_run": False,
    }
    ex.run(config_updates=cfg)
