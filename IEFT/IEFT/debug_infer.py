import torch

from IEFT.modules.vilt_module import ViLTransformerSS
from IEFT.datamodules.s2_npz_datamodule import S2NPZDataModule

cfg = {
    "data_root": r"data_npz",
    "batch_size": 1,
    "per_gpu_batchsize": 1,
    "num_workers": 0,
    "image_size": 384,
    "max_text_len": 40,
    "draw_false_image": 1,
    "draw_false_text": 2,
    "tokenizer": "bert-base-uncased",

    "seed": 0,
    "num_gpus": 0,
    "num_nodes": 1,
    "precision": 32,
    "fast_dev_run": False,
    "resume_from": None,
    "test_only": False,
    "get_recall_metric": False,

    "vit": "vit_base_patch32_384",
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "drop_rate": 0.1,
    "max_image_len": 144,

    # ✅ checkpoint actuel
    "load_path": r"result\finetune_irtr_rsicd_randaug_seed0_from_vilt_200k_mlm_itm\version_11\checkpoints\last.ckpt",

    "vqav2_label_size": 3129,
    "exp_name": "debug",

    "loss_names": {
        "mlm": 0,
        "itm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 1,
    },

    # ✅ active la pseudo change loss dans le modèle
    "change_loss_weight": 1.0,
}

print("[1] Build datamodule")
dm = S2NPZDataModule(cfg)
dm.setup()

# récupérer vocab_size depuis le tokenizer du datamodule
cfg["vocab_size"] = dm.vocab_size

print("[2] Build model")
model = ViLTransformerSS(cfg)
model.eval()

print("[3] Get batch")
batch = next(iter(dm.train_dataloader()))

print("[4] Infer")
with torch.no_grad():
    out = model.infer(batch)

print("out keys =", out.keys())
print("text_feats =", out["text_feats"].shape)
print("image_feats =", out["image_feats"].shape)

if out["image_t1_feats"] is not None:
    print("image_t1_feats =", out["image_t1_feats"].shape)
if out["temp_feats"] is not None:
    print("temp_feats =", out["temp_feats"].shape)
if out["image_t2_feats"] is not None:
    print("image_t2_feats =", out["image_t2_feats"].shape)

if out["change_logits"] is not None:
    print("change_logits =", out["change_logits"].shape)
if out["change_probs_fused"] is not None:
    print("change_probs_fused =", out["change_probs_fused"].shape)
if out["change_map"] is not None:
    print("change_map =", out["change_map"].shape)

print("\nALL OK")