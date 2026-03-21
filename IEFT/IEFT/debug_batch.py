from IEFT.datamodules.s2_npz_datamodule import S2NPZDataModule

cfg = {
    "data_root": r"data_npz",
    "batch_size": 1,
    "num_workers": 0,
    "image_size": 384,
    "max_text_len": 40,
    "draw_false_image": 1,
    "draw_false_text": 2,
    "tokenizer": "bert-base-uncased",
}

print("[1] Build datamodule")
dm = S2NPZDataModule(cfg)

print("[2] Setup")
dm.setup()

print("[3] train_dataset =", type(dm.train_dataset), "len =", len(dm.train_dataset))
print("[4] val_dataset   =", type(dm.val_dataset), "len =", len(dm.val_dataset))
print("[5] test_dataset  =", type(dm.test_dataset), "len =", len(dm.test_dataset))

print("[6] Build dataloader")
dl = dm.train_dataloader()
print("[7] dataloader =", type(dl))

print("[8] Get first raw sample")
sample0 = dm.train_dataset[0]
print("sample0 keys =", sample0.keys())
print("sample0 rgb_t1 shape =", sample0["rgb_t1"].shape)
print("sample0 rgb_t2 shape =", sample0["rgb_t2"].shape)
print("sample0 x8 shape =", sample0["x8"].shape)

print("[9] Get first batch")
b = next(iter(dl))
print("batch type =", type(b))
print("batch =", b)

if b is None:
    raise RuntimeError("Batch is None -> dataset.collate() returned None")

print("[10] batch keys =", b.keys())
print("image    =", b["image"][0].shape)
print("image_t1 =", b["image_t1"][0].shape)
print("image_t2 =", b["image_t2"][0].shape)
print("x8       =", b["x8"].shape)

print("\nALL OK")
