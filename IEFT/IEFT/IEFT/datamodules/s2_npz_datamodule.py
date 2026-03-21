import os
from typing import Optional, List

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from IEFT.datasets.s2_npz_dataset import S2NPZDataset


def _list_npz_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".npz")]
        files.sort()
        return files
    return []


class S2NPZDataModule:
    """
    Charge des NPZ shards:
      data_root/train/*.npz
      data_root/val/*.npz
      data_root/test/*.npz
    """

    def __init__(self, _config, dist: bool = False):
        self.config = dict(_config)

        self.data_root = self.config.get("data_root", "")
        self.batch_size = int(self.config.get("batch_size", 1))
        self.num_workers = int(self.config.get("num_workers", 0))
        self.image_size = int(self.config.get("image_size", 224))
        self.max_text_len = int(self.config.get("max_text_len", 40))

        self.draw_false_image = int(self.config.get("draw_false_image", 1))
        self.draw_false_text = int(self.config.get("draw_false_text", 15))

        self.s2_scale_div = float(self.config.get("s2_scale_div", 10000.0))

        self.osm_texts_json = str(self.config.get("osm_texts_json", "")).strip()
        self.osm_text_mode = str(self.config.get("osm_text_mode", "concat")).strip().lower()
        self.osm_max_phrases = int(self.config.get("osm_max_phrases", 3))

        tok_name = self.config.get("tokenizer", "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", 30522))

        self.mlm_collator = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        train_dir = os.path.join(self.data_root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Structure attendue:\n"
                f"  {self.data_root}\\train\\*.npz\n"
                f"  {self.data_root}\\val\\*.npz\n"
                f"  {self.data_root}\\test\\*.npz\n"
            )
        train_files = _list_npz_files(train_dir)
        if len(train_files) == 0:
            raise FileNotFoundError(f"Aucun .npz trouvé dans {train_dir}")

    def _build_dataset(self, npz_paths: List[str]) -> S2NPZDataset:
        return S2NPZDataset(
            npz_paths=npz_paths,
            image_size=self.image_size,
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            rgb_from="T2",
            s2_scale_div=self.s2_scale_div,
            osm_texts_json=self.osm_texts_json,
            osm_text_mode=self.osm_text_mode,
            osm_max_phrases=self.osm_max_phrases,
        )

    def setup(self, stage: Optional[str] = None):
        self.prepare_data()

        train_files = _list_npz_files(os.path.join(self.data_root, "train"))
        val_files = _list_npz_files(os.path.join(self.data_root, "val"))
        test_files = _list_npz_files(os.path.join(self.data_root, "test"))

        if len(val_files) == 0:
            val_files = train_files
        if len(test_files) == 0:
            test_files = train_files

        self.train_dataset = self._build_dataset(train_files)
        self.val_dataset = self._build_dataset(val_files)
        self.test_dataset = self._build_dataset(test_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        return self.train_dataset.collate(batch, mlm_collator=self.mlm_collator)