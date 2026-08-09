
import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from IEFT.datasets.levir_cd_dataset import LEVIRCDDataset


class LEVIRCDDataModule(pl.LightningDataModule):
    def __init__(self, _config, dist: bool = False):
        super().__init__()
        self.config = dict(_config)
        self.dist = bool(dist)

        self.data_root = self.config.get("data_root", "")
        self.batch_size = int(self.config.get("batch_size", 8))
        self.num_workers = int(self.config.get("num_workers", 0))
        self.image_size = int(self.config.get("image_size", 256))

        self.crop_size_train = int(self.config.get("levir_train_crop_size", self.image_size))
        self.crop_size_val = int(self.config.get("levir_val_crop_size", self.image_size))
        self.tile_stride = int(self.config.get("levir_tile_stride", self.crop_size_train))
        self.train_repeat = int(self.config.get("levir_train_repeat", 1))

        self.focus_positive = bool(self.config.get("levir_train_focus_positive", True))
        self.positive_focus_prob = float(self.config.get("levir_train_positive_focus_prob", 0.58))
        self.hard_negative_prob = float(self.config.get("levir_train_hard_negative_prob", 0.20))
        self.random_aug = bool(self.config.get("levir_train_random_aug", True))
        self.label_smoothing = float(self.config.get("levir_label_smoothing", 0.0))

        self.max_text_len = int(self.config.get("max_text_len", 40))
        self.fixed_text = str(self.config.get("levir_fixed_text", "building change detection"))
        self.mask_threshold = int(self.config.get("levir_mask_threshold", 127))

        self.a_dirname = str(self.config.get("levir_image_a_dirname", "A"))
        self.b_dirname = str(self.config.get("levir_image_b_dirname", "B"))
        self.label_dirname = str(self.config.get("levir_label_dirname", "label"))

        self.levir_use_osm = bool(self.config.get("levir_use_osm", False))
        self.levir_osm_texts_json = str(self.config.get("levir_osm_texts_json", ""))
        self.levir_osm_text_mode = str(self.config.get("levir_osm_text_mode", "concat"))
        self.levir_osm_max_phrases = int(self.config.get("levir_osm_max_phrases", 3))
        self.levir_osm_text_key = str(self.config.get("levir_osm_text_key", "text_v21"))
        self.levir_osm_fallback_text = str(self.config.get("levir_osm_fallback_text", "no_osm_context"))
        self.levir_osm_compose_mode = str(self.config.get("levir_osm_compose_mode", "signature_compact"))
        self.levir_osm_word_budget = int(self.config.get("levir_osm_word_budget", 32))
        self.levir_osm_joiner = str(self.config.get("levir_osm_joiner", " ; "))
        self.levir_osm_include_source_text = bool(self.config.get("levir_osm_include_source_text", False))

        tok_name = self.config.get("tokenizer", "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", 30522))
        self.mlm_collator = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        train_root = os.path.join(self.data_root, "train")
        if not os.path.isdir(train_root):
            raise FileNotFoundError(f"Missing LEVIR train split: {train_root}")

    def _make_ds(self, split: str, crop_size: int, stride: int, train_repeat: int, train_focus_positive: bool,
                 positive_focus_prob: float, hard_negative_prob: float, random_aug: bool, label_smoothing: float):
        return LEVIRCDDataset(
            root=self.data_root,
            split=split,
            image_size=self.image_size,
            crop_size=crop_size,
            tile_stride=stride,
            train_repeat=train_repeat,
            train_focus_positive=train_focus_positive,
            positive_focus_prob=positive_focus_prob,
            hard_negative_prob=hard_negative_prob,
            random_aug=random_aug,
            label_smoothing=label_smoothing,
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            fixed_text=self.fixed_text,
            mask_threshold=self.mask_threshold,
            a_dirname=self.a_dirname,
            b_dirname=self.b_dirname,
            label_dirname=self.label_dirname,
            use_osm=self.levir_use_osm,
            osm_texts_json=self.levir_osm_texts_json,
            osm_text_mode=self.levir_osm_text_mode,
            osm_max_phrases=self.levir_osm_max_phrases,
            osm_text_key=self.levir_osm_text_key,
            osm_fallback_text=self.levir_osm_fallback_text,
            osm_compose_mode=self.levir_osm_compose_mode,
            osm_word_budget=self.levir_osm_word_budget,
            osm_joiner=self.levir_osm_joiner,
            osm_include_source_text=self.levir_osm_include_source_text,
        )

    def setup(self, stage: Optional[str] = None):
        self.prepare_data()
        if stage in (None, "fit"):
            self.train_dataset = self._make_ds(
                split="train",
                crop_size=self.crop_size_train,
                stride=self.tile_stride,
                train_repeat=self.train_repeat,
                train_focus_positive=self.focus_positive,
                positive_focus_prob=self.positive_focus_prob,
                hard_negative_prob=self.hard_negative_prob,
                random_aug=self.random_aug,
                label_smoothing=self.label_smoothing,
            )
            self.val_dataset = self._make_ds(
                split="val",
                crop_size=self.crop_size_val,
                stride=self.crop_size_val,
                train_repeat=1,
                train_focus_positive=False,
                positive_focus_prob=0.0,
                hard_negative_prob=0.0,
                random_aug=False,
                label_smoothing=0.0,
            )
        if stage in (None, "test"):
            self.test_dataset = self._make_ds(
                split="test",
                crop_size=self.crop_size_val,
                stride=self.crop_size_val,
                train_repeat=1,
                train_focus_positive=False,
                positive_focus_prob=0.0,
                hard_negative_prob=0.0,
                random_aug=False,
                label_smoothing=0.0,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False)
