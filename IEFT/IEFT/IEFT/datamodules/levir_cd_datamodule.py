
import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from IEFT.datasets.levir_auxiliary import build_levir_tokenizer
from IEFT.datasets.levir_cd_dataset import LEVIRCDDataset


def _first_config(config, names, default=None):
    for name in names:
        value = config.get(name)
        if value is not None and value != "":
            return value
    return default


class LEVIRCDDataModule(pl.LightningDataModule):
    def __init__(self, _config, dist: bool = False):
        super().__init__()
        self.config = dict(_config)
        self.dist = bool(dist)

        self.data_root = self.config.get("data_root", "")
        # DataLoader uses the physical per-device batch. The training runner
        # preserves the requested effective global batch via accumulation.
        self.batch_size = int(
            self.config.get("per_gpu_batchsize", self.config.get("batch_size", 8))
        )
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

        self.levir_osm_manifest_json = str(
            _first_config(
                self.config,
                (
                    "levir_temporal_osm_manifest",
                    "levir_osm_manifest_json",
                    "levir_osm_manifest",
                    "osm_manifest_json",
                ),
                "",
            )
        )
        self.levir_spectral_manifest_json = str(
            _first_config(
                self.config,
                (
                    "levir_spectral_manifest",
                    "levir_spectral_manifest_json",
                    "spectral_manifest_json",
                ),
                "",
            )
        )
        spectral_policy = _first_config(
            self.config,
            ("levir_spectral_missing_policy",),
            None,
        )
        if spectral_policy is not None:
            spectral_policy = str(spectral_policy).strip().lower()
            if spectral_policy not in {"filter", "mask"}:
                raise ValueError("levir_spectral_missing_policy must be 'filter' or 'mask'")
            self.levir_auxiliary_policy = "strict" if spectral_policy == "filter" else "mask"
        else:
            self.levir_auxiliary_policy = str(
                _first_config(
                    self.config,
                    ("levir_auxiliary_policy", "levir_aux_policy", "auxiliary_policy"),
                    "mask",
                )
            )

        osm_enabled_is_canonical = "levir_temporal_osm_enabled" in self.config
        spectral_enabled_is_canonical = "levir_spectral_indices_enabled" in self.config
        if osm_enabled_is_canonical:
            self.levir_use_temporal_osm = bool(self.config["levir_temporal_osm_enabled"])
        else:
            legacy_use = _first_config(
                self.config, ("levir_use_temporal_osm", "levir_use_paired_osm"), None
            )
            self.levir_use_temporal_osm = (
                bool(self.levir_osm_manifest_json) if legacy_use is None else bool(legacy_use)
            )
        if spectral_enabled_is_canonical:
            self.levir_use_spectral = bool(self.config["levir_spectral_indices_enabled"])
        else:
            legacy_use = _first_config(
                self.config, ("levir_use_spectral", "change_use_spectral"), None
            )
            self.levir_use_spectral = (
                bool(self.levir_spectral_manifest_json) if legacy_use is None else bool(legacy_use)
            )

        temporal_osm_mode = _first_config(
            self.config, ("levir_temporal_osm_mode",), None
        )
        if temporal_osm_mode is None:
            temporal_osm_mode = "paired" if self.levir_use_temporal_osm else "none"
        temporal_osm_mode = str(temporal_osm_mode).strip().lower().replace("-", "_")
        temporal_osm_mode = {
            "off": "none",
            "disabled": "none",
            "temporal": "paired",
            "paired_temporal": "paired",
            "t2_only": "t2",
            "single_t2": "t2",
        }.get(temporal_osm_mode, temporal_osm_mode)
        if temporal_osm_mode not in {"none", "paired", "t2"}:
            raise ValueError(
                "levir_temporal_osm_mode must be one of 'none', 'paired', or 't2'"
            )
        if self.levir_use_temporal_osm and temporal_osm_mode == "none":
            raise ValueError(
                "levir_temporal_osm_enabled=True requires a non-'none' "
                "levir_temporal_osm_mode"
            )
        self.levir_temporal_osm_mode = temporal_osm_mode

        legacy_require_osm = _first_config(
            self.config,
            ("levir_require_paired_osm", "levir_filter_require_osm"),
            None,
        )
        legacy_require_spectral = _first_config(
            self.config,
            ("levir_require_spectral", "levir_filter_require_spectral"),
            None,
        )
        if osm_enabled_is_canonical:
            require_t1_by_default = self.levir_temporal_osm_mode != "t2"
            self.levir_require_osm_t1 = self.levir_use_temporal_osm and bool(
                self.config.get("levir_require_osm_t1", require_t1_by_default)
            )
            self.levir_require_osm_t2 = self.levir_use_temporal_osm and bool(
                self.config.get("levir_require_osm_t2", True)
            )
            # Legacy paired requirement remains useful for matched-subset RGB
            # ablations where the modality itself is deliberately disabled.
            if not self.levir_use_temporal_osm and legacy_require_osm is not None:
                self.levir_require_osm_t1 = bool(legacy_require_osm)
                self.levir_require_osm_t2 = bool(legacy_require_osm)
        else:
            default = bool(self.levir_osm_manifest_json) if legacy_require_osm is None else bool(legacy_require_osm)
            self.levir_require_osm_t1 = default
            self.levir_require_osm_t2 = default

        if spectral_enabled_is_canonical:
            self.levir_require_indices_t1 = self.levir_use_spectral and bool(
                self.config.get("levir_require_indices_t1", True)
            )
            self.levir_require_indices_t2 = self.levir_use_spectral and bool(
                self.config.get("levir_require_indices_t2", True)
            )
            if not self.levir_use_spectral and legacy_require_spectral is not None:
                self.levir_require_indices_t1 = bool(legacy_require_spectral)
                self.levir_require_indices_t2 = bool(legacy_require_spectral)
        else:
            default = (
                bool(self.levir_spectral_manifest_json)
                if legacy_require_spectral is None
                else bool(legacy_require_spectral)
            )
            self.levir_require_indices_t1 = default
            self.levir_require_indices_t2 = default

        self.levir_require_paired_osm = bool(
            self.levir_require_osm_t1 and self.levir_require_osm_t2
        )
        self.levir_require_spectral = bool(
            self.levir_require_indices_t1 and self.levir_require_indices_t2
        )
        self.levir_require_nonempty_osm = bool(
            self.config.get("levir_require_nonempty_osm", False)
        )
        self.levir_filter_failed_osm = bool(
            self.config.get("levir_filter_failed_osm", True)
        )
        self.levir_osm_timestamp_policy = str(
            self.config.get("levir_osm_timestamp_policy", "")
        )
        self.levir_accept_partial_indices = bool(
            self.config.get("levir_accept_partial_indices", True)
        )
        self.levir_allow_legacy_spectral_indices = bool(
            self.config.get("levir_allow_legacy_spectral_indices", False)
        )
        self.levir_spectral_min_valid_fraction = float(
            _first_config(
                self.config,
                (
                    "levir_min_index_valid_fraction",
                    "levir_spectral_min_valid_fraction",
                    "spectral_min_valid_fraction",
                ),
                0.0,
            )
        )
        self.levir_spectral_cache_size = int(self.config.get("levir_spectral_cache_size", 2))
        self.levir_osm_cache_size = int(
            _first_config(
                self.config,
                ("levir_osm_cache_size", "levir_temporal_osm_cache_size"),
                1,
            )
        )
        self.levir_index_normalization = str(
            self.config.get("levir_index_normalization", "natural")
        )
        self.levir_index_normalization_stats = str(
            self.config.get("levir_index_normalization_stats", "")
        )

        self.levir_tokenizer_mode = str(self.config.get("levir_tokenizer_mode", "simple"))
        self.levir_tokenizer_local_only = bool(self.config.get("levir_tokenizer_local_only", True))
        self.levir_tokenizer_fallback_to_simple = bool(
            self.config.get("levir_tokenizer_fallback_to_simple", False)
        )

        tok_name = self.config.get("tokenizer", "bert-base-uncased")
        self.tokenizer = build_levir_tokenizer(
            mode=self.levir_tokenizer_mode,
            name=tok_name,
            local_files_only=self.levir_tokenizer_local_only,
            fallback_to_simple=self.levir_tokenizer_fallback_to_simple,
            vocab_size=int(self.config.get("vocab_size", 30522)),
        )
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", 30522))
        self.mlm_collator = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.auxiliary_filtering_stats = {}

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
            osm_manifest_json=self.levir_osm_manifest_json,
            spectral_manifest_json=self.levir_spectral_manifest_json,
            auxiliary_policy=self.levir_auxiliary_policy,
            require_paired_osm=(
                None if self.levir_require_paired_osm is None else bool(self.levir_require_paired_osm)
            ),
            require_spectral=(
                None if self.levir_require_spectral is None else bool(self.levir_require_spectral)
            ),
            require_osm_t1=self.levir_require_osm_t1,
            require_osm_t2=self.levir_require_osm_t2,
            require_indices_t1=self.levir_require_indices_t1,
            require_indices_t2=self.levir_require_indices_t2,
            accept_partial_indices=self.levir_accept_partial_indices,
            require_nonempty_osm=self.levir_require_nonempty_osm,
            filter_failed_osm=self.levir_filter_failed_osm,
            osm_timestamp_policy=self.levir_osm_timestamp_policy,
            temporal_osm_mode=(
                self.levir_temporal_osm_mode
                if self.levir_temporal_osm_mode != "none"
                else "paired"
            ),
            use_temporal_osm=self.levir_use_temporal_osm,
            use_spectral=self.levir_use_spectral,
            allow_legacy_spectral_indices=self.levir_allow_legacy_spectral_indices,
            spectral_min_valid_fraction=self.levir_spectral_min_valid_fraction,
            spectral_cache_size=self.levir_spectral_cache_size,
            osm_cache_size=self.levir_osm_cache_size,
            index_normalization=self.levir_index_normalization,
            index_normalization_stats=self.levir_index_normalization_stats,
            tokenizer_mode=self.levir_tokenizer_mode,
            tokenizer_name=str(self.config.get("tokenizer", "bert-base-uncased")),
            tokenizer_local_only=self.levir_tokenizer_local_only,
            tokenizer_fallback_to_simple=self.levir_tokenizer_fallback_to_simple,
            tokenizer_vocab_size=int(self.config.get("vocab_size", 30522)),
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
        self.auxiliary_filtering_stats = {
            split: {
                **dataset.filtering_stats,
                "temporal_osm_mode": self.levir_temporal_osm_mode,
            }
            for split, dataset in (
                ("train", self.train_dataset),
                ("val", self.val_dataset),
                ("test", self.test_dataset),
            )
            if dataset is not None
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=LEVIRCDDataset.collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=LEVIRCDDataset.collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=LEVIRCDDataset.collate)
