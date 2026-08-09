
from typing import List

import pytorch_lightning as pl

from IEFT.datamodules.levir_cd_datamodule import LEVIRCDDataModule


class MTDataModule(pl.LightningDataModule):
    def __init__(self, _config, dist: bool = False):
        super().__init__()
        self.config = dict(_config)
        self.dist = bool(dist)

        dataset_names = self.config.get("datasets", [])
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = list(dataset_names)

        self.dm_dicts = {
            "levir_cd": LEVIRCDDataModule,
        }

        missing = [k for k in self.dataset_names if k not in self.dm_dicts]
        if missing:
            raise KeyError(
                f"Datamodules introuvables pour: {missing}\n"
                f"Disponibles: {sorted(self.dm_dicts.keys())}"
            )

        self.dms: List[pl.LightningDataModule] = [self.dm_dicts[k](self.config, dist=self.dist) for k in self.dataset_names]
        if len(self.dms) == 0:
            raise RuntimeError("Aucun datamodule actif.")
        self.dm = self.dms[0]

        self.tokenizer = getattr(self.dm, "tokenizer", None)
        self.vocab_size = getattr(self.dm, "vocab_size", 30522)
        self.mlm_collator = getattr(self.dm, "mlm_collator", None)

    def prepare_data(self):
        self.dm.prepare_data()

    def setup(self, stage=None):
        self.dm.setup(stage=stage)
        self.tokenizer = getattr(self.dm, "tokenizer", self.tokenizer)
        self.vocab_size = getattr(self.dm, "vocab_size", self.vocab_size)
        self.mlm_collator = getattr(self.dm, "mlm_collator", self.mlm_collator)

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()
