from typing import Dict, List, Optional

from pytorch_lightning import LightningDataModule

from IEFT.datamodules.coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from IEFT.datamodules.conceptual_caption_datamodule import ConceptualCaptionDataModule
from IEFT.datamodules.f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from IEFT.datamodules.nlvr2_datamodule import NLVR2DataModule
from IEFT.datamodules.rsicd_caption_karpathy_datamodule import RSICDCaptionKarpathyDataModule
from IEFT.datamodules.rsitmd_caption_karpathy_datamodule import RSITMDCaptionKarpathyDataModule
from IEFT.datamodules.s2_npz_datamodule import S2NPZDataModule
from IEFT.datamodules.sbu_datamodule import SBUCaptionDataModule
from IEFT.datamodules.sydney_caption_karpathy_datamodule import SydneyCaptionKarpathyDataModule
from IEFT.datamodules.ucm_caption_karpathy_datamodule import UCMCaptionKarpathyDataModule
from IEFT.datamodules.vg_caption_datamodule import VisualGenomeCaptionDataModule
from IEFT.datamodules.vqav2_datamodule import VQAv2DataModule
from IEFT.datamodules.levir_cd_datamodule import LEVIRCDDataModule


class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist: bool = False):
        super().__init__()
        self.config = dict(_config)
        self.dist = bool(dist)
        self.dataset_names: List[str] = list(self.config.get("datasets", []))
        if len(self.dataset_names) == 0:
            raise ValueError("La config doit contenir au moins un dataset dans `datasets`.")

        self.dm_dicts: Dict[str, type] = {
            "coco": CocoCaptionKarpathyDataModule,
            "f30k": F30KCaptionKarpathyDataModule,
            "gcc": ConceptualCaptionDataModule,
            "nlvr2": NLVR2DataModule,
            "rsicd": RSICDCaptionKarpathyDataModule,
            "rsitmd": RSITMDCaptionKarpathyDataModule,
            "s2_npz": S2NPZDataModule,
            "sbu": SBUCaptionDataModule,
            "sydney": SydneyCaptionKarpathyDataModule,
            "ucm": UCMCaptionKarpathyDataModule,
            "vg": VisualGenomeCaptionDataModule,
            "vqa": VQAv2DataModule,
            "levir_cd": LEVIRCDDataModule,
        }

        missing = [k for k in self.dataset_names if k not in self.dm_dicts]
        if missing:
            raise KeyError(
                f"Datamodules introuvables pour: {missing}\nDisponibles: {sorted(self.dm_dicts.keys())}"
            )

        self.dms: List[LightningDataModule] = [self.dm_dicts[k](self.config, dist=self.dist) for k in self.dataset_names]

        ref_dm = self.dms[0]
        self.tokenizer = getattr(ref_dm, "tokenizer", None)
        self.vocab_size = int(getattr(ref_dm, "vocab_size", self.config.get("vocab_size", 30522)))
        self.mlm_collator = getattr(ref_dm, "mlm_collator", None)

    def prepare_data(self):
        for dm in self.dms:
            if hasattr(dm, "prepare_data"):
                dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        for dm in self.dms:
            dm.setup(stage)
        ref_dm = self.dms[0]
        self.tokenizer = getattr(ref_dm, "tokenizer", self.tokenizer)
        self.vocab_size = int(getattr(ref_dm, "vocab_size", self.vocab_size))
        self.mlm_collator = getattr(ref_dm, "mlm_collator", self.mlm_collator)

    def _single(self):
        if len(self.dms) != 1:
            raise NotImplementedError(
                "Cette version de MTDataModule supporte seulement un dataset actif à la fois. "
                f"datasets={self.dataset_names}"
            )
        return self.dms[0]

    def train_dataloader(self):
        return self._single().train_dataloader()

    def val_dataloader(self):
        return self._single().val_dataloader()

    def test_dataloader(self):
        return self._single().test_dataloader()
