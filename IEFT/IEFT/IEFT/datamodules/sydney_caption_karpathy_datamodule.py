# IEFT/datamodules/sydney_caption_karpathy_datamodule.py
from .datamodule_base import BaseDataModule
from IEFT.datasets import SydneyCaptionKarpathyDataset


class SydneyCaptionKarpathyDataModule(BaseDataModule):
    @property
    def dataset_cls(self):
        return SydneyCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "sydney"
