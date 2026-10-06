"""LEVIR-CD datasets exposed by the maintained pipeline."""

from .levir_cd_dataset import LEVIRCDDataset

# Preserve the historical mixed-case import while exporting the real class.
LevirCDDataset = LEVIRCDDataset

__all__ = ["LEVIRCDDataset", "LevirCDDataset"]
