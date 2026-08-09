import warnings

_datamodules = {}

def _safe_add(key: str, cls):
    _datamodules[key] = cls

def _safe_import(module_path: str, class_name: str, key: str):
    """
    Import a datamodule safely. If it fails, we keep going so other datamodules work.
    """
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        _safe_add(key, cls)
    except Exception as e:
        warnings.warn(f"[IEFT.datamodules] Skipping '{key}' because import failed: {e}")

# --- core datamodules ---
_safe_import("IEFT.datamodules.vg_caption_datamodule", "VisualGenomeCaptionDataModule", "vg")
_safe_import("IEFT.datamodules.f30k_caption_karpathy_datamodule", "F30KCaptionKarpathyDataModule", "f30k")
_safe_import("IEFT.datamodules.coco_caption_karpathy_datamodule", "CocoCaptionKarpathyDataModule", "coco")
_safe_import("IEFT.datamodules.conceptual_caption_datamodule", "ConceptualCaptionDataModule", "gcc")
_safe_import("IEFT.datamodules.sbu_datamodule", "SBUCaptionDataModule", "sbu")
_safe_import("IEFT.datamodules.vqav2_datamodule", "VQAv2DataModule", "vqa")
_safe_import("IEFT.datamodules.nlvr2_datamodule", "NLVR2DataModule", "nlvr2")
_safe_import("IEFT.datamodules.rsicd_caption_karpathy_datamodule", "RSICDCaptionKarpathyDataModule", "rsicd")
_safe_import("IEFT.datamodules.ucm_caption_karpathy_datamodule", "UCMCaptionKarpathyDataModule", "ucm")
_safe_import("IEFT.datamodules.rsitmd_caption_karpathy_datamodule", "RSITMDCaptionKarpathyDataModule", "rsitmd")

# --- sydney (optional) ---
_safe_import("IEFT.datamodules.sydney_caption_karpathy_datamodule", "SydneyCaptionKarpathyDataModule", "sydney")

# --- current Sentinel-2 datamodule ---
_safe_import("IEFT.datamodules.s2_npz_datamodule", "S2NPZDataModule", "s2_npz")

# --- new LEVIR-CD datamodule ---
_safe_import("IEFT.datamodules.levir_cd_datamodule", "LEVIRCDDataModule", "levir_cd")