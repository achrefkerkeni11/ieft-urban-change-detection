# IEFT/datasets/__init__.py
# Certains datasets peuvent ne pas être disponibles / importables selon l'OS,
# les dépendances, ou des différences de noms de classes.
# On rend les imports optionnels pour ne pas bloquer l'exécution (ex: s2_npz).

# --- Always available (our custom dataset) ---
from .s2_npz_dataset import S2NPZDataset

# --- Optional datasets (do not break import if missing) ---
try:
    from .vg_caption_dataset import VisualGenomeCaptionDataset
except Exception:
    VisualGenomeCaptionDataset = None

try:
    from .coco_caption_karpathy_dataset import CocoCaptionKarpathyDataset
except Exception:
    CocoCaptionKarpathyDataset = None

try:
    from .f30k_caption_karpathy_dataset import F30KCaptionKarpathyDataset
except Exception:
    F30KCaptionKarpathyDataset = None

try:
    from .conceptual_caption_dataset import ConceptualCaptionDataset
except Exception:
    ConceptualCaptionDataset = None

try:
    from .sbu_caption_dataset import SBUCaptionDataset
except Exception:
    SBUCaptionDataset = None

try:
    from .vqav2_dataset import VQAv2Dataset
except Exception:
    VQAv2Dataset = None

try:
    from .nlvr2_dataset import NLVR2Dataset
except Exception:
    NLVR2Dataset = None

try:
    from .rsicd_caption_karpathy_dataset import RSICDCaptionKarpathyDataset
except Exception:
    RSICDCaptionKarpathyDataset = None


# -------------------- SYDNEY (robust) --------------------
# IMPORTANT:
# Le nom de la classe Sydney peut différer selon le repo.
# On tente plusieurs noms possibles sans casser l'import global.

SydneyCaptionKarpathyDataset = None
try:
    # On importe le module une seule fois, puis on cherche plusieurs noms possibles dedans.
    import IEFT.datasets.sydney_caption_karpathy_dataset as _syd_mod

    # Essais de noms possibles (ordre du plus probable au moins probable)
    for _name in ["SydneyCaptionKarpathyDataset", "SYDNEYCaptionKarpathyDataset", "SydneyCaptionDataset"]:
        if hasattr(_syd_mod, _name):
            SydneyCaptionKarpathyDataset = getattr(_syd_mod, _name)
            break
except Exception:
    SydneyCaptionKarpathyDataset = None

# Alias de compatibilité: certains datamodules utilisent ce nom en MAJUSCULES
SYDNEYCaptionKarpathyDataset = SydneyCaptionKarpathyDataset


try:
    from .ucm_caption_karpathy_dataset import UCMCaptionKarpathyDataset
except Exception:
    UCMCaptionKarpathyDataset = None

try:
    from .rsitmd_caption_karpathy_dataset import RSITMDCaptionKarpathyDataset
except Exception:
    RSITMDCaptionKarpathyDataset = None
