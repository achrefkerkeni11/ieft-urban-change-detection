import os
import json
import argparse
from pathlib import Path
from typing import Dict, List


def is_image_file(name: str) -> bool:
    name = name.lower()
    return name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def list_stems(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    stems = []
    for name in os.listdir(folder):
        if is_image_file(name):
            stems.append(Path(name).stem)
    stems.sort()
    return stems


def make_entry_blank(stem: str, split: str) -> Dict:
    # Version la plus sûre :
    # on crée une structure correcte sans inventer un faux contenu OSM.
    return {
        "text_v21": "no_osm_context",
        "summary": "unknown osm context",
        "source_text": "",
        "tags": [],
        "phrases": [],
        "split": split,
        "stem": stem
    }


def make_entry_generic(stem: str, split: str) -> Dict:
    # Version générique utile pour tester le pipeline,
    # mais ce n'est PAS un vrai OSM sémantique.
    return {
        "text_v21": "Generic LEVIR patch context with unknown detailed OSM semantics.",
        "summary": "generic context",
        "source_text": "",
        "tags": ["generic_context"],
        "phrases": [
            "generic patch context",
            "unknown detailed osm semantics"
        ],
        "split": split,
        "stem": stem
    }


def build_levir_osm_json(data_root: Path, mode: str = "blank") -> Dict[str, Dict]:
    result = {}

    for split in ["train", "val", "test"]:
        a_dir = data_root / split / "A"
        stems = list_stems(a_dir)

        for stem in stems:
            if mode == "generic":
                entry = make_entry_generic(stem, split)
            else:
                entry = make_entry_blank(stem, split)

            # clé exactement compatible LEVIR : train_1, val_3, test_60, etc.
            result[stem] = entry

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=r'Chemin vers data_levir_cd\raw\LEVIR CD'
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Chemin de sortie du nouveau JSON OSM LEVIR"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="blank",
        choices=["blank", "generic"],
        help="blank = structure vide sûre ; generic = texte générique de test"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root introuvable: {data_root}")

    data = build_levir_osm_json(data_root, mode=args.mode)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] JSON créé : {out_path}")
    print(f"[INFO] Nombre total d'entrées : {len(data)}")

    # petit aperçu
    sample_keys = list(data.keys())[:5]
    print("[INFO] Exemples de clés :")
    for k in sample_keys:
        print(" -", k)


if __name__ == "__main__":
    main()