import os
import csv
import argparse
import shutil
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Utilitaires de base
# ---------------------------------------------------------------------------

def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def compute_binary_metrics(pred, ref):
    """
    Calcule toutes les métriques binaires incluant le F1 score.
    Cas vrai-négatif pur (empty/empty) → toutes métriques = 1.0
    """
    pred = (pred > 0).astype(np.uint8)
    ref  = (ref  > 0).astype(np.uint8)

    tp = int(((pred == 1) & (ref == 1)).sum())
    fp = int(((pred == 1) & (ref == 0)).sum())
    fn = int(((pred == 0) & (ref == 1)).sum())
    tn = int(((pred == 0) & (ref == 0)).sum())

    empty_empty = (tp == 0 and fp == 0 and fn == 0)

    if empty_empty:
        return {
            'iou':             1.0,
            'dice':            1.0,
            'f1':              1.0,    # F1 = Dice pour la segmentation binaire
            'precision':       1.0,
            'recall':          1.0,
            'accuracy':        1.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'empty_empty_match': 1,
        }

    iou       = safe_div(tp, tp + fp + fn)
    # F1 (Dice) = 2*TP / (2*TP + FP + FN)  — identique au Dice pour B/W
    f1        = safe_div(2 * tp, 2 * tp + fp + fn)
    dice      = f1   # alias explicite pour cohérence avec l'ancien code
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    accuracy  = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        'iou':             iou,
        'dice':            dice,
        'f1':              f1,
        'precision':       precision,
        'recall':          recall,
        'accuracy':        accuracy,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'empty_empty_match': 0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_npz_files(folder):
    npz_dir = os.path.join(folder, 'npz')
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(f'NPZ folder not found: {npz_dir}')
    files = sorted(
        os.path.join(npz_dir, name)
        for name in os.listdir(npz_dir)
        if name.lower().endswith('.npz')
    )
    return files


def summarize_rows(rows, key):
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    if len(vals) == 0:
        return {k: 0.0 for k in [f'{key}_mean', f'{key}_std', f'{key}_min', f'{key}_max', f'{key}_median']}
    return {
        f'{key}_mean':   float(vals.mean()),
        f'{key}_std':    float(vals.std()),
        f'{key}_min':    float(vals.min()),
        f'{key}_max':    float(vals.max()),
        f'{key}_median': float(np.median(vals)),
    }


def compute_component_stats(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    labeled, num = ndi.label(mask)
    if num == 0:
        return 0, 0.0, 0.0
    sizes = np.asarray(
        ndi.sum(mask, labeled, index=np.arange(1, num + 1)),
        dtype=np.float64,
    )
    return int(num), float(sizes.mean()), float(sizes.max())


def border_stats(pred, ref, border_band=8):
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    ref  = (np.asarray(ref)  > 0).astype(np.uint8)
    h, w = pred.shape
    b = int(max(1, border_band))
    border = np.zeros((h, w), dtype=np.uint8)
    border[:b, :] = 1
    border[-b:, :] = 1
    border[:, :b] = 1
    border[:, -b:] = 1
    fp_border   = int(((pred == 1) & (ref == 0) & (border == 1)).sum())
    pred_border = int(((pred == 1) & (border == 1)).sum())
    border_area = int(border.sum())
    return safe_div(fp_border, border_area), safe_div(pred_border, border_area)


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    ensure_parent(path)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def make_rank_sort_key(row: Dict, prefix: str, rank_metric: str) -> Tuple:
    return (
        -float(row.get(f'{prefix}_{rank_metric}', 0.0)),
        -float(row.get(f'{prefix}_iou',       0.0)),
        -float(row.get(f'{prefix}_f1',        0.0)),
        -float(row.get(f'{prefix}_dice',      0.0)),
        -float(row.get(f'{prefix}_precision', 0.0)),
         float(row.get(f'{prefix}_abs_ratio_gap',         0.0)),
         float(row.get(f'{prefix}_pred_component_count',  0.0)),
         str(row.get('patch_id', '')),
    )


def ranked_subset(rows: List[Dict], prefix: str, ranking_mode: str) -> List[Dict]:
    if ranking_mode == 'all':
        return list(rows)
    return [
        r for r in rows
        if int(r.get('ref_has_change', 0)) == 1 or int(r.get(f'{prefix}_pred_has_change', 0)) == 1
    ]


def build_rank_rows(rows: List[Dict], prefix: str, rank_metric: str, top_k: int, ranking_mode: str):
    subset      = ranked_subset(rows, prefix, ranking_mode)
    best_sorted = sorted(subset, key=lambda r: make_rank_sort_key(r, prefix, rank_metric))
    best_rows   = best_sorted[:top_k]
    worst_rows  = list(reversed(best_sorted[-top_k:])) if best_sorted else []
    return subset, best_rows, worst_rows


def compact_rank_record(row: Dict, prefix: str, section: str, rank_idx: int) -> Dict:
    return {
        'section':                  section,
        'rank':                     rank_idx,
        'patch_id':                 row['patch_id'],
        'ref_has_change':           row['ref_has_change'],
        'pred_has_change':          row[f'{prefix}_pred_has_change'],
        'ref_ratio':                row['ref_ratio'],
        'pred_ratio':               row[f'{prefix}_pred_ratio'],
        'abs_ratio_gap':            row[f'{prefix}_abs_ratio_gap'],
        'iou':                      row[f'{prefix}_iou'],
        'f1':                       row[f'{prefix}_f1'],
        'dice':                     row[f'{prefix}_dice'],
        'precision':                row[f'{prefix}_precision'],
        'recall':                   row[f'{prefix}_recall'],
        'accuracy':                 row[f'{prefix}_accuracy'],
        'pred_component_count':     row[f'{prefix}_pred_component_count'],
        'pred_mean_component_area': row[f'{prefix}_pred_mean_component_area'],
        'pred_max_component_area':  row[f'{prefix}_pred_max_component_area'],
        'fp_border_ratio':          row[f'{prefix}_fp_border_ratio'],
        'pred_border_ratio':        row[f'{prefix}_pred_border_ratio'],
        'empty_empty_match':        row[f'{prefix}_empty_empty_match'],
    }


def copy_best_panel_if_exists(input_dir: str, output_root: str, prefix: str, best_patch_id: str) -> str:
    src = os.path.join(input_dir, 'panels', f'{best_patch_id}.png')
    if not os.path.isfile(src):
        return ''
    best_dir = os.path.join(output_root, 'best_patch_panels')
    os.makedirs(best_dir, exist_ok=True)
    dst = os.path.join(best_dir, f'{prefix}__best__{best_patch_id}.png')
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Affichage lisible
# ---------------------------------------------------------------------------

SEP_WIDE  = "=" * 90
SEP_THIN  = "-" * 90
SEP_MED   = "─" * 70


def _fmt_pct(v):
    """Formate un float 0-1 en pourcentage avec 2 décimales."""
    return f"{v * 100:6.2f}%"


def _fmt_val(v):
    return f"{v:.6f}"


def print_global_summary(prefix: str, rows: List[Dict], active_rows: List[Dict]):
    """Affichage groupé et lisible du résumé global."""
    print()
    print(SEP_WIDE)
    print(f"  VARIANT : {prefix}")
    print(f"  Patches total : {len(rows)}   |   Patches actifs : {len(active_rows)}")
    print(SEP_WIDE)

    # Métriques principales sur tous les patches
    main_metrics = ['iou', 'f1', 'dice', 'precision', 'recall', 'accuracy']
    print()
    print("  ── Métriques principales (tous patches) ──")
    print(f"  {'Métrique':<14}  {'Moyenne':>8}  {'Médiane':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for m in main_metrics:
        key = f'{prefix}_{m}'
        s = summarize_rows(rows, key)
        print(
            f"  {m:<14}  "
            f"{_fmt_pct(s[f'{key}_mean']):>8}  "
            f"{_fmt_pct(s[f'{key}_median']):>8}  "
            f"{_fmt_pct(s[f'{key}_min']):>8}  "
            f"{_fmt_pct(s[f'{key}_max']):>8}  "
            f"{s[f'{key}_std']:>8.4f}"
        )

    # Métriques sur patches actifs
    if active_rows:
        print()
        print(f"  ── Métriques sur patches actifs uniquement ({len(active_rows)}) ──")
        print(f"  {'Métrique':<14}  {'Moyenne':>8}  {'Médiane':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        for m in main_metrics:
            key = f'{prefix}_{m}'
            s = summarize_rows(active_rows, key)
            print(
                f"  {m:<14}  "
                f"{_fmt_pct(s[f'{key}_mean']):>8}  "
                f"{_fmt_pct(s[f'{key}_median']):>8}  "
                f"{_fmt_pct(s[f'{key}_min']):>8}  "
                f"{_fmt_pct(s[f'{key}_max']):>8}"
            )

    # Métriques composantes
    comp_metrics = ['pred_component_count', 'pred_mean_component_area', 'pred_max_component_area',
                    'fp_border_ratio', 'pred_border_ratio', 'abs_ratio_gap', 'empty_empty_match']
    print()
    print("  ── Métriques composantes / qualité ──")
    print(f"  {'Métrique':<30}  {'Moyenne':>10}  {'Médiane':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for m in comp_metrics:
        key = f'{prefix}_{m}'
        s = summarize_rows(rows, key)
        print(
            f"  {m:<30}  "
            f"{s[f'{key}_mean']:>10.4f}  "
            f"{s[f'{key}_median']:>10.4f}  "
            f"{s[f'{key}_min']:>10.4f}  "
            f"{s[f'{key}_max']:>10.4f}"
        )


def print_rank_block(title: str, rows: List[Dict], prefix: str, rank_metric: str):
    if not rows:
        print(f"\n  {title} : aucun patch.")
        return

    print()
    print(f"  {title}")
    print(f"  {'#':>3}  {'patch_id':<22}  {'IoU':>7}  {'F1':>7}  {'Prec':>7}  {'Recall':>7}  {'#Comp':>5}  {'change':>6}")
    print(f"  {'─'*3}  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*6}")
    for idx, row in enumerate(rows, start=1):
        print(
            f"  {idx:>3}.  {str(row['patch_id']):<22}  "
            f"{_fmt_pct(float(row[f'{prefix}_iou'])):>7}  "
            f"{_fmt_pct(float(row[f'{prefix}_f1'])):>7}  "
            f"{_fmt_pct(float(row[f'{prefix}_precision'])):>7}  "
            f"{_fmt_pct(float(row[f'{prefix}_recall'])):>7}  "
            f"{int(row[f'{prefix}_pred_component_count']):>5}  "
            f"{'oui' if row['ref_has_change'] else 'non':>6}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comparaison qualité overlay avec F1 score")
    parser.add_argument('--input_dir',    type=str, required=True)
    parser.add_argument('--output_csv',   type=str, required=True)
    parser.add_argument('--border_band',  type=int, default=8)
    parser.add_argument('--top_k',        type=int, default=10)
    parser.add_argument('--rank_metric',  type=str, default='f1',
                        choices=['iou', 'f1', 'dice', 'precision', 'recall', 'accuracy'])
    parser.add_argument('--ranking_mode', type=str, default='active',
                        choices=['active', 'all'])
    args = parser.parse_args()

    npz_files = list_npz_files(args.input_dir)
    rows: List[Dict] = []
    variant_names = None

    for path in npz_files:
        data = np.load(path)
        patch_id = str(data['patch_id'])
        ref      = data['gt_mask'].astype(np.uint8)

        pred_keys = sorted(k for k in data.keys() if k.startswith('pred_mask_'))
        if variant_names is None:
            variant_names = pred_keys

        row: Dict = {
            'patch_id':       patch_id,
            'ref_ratio':      float(ref.mean()),
            'ref_has_change': 1 if float(ref.mean()) > 0 else 0,
        }

        for key in pred_keys:
            pred   = data[key].astype(np.uint8)
            prefix = key.replace('pred_mask_', '')
            m      = compute_binary_metrics(pred, ref)
            comp_count, comp_mean_area, comp_max_area = compute_component_stats(pred)
            fp_border_ratio, pred_border_ratio = border_stats(pred, ref, border_band=args.border_band)

            row[f'{prefix}_pred_ratio']              = float(pred.mean())
            row[f'{prefix}_pred_has_change']         = 1 if float(pred.mean()) > 0 else 0
            row[f'{prefix}_abs_ratio_gap']           = abs(float(pred.mean()) - float(ref.mean()))
            row[f'{prefix}_iou']                     = m['iou']
            row[f'{prefix}_f1']                      = m['f1']
            row[f'{prefix}_dice']                    = m['dice']
            row[f'{prefix}_precision']               = m['precision']
            row[f'{prefix}_recall']                  = m['recall']
            row[f'{prefix}_accuracy']                = m['accuracy']
            row[f'{prefix}_empty_empty_match']       = m['empty_empty_match']
            row[f'{prefix}_pred_component_count']    = comp_count
            row[f'{prefix}_pred_mean_component_area']= comp_mean_area
            row[f'{prefix}_pred_max_component_area'] = comp_max_area
            row[f'{prefix}_fp_border_ratio']         = fp_border_ratio
            row[f'{prefix}_pred_border_ratio']       = pred_border_ratio

        rows.append(row)
        data.close()

    # Sauvegarde CSV principal
    fieldnames = list(rows[0].keys()) if rows else ['patch_id']
    write_csv(args.output_csv, fieldnames, rows)

    output_root = os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else '.'
    ranking_dir = os.path.join(output_root, 'rankings')
    os.makedirs(ranking_dir, exist_ok=True)

    print()
    print(SEP_WIDE)
    print(f"  RÉSULTATS — {args.input_dir}")
    print(f"  CSV de sortie : {args.output_csv}")
    print(f"  Nombre de patches : {len(rows)}")
    print(SEP_WIDE)

    for variant_key in (variant_names or []):
        prefix = variant_key.replace('pred_mask_', '')
        active_rows = [
            r for r in rows
            if r['ref_has_change'] == 1 or r[f'{prefix}_pred_has_change'] == 1
        ]
        print_global_summary(prefix, rows, active_rows)

        # Ranking
        ranked_rows, best_rows, worst_rows = build_rank_rows(
            rows, prefix, args.rank_metric, args.top_k, args.ranking_mode
        )
        print()
        print(f"  Ranking mode : {args.ranking_mode}  |  Métrique : {args.rank_metric}  |  Patches classés : {len(ranked_rows)}")
        print_rank_block(f"Top {args.top_k} meilleurs patches", best_rows,  prefix, args.rank_metric)
        print_rank_block(f"Top {args.top_k} pires patches",     worst_rows, prefix, args.rank_metric)

        # Copie du meilleur panel
        best_patch_copy = ''
        if best_rows:
            best_patch_id   = best_rows[0]['patch_id']
            best_patch_copy = copy_best_panel_if_exists(args.input_dir, output_root, prefix, best_patch_id)
            print()
            print(
                f"  ★ Meilleur patch : {best_patch_id}"
                f"  IoU={_fmt_pct(float(best_rows[0][f'{prefix}_iou']))}"
                f"  F1={_fmt_pct(float(best_rows[0][f'{prefix}_f1']))}"
            )
            if best_patch_copy:
                print(f"  Panel copié → {best_patch_copy}")

        # Sauvegarde ranking CSV
        rank_records = []
        for idx, row in enumerate(best_rows,  start=1):
            rank_records.append(compact_rank_record(row, prefix, 'best',  idx))
        for idx, row in enumerate(worst_rows, start=1):
            rank_records.append(compact_rank_record(row, prefix, 'worst', idx))

        if rank_records:
            rank_csv = os.path.join(ranking_dir, f'{prefix}_top_bottom.csv')
            write_csv(rank_csv, list(rank_records[0].keys()), rank_records)

        # Sauvegarde ranking TXT
        summary_txt = os.path.join(ranking_dir, f'{prefix}_ranking_summary.txt')
        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write(f"Variant      : {prefix}\n")
            f.write(f"Mode ranking : {args.ranking_mode}\n")
            f.write(f"Métrique     : {args.rank_metric}\n")
            f.write(f"Patches classés : {len(ranked_rows)}\n\n")
            # Résumé global
            for m in ['iou', 'f1', 'dice', 'precision', 'recall']:
                key = f'{prefix}_{m}'
                s = summarize_rows(rows, key)
                f.write(
                    f"{m:<10} mean={s[f'{key}_mean']:.4f}  "
                    f"median={s[f'{key}_median']:.4f}  "
                    f"min={s[f'{key}_min']:.4f}  "
                    f"max={s[f'{key}_max']:.4f}\n"
                )
            f.write("\n")
            if best_rows:
                br = best_rows[0]
                f.write("Meilleur patch\n")
                f.write(f"  patch_id  = {br['patch_id']}\n")
                f.write(f"  IoU       = {float(br[f'{prefix}_iou']):.4f}\n")
                f.write(f"  F1        = {float(br[f'{prefix}_f1']):.4f}\n")
                f.write(f"  Dice      = {float(br[f'{prefix}_dice']):.4f}\n")
                f.write(f"  Precision = {float(br[f'{prefix}_precision']):.4f}\n")
                f.write(f"  Recall    = {float(br[f'{prefix}_recall']):.4f}\n")
                if best_patch_copy:
                    f.write(f"  Panel     = {best_patch_copy}\n")
                f.write("\n")
            f.write(f"Top {args.top_k} meilleurs\n")
            for idx, row in enumerate(best_rows, start=1):
                f.write(
                    f"  {idx:2d}. {row['patch_id']:<22}"
                    f"  IoU={float(row[f'{prefix}_iou']):.4f}"
                    f"  F1={float(row[f'{prefix}_f1']):.4f}"
                    f"  Prec={float(row[f'{prefix}_precision']):.4f}"
                    f"  Recall={float(row[f'{prefix}_recall']):.4f}\n"
                )
            f.write(f"\nTop {args.top_k} pires\n")
            for idx, row in enumerate(worst_rows, start=1):
                f.write(
                    f"  {idx:2d}. {row['patch_id']:<22}"
                    f"  IoU={float(row[f'{prefix}_iou']):.4f}"
                    f"  F1={float(row[f'{prefix}_f1']):.4f}"
                    f"  Prec={float(row[f'{prefix}_precision']):.4f}"
                    f"  Recall={float(row[f'{prefix}_recall']):.4f}\n"
                )

        print()
        print(f"  CSV ranking → {os.path.join(ranking_dir, prefix + '_top_bottom.csv')}")
        print(f"  TXT résumé  → {summary_txt}")

    print()
    print(SEP_WIDE)
    print("  Export terminé.")
    print(SEP_WIDE)
    print()


if __name__ == '__main__':
    main()