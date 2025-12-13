"""
4.2 - Número de colores por imagen (paleta)

Cuenta colores únicos por imagen (muestra) y clasifica en:
- Low palette (<=32)
- Mid palette (33-128)
- High palette (>128)

Salida:
  - reports/figures/eda/palette/palette_hist.png
  - reports/figures/eda/palette/palette_boxplot_by_class.png
  - Entrada 'palette_stats' en reports/eda/eda.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from .utils import update_eda_json

RAW_DIR = Path("data/raw")
SPRITES_NPY = RAW_DIR / "sprites.npy"
SPRITES_LABELS_NPY = RAW_DIR / "sprites_labels.npy"

FIG_DIR = Path("reports/figures/eda/palette")

LOW_PALETTE_MAX = 32
MID_PALETTE_MAX = 128


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_int_labels(labels: np.ndarray) -> Optional[np.ndarray]:
    if labels.ndim != 2 or labels.size == 0:
        return None
    row_sums = labels.sum(axis=1)
    is_one_hot = np.all((labels == 0) | (labels == 1)) and np.all(row_sums == 1)
    if not is_one_hot:
        return None
    return np.argmax(labels, axis=1).astype(int)


def _unique_color_count(img: np.ndarray) -> int:
    # img: (16,16,3) uint8
    return int(np.unique(img.reshape(-1, 3), axis=0).shape[0])


def _palette_class(n_colors: int) -> str:
    if n_colors <= LOW_PALETTE_MAX:
        return "Low_Palette (<=32)"
    if n_colors <= MID_PALETTE_MAX:
        return "Mid_Palette (33-128)"
    return "High_Palette (>128)"


def run_palette_stats(sample_images: int = 8000, seed: int = 42) -> Dict:
    print("\n=== 4.2 Palette stats (colores únicos) ===\n")

    if not SPRITES_NPY.exists():
        raise FileNotFoundError(f"No se encuentra {SPRITES_NPY}")

    sprites = np.load(SPRITES_NPY)
    labels = np.load(SPRITES_LABELS_NPY) if SPRITES_LABELS_NPY.exists() else None
    int_labels = _infer_int_labels(labels) if labels is not None else None

    rng = np.random.default_rng(seed)
    n = min(int(sample_images), int(sprites.shape[0]))
    idx = rng.choice(sprites.shape[0], size=n, replace=False)
    sample = sprites[idx]
    sample_labels = int_labels[idx] if int_labels is not None else None

    counts = np.array([_unique_color_count(img) for img in sample], dtype=np.int32)
    palette_classes = np.array([_palette_class(int(c)) for c in counts])

    _ensure_dir(FIG_DIR)

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=30, color="purple", alpha=0.85)
    plt.axvline(LOW_PALETTE_MAX, color="red", linestyle="--", label="Low/Mid cutoff")
    plt.axvline(MID_PALETTE_MAX, color="green", linestyle="--", label="Mid/High cutoff")
    plt.title("Distribución del número de colores únicos por imagen (muestra)")
    plt.xlabel("Colores únicos (máx 256)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    hist_path = FIG_DIR / "palette_hist.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()

    palette_counts = {k: int(v) for k, v in zip(*np.unique(palette_classes, return_counts=True))}

    # Checks requested in the report (computed on the same sample)
    threshold_checks = {
        "lt_3_colors": int(np.sum(counts < 3)),
        "gt_50_colors": int(np.sum(counts > 50)),
        "gt_128_colors": int(np.sum(counts > 128)),
    }

    unique_color_count_summary = {
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "p25": float(np.percentile(counts, 25)),
        "p75": float(np.percentile(counts, 75)),
    }

    boxplot_path = None
    by_class = None
    if sample_labels is not None:
        by_class = {}
        for c in np.unique(sample_labels):
            cls_counts = counts[sample_labels == c]
            by_class[str(int(c))] = {
                "n": int(cls_counts.size),
                "mean": float(np.mean(cls_counts)),
                "median": float(np.median(cls_counts)),
            }

        # simple boxplot by class
        classes = sorted(by_class.keys(), key=int)
        data = [counts[sample_labels == int(c)] for c in classes]
        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=[f"C{c}" for c in classes], showfliers=False)
        plt.title("Colores únicos por imagen (muestra) - por clase")
        plt.xlabel("Clase")
        plt.ylabel("Colores únicos")
        plt.tight_layout()
        boxplot_path = FIG_DIR / "palette_boxplot_by_class.png"
        plt.savefig(boxplot_path, dpi=200)
        plt.close()

    results = {
        "n_samples_used": int(n),
        "palette_class_thresholds": {"low_max": LOW_PALETTE_MAX, "mid_max": MID_PALETTE_MAX},
        "palette_class_counts": palette_counts,
        "unique_color_count_summary": unique_color_count_summary,
        "threshold_checks": threshold_checks,
        "by_class_summary": by_class,
        "figures": {
            "palette_hist": str(hist_path),
            "palette_boxplot_by_class": str(boxplot_path) if boxplot_path else None,
        },
    }

    update_eda_json("palette_stats", results)
    print("\n[4.2 PALETTE COMPLETADO]\n")
    return results


if __name__ == "__main__":
    run_palette_stats()
