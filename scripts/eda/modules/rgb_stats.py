"""
4.1 - Canales RGB como variables (EDA)

Calcula:
- Estadísticas descriptivas globales por canal (R/G/B) sobre sprites.npy (uint8)
- Estadísticas por clase (si se dispone de labels one-hot en sprites_labels.npy)
- Histogramas globales (muestreo de píxeles para eficiencia)

Salida:
  - reports/figures/eda/rgb/rgb_global_histograms.png
  - reports/figures/eda/rgb/rgb_class_means.png
  - Entrada 'rgb_stats' en reports/eda/eda.json
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

FIG_DIR = Path("reports/figures/eda/rgb")


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


def _channel_stats(channel: np.ndarray) -> Dict[str, float]:
    # channel is uint8 view
    q1, med, q3 = np.percentile(channel, [25, 50, 75])
    return {
        "mean": float(np.mean(channel)),
        "std": float(np.std(channel)),
        "min": float(np.min(channel)),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(np.max(channel)),
    }


def run_rgb_stats(pixel_sample: int = 1_000_000, seed: int = 42) -> Dict:
    print("\n=== 4.1 RGB stats ===\n")

    if not SPRITES_NPY.exists():
        raise FileNotFoundError(f"No se encuentra {SPRITES_NPY}")

    sprites = np.load(SPRITES_NPY)
    labels = np.load(SPRITES_LABELS_NPY) if SPRITES_LABELS_NPY.exists() else None
    int_labels = _infer_int_labels(labels) if labels is not None else None

    r = sprites[..., 0].reshape(-1)
    g = sprites[..., 1].reshape(-1)
    b = sprites[..., 2].reshape(-1)

    _ensure_dir(FIG_DIR)

    # Global stats (exact)
    global_stats = {"R": _channel_stats(r), "G": _channel_stats(g), "B": _channel_stats(b)}

    # Histograms (sample pixels for speed)
    rng = np.random.default_rng(seed)
    n_pix = min(int(pixel_sample), int(r.shape[0]))
    idx = rng.choice(r.shape[0], size=n_pix, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ch_name, arr, color in zip(axes, ["R", "G", "B"], [r[idx], g[idx], b[idx]], ["red", "green", "blue"]):
        ax.hist(arr, bins=50, color=color, alpha=0.85, density=True)
        ax.set_title(f"Histograma global canal {ch_name} (muestra)")
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Densidad")
    fig.tight_layout()
    hist_path = FIG_DIR / "rgb_global_histograms.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)

    class_means = None
    class_means_fig = None
    if int_labels is not None:
        class_means = {}
        classes = np.unique(int_labels)
        for c in classes:
            subset = sprites[int_labels == c]
            mean_rgb = subset.mean(axis=(0, 1, 2))
            class_means[str(int(c))] = {"mean_rgb": [float(x) for x in mean_rgb]}

        # Plot mean RGB per class
        x = np.arange(len(classes))
        means_r = [class_means[str(int(c))]["mean_rgb"][0] for c in classes]
        means_g = [class_means[str(int(c))]["mean_rgb"][1] for c in classes]
        means_b = [class_means[str(int(c))]["mean_rgb"][2] for c in classes]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        width = 0.25
        ax2.bar(x - width, means_r, width, color="red", label="R")
        ax2.bar(x, means_g, width, color="green", label="G")
        ax2.bar(x + width, means_b, width, color="blue", label="B")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"C{int(c)}" for c in classes])
        ax2.set_title("Media RGB por clase")
        ax2.set_ylabel("Intensidad media (0-255)")
        ax2.legend()
        fig2.tight_layout()
        class_means_fig = FIG_DIR / "rgb_class_means.png"
        fig2.savefig(class_means_fig, dpi=200)
        plt.close(fig2)

    results = {
        "global_stats": global_stats,
        "by_class": class_means,
        "figures": {
            "rgb_global_histograms": str(hist_path),
            "rgb_class_means": str(class_means_fig) if class_means_fig else None,
        },
        "pixel_sample_for_hist": int(n_pix),
    }

    update_eda_json("rgb_stats", results)
    print("\n[4.1 RGB COMPLETADO]\n")
    return results


if __name__ == "__main__":
    run_rgb_stats()

