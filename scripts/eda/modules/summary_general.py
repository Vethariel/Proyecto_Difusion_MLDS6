"""
1.x - Resumen general del dataset (EDA)

Estandariza el reporte de:
- Tamaños y consistencia entre labels.csv / sprites.npy / sprites_labels.npy / carpeta de imágenes
- Tipos/dtypes y rangos de píxeles
- Distribución de clases (si sprites_labels.npy es one-hot)
- Mosaico aleatorio de imágenes

Salida:
  - reports/figures/eda/summary_general/mosaic_300.png
  - Entrada 'summary_general' en reports/eda/eda.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import update_eda_json

RAW_DIR = Path("data/raw")
RAW_IMAGES_DIR = RAW_DIR / "images" / "images"
LABELS_CSV = RAW_DIR / "labels.csv"
SPRITES_NPY = RAW_DIR / "sprites.npy"
SPRITES_LABELS_NPY = RAW_DIR / "sprites_labels.npy"

FIG_DIR = Path("reports/figures/eda/summary_general")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _infer_int_labels(sprites_labels: np.ndarray) -> Optional[np.ndarray]:
    if sprites_labels.ndim != 2:
        return None
    if sprites_labels.size == 0:
        return None

    row_sums = sprites_labels.sum(axis=1)
    is_one_hot = np.all((sprites_labels == 0) | (sprites_labels == 1)) and np.all(row_sums == 1)
    if not is_one_hot:
        return None
    return np.argmax(sprites_labels, axis=1).astype(int)


def _count_files_fast(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.iterdir() if _.is_file())


def _sample_path_checks(n: int, seed: int, expected_n: int) -> Dict:
    if not RAW_IMAGES_DIR.exists() or expected_n <= 0:
        return {"checked": 0, "missing": 0, "examples": []}

    rng = np.random.default_rng(seed)
    idx = rng.choice(expected_n, size=min(n, expected_n), replace=False)
    missing = []
    for i in idx.tolist():
        fp = RAW_IMAGES_DIR / f"image_{i}.JPEG"
        if not fp.exists():
            missing.append(str(fp))
            if len(missing) >= 5:
                break
    return {"checked": int(len(idx)), "missing": int(len(missing)), "examples": missing}


def _save_mosaic(images: np.ndarray, out_path: Path, seed: int = 42, n_images: int = 300) -> str:
    _ensure_dir(out_path.parent)

    n_images = min(n_images, images.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(images.shape[0], size=n_images, replace=False)

    n_cols = 20
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        ax.axis("off")
        if ax_i < n_images:
            ax.imshow(images[idx[ax_i]])

    fig.suptitle(f"Mosaico aleatorio ({n_images} sprites)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def run_summary_general() -> Dict:
    print("\n=== 1.x Resumen general del dataset ===\n")

    if not SPRITES_NPY.exists():
        raise FileNotFoundError(f"No se encuentra {SPRITES_NPY}")
    if not SPRITES_LABELS_NPY.exists():
        raise FileNotFoundError(f"No se encuentra {SPRITES_LABELS_NPY}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"No se encuentra {LABELS_CSV}")

    sprites = np.load(SPRITES_NPY)
    sprites_labels = np.load(SPRITES_LABELS_NPY)
    labels_df = pd.read_csv(LABELS_CSV)

    n_sprites = int(sprites.shape[0])
    images_count = _count_files_fast(RAW_IMAGES_DIR)

    int_labels = _infer_int_labels(sprites_labels)
    label_counts: Optional[Dict[str, int]] = None
    n_classes: Optional[int] = None
    if int_labels is not None:
        unique, counts = np.unique(int_labels, return_counts=True)
        label_counts = {str(int(u)): int(c) for u, c in zip(unique, counts)}
        n_classes = int(len(unique))

    mosaic_path = _save_mosaic(sprites, FIG_DIR / "mosaic_300.png", seed=42, n_images=300)
    path_checks = _sample_path_checks(n=500, seed=123, expected_n=n_sprites)

    results = {
        "raw_paths": {
            "sprites_npy": str(SPRITES_NPY),
            "sprites_labels_npy": str(SPRITES_LABELS_NPY),
            "labels_csv": str(LABELS_CSV),
            "images_dir": str(RAW_IMAGES_DIR),
        },
        "counts": {
            "n_sprites_npy": n_sprites,
            "n_sprites_labels_npy": int(sprites_labels.shape[0]),
            "n_labels_csv_rows": int(labels_df.shape[0]),
            "n_image_files": int(images_count),
        },
        "image_shape": list(sprites.shape[1:]),
        "dtypes": {
            "sprites": str(sprites.dtype),
            "sprites_labels": str(sprites_labels.dtype),
        },
        "pixel_range": {
            "min": int(np.min(sprites)),
            "max": int(np.max(sprites)),
        },
        "labels": {
            "n_classes_inferred": n_classes,
            "class_counts": label_counts,
            "encoding": "one_hot" if int_labels is not None else "unknown",
        },
        "consistency_checks": {
            "sample_path_checks": path_checks,
        },
        "figures": {
            "mosaic_300": mosaic_path,
        },
    }

    update_eda_json("summary_general", results)
    print("\n[1.x RESUMEN GENERAL COMPLETADO]\n")
    return results


if __name__ == "__main__":
    run_summary_general()

