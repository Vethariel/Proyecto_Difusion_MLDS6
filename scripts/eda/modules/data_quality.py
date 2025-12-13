"""
2.x - Resumen de calidad de datos (EDA)

Incluye:
- Verificación de faltantes básicos (CSV, NaNs en arrays)
- Verificación de lectura (muestra) de imágenes en disco
- Duplicados exactos por hash sobre sprites.npy (nivel píxel)
- Conteos de casos extremos (completamente negro/blanco)

Salida:
  - reports/figures/eda/data_quality/mean_intensity_hist.png
  - (opcional) mosaicos de outliers
  - Entrada 'data_quality' en reports/eda/eda.json
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from .utils import update_eda_json

RAW_DIR = Path("data/raw")
RAW_IMAGES_DIR = RAW_DIR / "images" / "images"
LABELS_CSV = RAW_DIR / "labels.csv"
SPRITES_NPY = RAW_DIR / "sprites.npy"

FIG_DIR = Path("reports/figures/eda/data_quality")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sample_read_images(sample_n: int = 500, seed: int = 42) -> Dict:
    if not RAW_IMAGES_DIR.exists():
        return {"checked": 0, "errors": 0, "examples": []}

    files = sorted([p for p in RAW_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"}])
    if not files:
        return {"checked": 0, "errors": 0, "examples": []}

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(sample_n, len(files)), replace=False)
    errors = []
    for i in idx.tolist():
        fp = files[i]
        try:
            with Image.open(fp) as im:
                im.convert("RGB")
        except Exception:
            errors.append(str(fp))
            if len(errors) >= 5:
                break
    return {"checked": int(len(idx)), "errors": int(len(errors)), "examples": errors}


def _hash_duplicates(images: np.ndarray) -> Tuple[int, int, float, Optional[Dict[str, int]]]:
    counts: Dict[bytes, int] = {}
    for img in images:
        h = hashlib.md5(img.tobytes()).digest()
        counts[h] = counts.get(h, 0) + 1
    total = int(images.shape[0])
    unique = int(len(counts))
    dup = int(total - unique)
    pct = float(100.0 * dup / total) if total else 0.0

    top_groups = None
    if dup > 0:
        # keep top 5 duplicate group sizes
        sizes = sorted((c for c in counts.values() if c > 1), reverse=True)[:5]
        top_groups = {f"group_{i+1}": int(s) for i, s in enumerate(sizes)}
    return unique, dup, pct, top_groups


def _plot_mean_intensity(images: np.ndarray, out_path: Path) -> str:
    _ensure_dir(out_path.parent)
    mean_intensity = images.mean(axis=(1, 2, 3))
    plt.figure(figsize=(8, 4))
    plt.hist(mean_intensity, bins=50, color="teal", alpha=0.85)
    plt.title("Distribución de intensidad media por imagen")
    plt.xlabel("Intensidad media (0-255)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return str(out_path)


def run_data_quality() -> Dict:
    print("\n=== 2.x Calidad de datos ===\n")

    if not SPRITES_NPY.exists():
        raise FileNotFoundError(f"No se encuentra {SPRITES_NPY}")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"No se encuentra {LABELS_CSV}")

    sprites = np.load(SPRITES_NPY)
    labels_df = pd.read_csv(LABELS_CSV)

    csv_missing = labels_df.isna().sum().to_dict()
    array_nans = int(np.isnan(sprites).sum()) if np.issubdtype(sprites.dtype, np.floating) else 0

    unique_count, dup_count, dup_pct, top_dup_groups = _hash_duplicates(sprites)
    black_count = int(np.sum(np.all(sprites == 0, axis=(1, 2, 3))))
    white_count = int(np.sum(np.all(sprites == 255, axis=(1, 2, 3))))

    read_check = _sample_read_images(sample_n=500, seed=42)
    mean_intensity_fig = _plot_mean_intensity(sprites.astype(np.float32), FIG_DIR / "mean_intensity_hist.png")

    results = {
        "csv_missing_values": {str(k): int(v) for k, v in csv_missing.items()},
        "array_nan_count": array_nans,
        "disk_read_sample_check": read_check,
        "duplicates_exact_pixel_hash": {
            "total_images": int(sprites.shape[0]),
            "unique_images": unique_count,
            "duplicate_images": dup_count,
            "duplicate_percentage": dup_pct,
            "top_duplicate_group_sizes": top_dup_groups,
        },
        "extremes": {
            "all_black_images": black_count,
            "all_white_images": white_count,
        },
        "figures": {
            "mean_intensity_hist": mean_intensity_fig,
        },
    }

    update_eda_json("data_quality", results)
    print("\n[2.x CALIDAD DE DATOS COMPLETADO]\n")
    return results


if __name__ == "__main__":
    run_data_quality()

