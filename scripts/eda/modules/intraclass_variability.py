"""
3.3 - Variabilidad intra-clase (mosaicos por clase)

Genera un mosaico comparativo 5×5 para 2 clases (seed fija), para inspección
de variabilidad visual dentro de la misma etiqueta.

Salida:
  - reports/figures/eda/variability/intraclass_mosaic_seed18.png
  - Entrada 'intraclass_variability' en reports/eda/eda.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import update_eda_json

DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/variability")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _choose_two_classes(labels: np.ndarray, min_per_class: int, seed: int) -> Tuple[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    valid = classes[counts >= min_per_class]
    if len(valid) < 2:
        raise ValueError(f"No hay al menos 2 clases con >= {min_per_class} ejemplos.")
    rng = np.random.default_rng(seed)
    c1, c2 = rng.choice(valid, size=2, replace=False).tolist()
    return int(c1), int(c2)


def _plot_side_by_side(images: np.ndarray, labels: np.ndarray, classes: List[int], out_path: Path, seed: int) -> str:
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"Comparación de variabilidad intra‑clase (seed={seed})\n"
        f"Izquierda: Clase {classes[0]} | Derecha: Clase {classes[1]}",
        fontsize=14,
    )

    for mosaic_idx, c in enumerate(classes):
        idx_c = np.where(labels == c)[0]
        chosen = rng.choice(idx_c, size=25, replace=False)
        for i, img_idx in enumerate(chosen.tolist()):
            row = i // 5
            col = (i % 5) + (mosaic_idx * 5)
            ax = fig.add_subplot(5, 10, (row * 10) + col + 1)
            ax.imshow(images[img_idx])
            ax.axis("off")
            if i == 2:
                ax.set_title(f"Clase {c}", fontsize=10, pad=8)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def run_intraclass_variability(seed: int = 18) -> Dict:
    print("\n=== 3.3 Variabilidad intra-clase ===\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {DATA_PATH}")

    data = np.load(DATA_PATH)
    images = data["images"]
    labels = data["labels"].astype(int)

    c1, c2 = _choose_two_classes(labels, min_per_class=25, seed=seed)
    _ensure_dir(FIG_DIR)
    out_path = FIG_DIR / f"intraclass_mosaic_seed{seed}.png"
    fig_path = _plot_side_by_side(images, labels, [c1, c2], out_path, seed=seed)

    results = {
        "seed": int(seed),
        "classes_compared": [int(c1), int(c2)],
        "figure_file": fig_path,
    }

    update_eda_json("intraclass_variability", results)
    print("\n[3.3 INTRACLASS COMPLETADO]\n")
    return results


if __name__ == "__main__":
    run_intraclass_variability()

