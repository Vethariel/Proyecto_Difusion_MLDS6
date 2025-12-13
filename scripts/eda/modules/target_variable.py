"""
3.1/3.2 - Variable objetivo (imágenes) y contraste real vs ruido

Este módulo genera evidencia visual de por qué, en un modelo generativo (difusión),
la "variable objetivo" es la propia imagen: buscamos modelar p_data(x).

Salida:
  - reports/figures/eda/target_variable/target_variable.png
  - Entrada 'target_variable' en reports/eda/eda.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from .utils import update_eda_json

DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/target_variable")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_target_variable_analysis(sample_images: int = 2000, seed: int = 42) -> Dict:
    print("\n=== 3.x Variable objetivo: imagen (real vs ruido) ===\n")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {DATA_PATH}")

    data = np.load(DATA_PATH)
    images = data["images"].astype(np.float32)  # [0,1]

    rng = np.random.default_rng(seed)
    n = min(sample_images, images.shape[0])
    idx = rng.choice(images.shape[0], size=n, replace=False)
    sample = images[idx]

    # Distribución de intensidades (real) vs ruido uniforme
    real_pixels = sample.reshape(-1)
    noise = rng.random(sample.shape, dtype=np.float32)
    noise_pixels = noise.reshape(-1)

    _ensure_dir(FIG_DIR)
    out_path = FIG_DIR / "target_variable.png"

    # Imagen real y ruido (una sola muestra)
    real_img = sample[0]
    noise_img = noise[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(real_pixels, bins=50, color="teal", alpha=0.85, density=True)
    axes[0, 0].set_title("Distribución de intensidades (real)")
    axes[0, 0].set_xlabel("Intensidad [0,1]")
    axes[0, 0].set_ylabel("Densidad")

    axes[0, 1].hist(noise_pixels, bins=50, color="orange", alpha=0.85, density=True)
    axes[0, 1].set_title("Distribución de intensidades (ruido)")
    axes[0, 1].set_xlabel("Intensidad [0,1]")
    axes[0, 1].set_ylabel("Densidad")

    axes[1, 0].imshow(real_img)
    axes[1, 0].set_title("Ejemplo real (x ~ p_data)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(noise_img)
    axes[1, 1].set_title("Ejemplo ruido (x ~ p_ruido)")
    axes[1, 1].axis("off")

    fig.suptitle("Variable objetivo en modelos generativos: la imagen")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    results = {
        "n_samples_used": int(n),
        "figures": {"target_variable": str(out_path)},
    }

    update_eda_json("target_variable", results)
    print("\n[3.x VARIABLE OBJETIVO COMPLETADO]\n")
    return results


if __name__ == "__main__":
    run_target_variable_analysis()

