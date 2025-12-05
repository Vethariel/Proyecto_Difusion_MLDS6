"""
5.1 PCA Analysis - TDSP EDA
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

from .utils import update_eda_json

DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/pca")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_pca_analysis():

    # 1. Load data
    data = np.load(DATA_PATH)
    images = data["images"]

    N = images.shape[0]
    flat = images.reshape(N, -1)

    ensure_dir(FIG_DIR)

    # 2. PCA base
    pca = PCA(n_components=50, random_state=42)
    pca.fit(flat)

    var_ratio = pca.explained_variance_ratio_

    # FIG 1 - Variance curve
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(var_ratio[:40]), marker="o")
    plt.title("PCA – Variancia explicada acumulada")
    plt.xlabel("Componentes")
    plt.ylabel("Varianza acumulada")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "variance_ratio.png", dpi=300)
    plt.close()

    # 3. Reconstructions
    sample_idx = 0
    original_flat = flat[sample_idx:sample_idx+1]

    recon_dict = {}

    for k in [10, 20, 30]:
        pca_k = PCA(n_components=k, random_state=42)
        pca_k.fit(flat)

        encoded = pca_k.transform(original_flat)
        decoded = pca_k.inverse_transform(encoded)
        img = np.clip(decoded.reshape(16,16,3), 0, 1)

        plt.figure(figsize=(3,3))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        out_path = FIG_DIR / f"reconstruction_k{k}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        recon_dict[f"k{k}"] = str(out_path)

    # 4. Build output dict
    results = {
        "n_samples": int(N),
        "flat_dim": int(flat.shape[1]),
        "variance_first_component": float(var_ratio[0]),
        "variance_first_10": float(np.sum(var_ratio[:10])),
        "variance_first_20": float(np.sum(var_ratio[:20])),
        "variance_curve_file": str(FIG_DIR / "variance_ratio.png"),
        "reconstruction_files": recon_dict
    }

    # 5. Update eda.json
    update_eda_json("pca_analysis", results)

    return results


if __name__ == "__main__":
    r = run_pca_analysis()
    print("\n[PCA 5.1 COMPLETADO]\n")
