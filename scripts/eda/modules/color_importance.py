"""
5.2 Color Importance Analysis - TDSP EDA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

from .utils import update_eda_json

DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/color")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Métrica de color Hasler & Süsstrunk
# -------------------------------------------------------
def colorfulness_metric(img):
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return (
        np.sqrt(np.std(rg)**2 + np.std(yb)**2)
        + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    )


# -------------------------------------------------------
# PROCESO PRINCIPAL
# -------------------------------------------------------
def run_color_importance():

    print("\n=== Ejecutando 5.2 Color Importance ===\n")

    data = np.load(DATA_PATH)
    images = data["images"]       # (N,16,16,3)
    labels = data["labels"]       # (N,)
    num_classes = len(np.unique(labels))

    ensure_dir(FIG_DIR)

    flattened = images.reshape(images.shape[0], -1, 3)  # (N,256,3)

    # =====================================================================
    # 1. VARIANZA GLOBAL POR CANAL
    # =====================================================================
    global_variance = np.var(flattened, axis=(0, 1))  # [varR, varG, varB]

    plt.figure(figsize=(6, 4))
    plt.bar(["R", "G", "B"], global_variance, color=["red", "green", "blue"])
    plt.title("Varianza global por canal")
    plt.ylabel("Varianza")

    fig_global_var = FIG_DIR / "global_variance.png"
    plt.savefig(fig_global_var, dpi=200)
    plt.close()

    # =====================================================================
    # 2. VARIANZA POR CLASE
    # =====================================================================
    class_variances = {}

    for c in range(num_classes):
        imgs_c = images[labels == c]
        flat_c = imgs_c.reshape(imgs_c.shape[0], -1, 3)
        class_variances[c] = np.var(flat_c, axis=(0, 1)).tolist()

    plt.figure(figsize=(10, 6))
    width = 0.25
    x = np.arange(num_classes)

    plt.bar(x - width, [class_variances[c][0] for c in range(num_classes)], width, label="R", color="red")
    plt.bar(x       , [class_variances[c][1] for c in range(num_classes)], width, label="G", color="green")
    plt.bar(x + width, [class_variances[c][2] for c in range(num_classes)], width, label="B", color="blue")

    plt.xticks(x, [f"Clase {c}" for c in range(num_classes)])
    plt.title("Varianza por canal en cada clase")
    plt.legend()

    fig_var_by_class = FIG_DIR / "variance_by_class.png"
    plt.savefig(fig_var_by_class, dpi=200)
    plt.close()

    # =====================================================================
    # 3. COLORFULNESS POR CLASE
    # =====================================================================
    colorfulness_per_class = {}

    for c in range(num_classes):
        imgs_c = images[labels == c][:300]  # muestreo para rapidez
        colorfulness_per_class[c] = float(np.mean([colorfulness_metric(img) for img in imgs_c]))

    plt.figure(figsize=(8, 5))
    plt.bar([f"C{c}" for c in range(num_classes)],
            [colorfulness_per_class[c] for c in range(num_classes)],
            color="purple")
    plt.title("Colorfulness (Hasler & Süsstrunk) por clase")
    plt.ylabel("Colorfulness")

    fig_colorfulness = FIG_DIR / "colorfulness_by_class.png"
    plt.savefig(fig_colorfulness, dpi=200)
    plt.close()

    # =====================================================================
    # 4. PCA POR CANAL
    # =====================================================================
    pca_results = {}

    for i, ch in enumerate(["R", "G", "B"]):
        channel_data = flattened[..., i].reshape(images.shape[0], -1)
        pca = PCA(n_components=1)
        pca.fit(channel_data)

        pca_results[ch] = float(pca.explained_variance_ratio_[0])

    plt.figure(figsize=(6, 4))
    plt.bar(["R", "G", "B"],
            [pca_results["R"], pca_results["G"], pca_results["B"]],
            color=["red", "green", "blue"])
    plt.ylabel("Varianza explicada (PC1)")
    plt.title("PCA por canal - Importancia de información")

    fig_pca_channel = FIG_DIR / "pca_by_channel.png"
    plt.savefig(fig_pca_channel, dpi=200)
    plt.close()

    # =====================================================================
    # 5. RANKING FINAL
    # =====================================================================
    ranking = {
        ch: float(global_variance[i] + pca_results[ch])
        for i, ch in enumerate(["R", "G", "B"])
    }

    # =====================================================================
    # Registrar en JSON (formato uniforme con 5.1)
    # =====================================================================
    results_dict = {
        "global_variance": {
            "R": float(global_variance[0]),
            "G": float(global_variance[1]),
            "B": float(global_variance[2])
        },
        "class_variances": class_variances,
        "colorfulness": colorfulness_per_class,
        "pca_channel_importance": pca_results,
        "ranking": ranking,
        "figure_files": {
            "global_variance": str(fig_global_var),
            "variance_by_class": str(fig_var_by_class),
            "colorfulness_by_class": str(fig_colorfulness),
            "pca_by_channel": str(fig_pca_channel)
        }
    }

    update_eda_json("color_importance", results_dict)

    print("\n[5.2 Color Importance COMPLETADO]\n")
    return results_dict


if __name__ == "__main__":
    run_color_importance()
