"""
5.3 - Class separability & label analysis
TDSP - EDA

Analiza qué tan separables son las clases del dataset desde un punto de vista
visual y geométrico, integrando:

1. Mosaicos por clase (inspección visual)
2. Imagen promedio por clase
3. Estadísticas de color por clase
4. t-SNE coloreado por etiqueta
5. UMAP coloreado por etiqueta (si está disponible)
6. Silhouette score (raw vs PCA)
7. K-means sobre PCA + matriz etiqueta vs cluster (ARI / NMI)

INPUT:
    data/intermediate/pixel_art_data.npz

OUTPUT FIGURAS:
    reports/figures/eda/class_separability/label_grid_classX.png
    reports/figures/eda/class_separability/label_mean_classX.png
    reports/figures/eda/class_separability/tsne_labels.png
    reports/figures/eda/class_separability/umap_labels.png (opcional)
    reports/figures/eda/class_separability/silhouette_scores.png
    reports/figures/eda/class_separability/confusion_clusters.png

OUTPUT METRICAS (eda.json):
    key: "class_separability"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.cluster import KMeans

from .utils import update_eda_json


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/class_separability")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# VISUALIZACIONES POR CLASE
# ----------------------------------------------------------------------
def plot_class_grid(images, labels, c, fig_dir: Path, max_samples=64):
    """Mosaico de ejemplos de una clase."""
    idx = np.where(labels == c)[0]
    if len(idx) == 0:
        return None

    n_samples = min(max_samples, len(idx))
    idx = np.random.choice(idx, size=n_samples, replace=False)
    imgs = images[idx]

    side = int(np.sqrt(n_samples))
    imgs = imgs[: side * side]

    fig, axes = plt.subplots(side, side, figsize=(side, side))
    for ax, img in zip(axes.flatten(), imgs):
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle(f"Clase {c} - Mosaico visual")
    fig.tight_layout()

    out_path = fig_dir / f"label_grid_class{c}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return str(out_path)


def plot_class_mean(images, labels, c, fig_dir: Path):
    """Imagen promedio de una clase."""
    idx = np.where(labels == c)[0]
    if len(idx) == 0:
        return None

    mean_img = images[idx].mean(axis=0)

    plt.figure(figsize=(3, 3))
    plt.imshow(mean_img)
    plt.axis("off")
    plt.title(f"Clase {c} - Imagen promedio")
    out_path = fig_dir / f"label_mean_class{c}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return str(out_path)


def color_stats_by_class(images, labels, num_classes: int):
    """Media y desviación estándar RGB por clase."""
    stats = {}
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        subset = images[idx]
        if subset.size == 0:
            continue

        mean_rgb = subset.mean(axis=(0, 1, 2))
        std_rgb = subset.std(axis=(0, 1, 2))

        stats[c] = {
            "mean_rgb": [float(x) for x in mean_rgb],
            "std_rgb": [float(x) for x in std_rgb],
        }
    return stats


# ----------------------------------------------------------------------
# t-SNE y UMAP
# ----------------------------------------------------------------------
def compute_tsne(X_pca, labels, fig_dir: Path):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=42,
    )
    emb = tsne.fit_transform(X_pca)

    plt.figure(figsize=(10, 8))
    classes = np.unique(labels)
    for c in classes:
        idx = labels == c
        plt.scatter(emb[idx, 0], emb[idx, 1], s=10, label=f"Clase {c}")
    plt.title("t-SNE coloreado por clase")
    plt.legend()
    plt.tight_layout()
    out_path = fig_dir / "tsne_labels.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return str(out_path)

# ----------------------------------------------------------------------
# SILHOUETTE + KMEANS
# ----------------------------------------------------------------------
def compute_silhouette(X_flat, X_pca, labels, fig_dir: Path):
    """Silhouette usando las etiquetas verdaderas como partición."""
    sil_raw = float(silhouette_score(X_flat, labels))
    sil_pca = float(silhouette_score(X_pca, labels))

    plt.figure(figsize=(6, 4))
    plt.bar(["Raw", "PCA 50"], [sil_raw, sil_pca], color=["gray", "blue"])
    plt.ylabel("Score")
    plt.title("Silhouette score")
    plt.tight_layout()
    out_path = fig_dir / "silhouette_scores.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    return sil_raw, sil_pca, str(out_path)


def compute_kmeans(X_pca, labels, n_classes: int, fig_dir: Path):
    """K-means en PCA + ARI / NMI + matriz etiqueta vs cluster."""
    kmeans = KMeans(n_clusters=n_classes, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    ari = float(adjusted_rand_score(labels, cluster_labels))
    nmi = float(normalized_mutual_info_score(labels, cluster_labels))

    conf = confusion_matrix(labels, cluster_labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(conf, cmap="viridis")
    plt.colorbar()
    plt.title("Matriz etiquetas vs clusters (K-means)")
    plt.xlabel("Cluster k-means")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    out_path = fig_dir / "confusion_clusters.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        "n_clusters": int(n_classes),
        "ari": ari,
        "nmi": nmi,
        "confusion_matrix_file": str(out_path),
        "confusion_matrix": conf.tolist(),
    }


# ----------------------------------------------------------------------
# MAIN RUNNER (para run_eda.py)
# ----------------------------------------------------------------------
def run_class_separability_analysis():
    print("\n=== 5.3 Class separability & label analysis ===\n")

    ensure_dir(FIG_DIR)

    # 1. Cargar datos
    data = np.load(DATA_PATH)
    images = data["images"]      # (N,16,16,3)
    labels = data["labels"]      # (N,)
    n_samples = images.shape[0]
    n_classes = len(np.unique(labels))

    print(f"[INFO] Dataset: {images.shape}, clases: {n_classes}")

    X_flat = images.reshape(n_samples, -1)
    pca_50 = PCA(n_components=50, random_state=42)
    X_pca = pca_50.fit_transform(X_flat)

    # 2. Visualizaciones por clase
    mosaic_files = {}
    mean_files = {}
    for c in range(n_classes):
        mosaic_files[c] = plot_class_grid(images, labels, c, FIG_DIR)
        mean_files[c] = plot_class_mean(images, labels, c, FIG_DIR)

    color_stats = color_stats_by_class(images, labels, n_classes)

    # 3. t-SNE / UMAP
    tsne_file = compute_tsne(X_pca, labels, FIG_DIR)

    # 4. Silhouette
    sil_raw, sil_pca, sil_fig = compute_silhouette(X_flat, X_pca, labels, FIG_DIR)

    # 5. K-means
    kmeans_info = compute_kmeans(X_pca, labels, n_classes, FIG_DIR)

    # 6. Armar resultados para eda.json
    results = {
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "label_visualization": {
            "mosaic_files": {str(k): v for k, v in mosaic_files.items()},
            "mean_image_files": {str(k): v for k, v in mean_files.items()},
        },
        "color_stats": {str(k): v for k, v in color_stats.items()},
        "embeddings": {
            "tsne_file": tsne_file
        },
        "silhouette": {
            "raw_pixels": sil_raw,
            "pca_50": sil_pca,
            "figure_file": sil_fig,
        },
        "kmeans": kmeans_info,
    }

    update_eda_json("class_separability", results)

    print("\n[5.3 CLASS SEPARABILITY COMPLETADO]\n")
    return results


if __name__ == "__main__":
    run_class_separability_analysis()
