"""
5.1.1 Label Analysis - TDSP EDA
--------------------------------------------------------
Objetivo:
    Entender qué representan las etiquetas del dataset.
    ¿Son tipos de sprites? ¿Orígenes gráficos? ¿Estilos? ¿Nada?

Este script implementa cinco análisis:

1. Mosaicos por clase (visual inspection)
2. Imagen promedio por clase
3. Estadísticas de color por clase
4. t-SNE coloreado por clase
5. K-Means vs labels (ARI / NMI)

INPUT:
    data/intermediate/pixel_art_data.npz

OUTPUT:
    reports/figures/label_grid_classX.png
    reports/figures/label_mean_classX.png
    reports/figures/tsne_by_label.png
    reports/figures/kmeans_vs_labels.png
    reports/eda/label_analysis_report.txt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

DATA_PATH = "data/intermediate/pixel_art_data.npz"
FIG_DIR = "reports/figures"
REPORT_PATH = "reports/eda/label_analysis_report.txt"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def load_data():
    data = np.load(DATA_PATH)
    return data["images"], data["labels"]


# --------------------------------------------------------------
# 1. MOSAICOS POR CLASE
# --------------------------------------------------------------
def plot_class_grid(images, labels, c, max_samples=64):
    idx = np.where(labels == c)[0]
    idx = np.random.choice(idx, size=min(max_samples, len(idx)), replace=False)
    imgs = images[idx]

    rows = cols = int(np.sqrt(len(imgs)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

    for ax, img in zip(axes.flatten(), imgs):
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle(f"Clase {c} - Mosaico visual")
    fig.tight_layout()

    save_path = f"{FIG_DIR}/label_grid_class{c}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# --------------------------------------------------------------
# 2. IMAGEN PROMEDIO POR CLASE
# --------------------------------------------------------------
def plot_class_mean(images, labels, c):
    idx = np.where(labels == c)[0]
    mean_img = images[idx].mean(axis=0)

    plt.figure(figsize=(3, 3))
    plt.imshow(mean_img)
    plt.axis("off")
    plt.title(f"Clase {c} - Imagen promedio")

    save_path = f"{FIG_DIR}/label_mean_class{c}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# --------------------------------------------------------------
# 3. ESTADÍSTICAS DE COLOR POR CLASE
# --------------------------------------------------------------
def color_stats_by_class(images, labels, num_classes=5):
    stats = {}

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        subset = images[idx]

        mean_rgb = subset.mean(axis=(0,1,2))
        std_rgb = subset.std(axis=(0,1,2))

        stats[c] = {
            "mean_rgb": mean_rgb.tolist(),
            "std_rgb": std_rgb.tolist(),
            "colorfulness": float(np.sqrt(mean_rgb.var()))
        }

    return stats


# --------------------------------------------------------------
# 4. t-SNE coloreado por clase
# --------------------------------------------------------------
def tsne_by_class(images, labels):
    N = images.shape[0]
    flat = images.reshape(N, -1)

    pca = PCA(n_components=40, random_state=42)
    flat_40 = pca.fit_transform(flat)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb = tsne.fit_transform(flat_40)

    plt.figure(figsize=(8, 6))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(emb[idx, 0], emb[idx, 1], s=4, alpha=0.6, label=f"Clase {c}")

    plt.legend()
    plt.title("t-SNE coloreado por clase")
    plt.tight_layout()

    save_path = f"{FIG_DIR}/tsne_by_label.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# --------------------------------------------------------------
# 5. K-Means vs labels (ARI & NMI)
# --------------------------------------------------------------
def clustering_vs_labels(images, labels):
    N = images.shape[0]
    flat = images.reshape(N, -1)

    pca = PCA(n_components=40, random_state=42)
    flat_40 = pca.fit_transform(flat)

    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(flat_40)

    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)

    return ari, nmi


# --------------------------------------------------------------
# REPORTE FINAL AUTOMÁTICO
# --------------------------------------------------------------
def write_report(color_stats, ari, nmi):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== Análisis de Etiquetas ===\n\n")
        f.write("Objetivo: inferir qué representan las etiquetas del dataset.\n\n")

        f.write("---- Estadísticas de color ----\n")
        for c, info in color_stats.items():
            f.write(f"\nClase {c}:\n")
            f.write(f"  media RGB: {info['mean_rgb']}\n")
            f.write(f"  std RGB: {info['std_rgb']}\n")
            f.write(f"  colorfulness: {info['colorfulness']:.4f}\n")

        f.write("\n---- Clustering vs Labels ----\n")
        f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
        f.write(f"Normalized Mutual Information (NMI): {nmi:.4f}\n")

        f.write("\nInterpretación inicial:\n")

        if ari > 0.6:
            f.write("* Las clases coinciden fuertemente con clusters visuales.\n")
            f.write("* Esto sugiere que las labels representan estilos/grupos de origen.\n")
        elif ari > 0.3:
            f.write("* Las clases capturan parte de la estructura visual, pero no toda.\n")
            f.write("* Podrían ser categorías semánticas generales.\n")
        else:
            f.write("* Poco solapamiento entre clusters y labels.\n")
            f.write("* Las etiquetas podrían representar metadatos no visuales.\n")

        f.write("\nAnálisis generado automáticamente.\n")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():

    ensure_dirs()
    images, labels = load_data()

    print("[INFO] Generando mosaicos y medias por clase...")
    for c in np.unique(labels):
        plot_class_grid(images, labels, c)
        plot_class_mean(images, labels, c)

    print("[INFO] Analizando color por clase...")
    color_stats = color_stats_by_class(images, labels)

    print("[INFO] Ejecutando t-SNE...")
    tsne_by_class(images, labels)

    print("[INFO] Ejecutando K-Means vs labels...")
    ari, nmi = clustering_vs_labels(images, labels)

    print("[INFO] Escribiendo reporte...")
    write_report(color_stats, ari, nmi)

    print("\n[SUCCESS] Análisis de etiquetas completado.")


if __name__ == "__main__":
    main()
