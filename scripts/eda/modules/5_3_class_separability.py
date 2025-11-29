"""
5.3 - Separabilidad entre clases (t-SNE, Silhouette, K-means)
TDSP - EDA

Este script analiza qué tan separables son las clases del dataset
desde un punto de vista visual y geométrico usando:
    - t-SNE coloreado por etiqueta
    - Silhouette score
    - K-means sobre PCA
    - Matriz etiqueta vs cluster

Salida:
    - Figuras PNG en reports/figures/eda/separabilidad/
    - Métricas impresas en consola
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from pathlib import Path

# =====================
# CONFIGURACIÓN
# =====================
DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/separabilidad")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# CARGA DE DATOS
# =====================
data = np.load(DATA_PATH)
images = data["images"]      # (N,16,16,3)
labels = data["labels"]      # (N,)
num_classes = len(np.unique(labels))

X_flat = images.reshape(images.shape[0], -1)  # (N, 768)

print(f"[INFO] Dataset cargado: {images.shape}")
print(f"[INFO] Clases detectadas: {num_classes}")

# =====================
# 1. t-SNE coloreado por clase
# =====================
print("\n=== 5.3.1 t-SNE ===")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    max_iter=1000,
    init='pca',
    random_state=42
)

X_tsne = tsne.fit_transform(X_flat)

plt.figure(figsize=(10, 8))
for c in range(num_classes):
    idx = (labels == c)
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=10, label=f"Clase {c}")

plt.title("t-SNE coloreado por clase")
plt.legend()
plt.savefig(FIG_DIR / "tsne_labels.png", dpi=200)
plt.close()

print("→ Figura guardada: tsne_labels.png")

# =====================
# 2. Silhouette Score (raw y PCA)
# =====================
print("\n=== 5.3.2 Silhouette score ===")

sil_raw = silhouette_score(X_flat, labels)

pca_50 = PCA(n_components=50)
X_pca = pca_50.fit_transform(X_flat)
sil_pca = silhouette_score(X_pca, labels)

print(f"Silhouette (raw pixels): {sil_raw:.4f}")
print(f"Silhouette (PCA=50):    {sil_pca:.4f}")

# Gráfico
plt.figure(figsize=(6,4))
plt.bar(["Raw", "PCA 50"], [sil_raw, sil_pca], color=["gray", "blue"])
plt.ylabel("Score")
plt.title("Silhouette score")
plt.savefig(FIG_DIR / "silhouette_scores.png", dpi=200)
plt.close()

print("→ Figura guardada: silhouette_scores.png")

# =====================
# 3. K-means clustering
# =====================
print("\n=== 5.3.3 K-means ===")

k = num_classes
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)

ari = adjusted_rand_score(labels, cluster_labels)
nmi = normalized_mutual_info_score(labels, cluster_labels)

print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# =====================
# 4. Matriz etiqueta vs cluster
# =====================

conf = confusion_matrix(labels, cluster_labels)

plt.figure(figsize=(8, 6))
plt.imshow(conf, cmap="viridis")
plt.colorbar()
plt.title("Matriz etiquetas vs clusters (K-means)")
plt.xlabel("Cluster k-means")
plt.ylabel("Etiqueta real")
plt.savefig(FIG_DIR / "confusion_clusters.png", dpi=200)
plt.close()

print("→ Figura guardada: confusion_clusters.png")

print("\nFiguras guardadas en:", FIG_DIR)
print("\n=== COMPLETADO 5.3 ===")


# =====================
# 1-B. UMAP coloreado por clase
# =====================
print("\n=== 5.3.1-B UMAP ===")

import umap

umap_model = umap.UMAP(
    n_neighbors=30,          # sensibilidad local
    min_dist=0.1,            # compactación de clusters
    n_components=2,
    metric="euclidean",
    random_state=42
)

X_umap = umap_model.fit_transform(X_flat)

plt.figure(figsize=(10, 8))
for c in range(num_classes):
    idx = (labels == c)
    plt.scatter(X_umap[idx, 0], X_umap[idx, 1], s=10, label=f"Clase {c}")

plt.title("UMAP coloreado por clase")
plt.legend()
plt.savefig(FIG_DIR / "umap_labels.png", dpi=200)
plt.close()

print("→ Figura guardada: umap_labels.png")
