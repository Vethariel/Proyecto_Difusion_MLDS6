"""
5.1 PCA Analysis - TDSP EDA
--------------------------------------------------------
- Calcula PCA sobre el dataset de pixel art (16x16x3).
- Genera gráfica de variancia explicada.
- Reconstruye imágenes usando k = 10, 20, 30 componentes.

INPUT:
    data/intermediate/pixel_art_data.npz

OUTPUT:
    reports/figures/pca_variance.png
    reports/figures/pca_reconstruction_k10.png
    reports/figures/pca_reconstruction_k20.png
    reports/figures/pca_reconstruction_k30.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

DATA_PATH = "data/intermediate/pixel_art_data.npz"
OUTPUT_DIR = "reports/figures"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("[INFO] Carpeta creada:", path)

def load_data():
    print("[INFO] Cargando data NPZ...")
    data = np.load(DATA_PATH)
    images = data["images"]
    labels = data["labels"]  # por si se quiere colorear luego
    print("[INFO] Imágenes:", images.shape)
    return images, labels

def prepare_flatten(images):
    """
    Convierte imágenes (N,16,16,3) a (N,768)
    """
    print("[INFO] Aplanando imágenes para PCA...")
    N = images.shape[0]
    flat = images.reshape(N, -1)
    print("[INFO] Matriz final:", flat.shape)
    return flat

def run_pca(flat_images, n_components=50):
    """
    PCA con suficientes componentes para análisis posterior.
    """
    print("[INFO] Ejecutando PCA con", n_components, "componentes...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(flat_images)
    return pca

def plot_variance(pca):
    """
    Save plot de variancia explicada.
    """
    ensure_dir(OUTPUT_DIR)
    plt.figure(figsize=(8,5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.title("PCA - Variancia explicada acumulada")
    plt.xlabel("Número de componentes")
    plt.ylabel("Varianza explicada acumulada")
    plt.grid()
    save_path = os.path.join(OUTPUT_DIR, "pca_variance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("[SUCCESS] Variancia guardada en:", save_path)

def reconstruct_and_save(pca, flat_images, sample_idx, k):
    """
    Reconstruye imagen sample_idx usando k componentes.
    """
    # PCA parcial
    pca_k = PCA(n_components=k, random_state=42)
    pca_k.fit(flat_images)

    encoded = pca_k.transform(flat_images[sample_idx:sample_idx+1])
    decoded = pca_k.inverse_transform(encoded)

    # Reconstruir forma original
    decoded_img = decoded.reshape(16,16,3)

    plt.figure(figsize=(3,3))
    plt.imshow(decoded_img)
    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, f"pca_reconstruction_k{k}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Reconstrucción k={k} guardada en:", save_path)

def main():
    print("\n=== 5.1 PCA Analysis ===\n")

    # 1. Load data
    images, labels = load_data()

    # 2. Prepare flattened matrix
    flat = prepare_flatten(images)

    # 3. PCA general
    pca = run_pca(flat, n_components=50)

    # 4. Variance plot
    plot_variance(pca)

    # 5. Reconstrucciones con k = 10, 20, 30
    sample_idx = 0  # primera imagen como referencia
    for k in [10, 20, 30]:
        reconstruct_and_save(pca, flat, sample_idx, k)

    print("\n=== COMPLETADO PCA 5.1 ===\n")


if __name__ == "__main__":
    main()
