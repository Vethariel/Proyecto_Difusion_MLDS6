"""
5.2 - Importancia del color en el dataset
TDSP - EDA

Este script analiza qué canal de color (R, G, B) aporta mayor variabilidad
y cómo cambia entre clases del dataset. Además genera gráficas y figuras
para el informe final.

Salida:
    - Figuras PNG en reports/figures/eda/
    - Métricas impresas en consola
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# =====================
# CONFIGURACIÓN
# =====================
DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/color")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# UTILIDADES
# =====================
def colorfulness_metric(img):
    """Métrica de Hasler & Süsstrunk"""
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return (
        np.sqrt(np.std(rg)**2 + np.std(yb)**2) +
        0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    )

# =====================
# CARGA DE DATOS
# =====================
data = np.load(DATA_PATH)
images = data["images"]       # (89400, 16, 16, 3)
labels = data["labels"]       # (89400,)
num_classes = len(np.unique(labels))

flattened = images.reshape(images.shape[0], -1, 3)  # (N,256,3)

print(f"[INFO] Dataset cargado: {images.shape}")
print(f"[INFO] Clases detectadas: {num_classes}")

# =====================
# 1. VARIANZA GLOBAL
# =====================
global_var = np.var(flattened, axis=(0,1))

# GRAFICO: Barras globales
plt.figure(figsize=(6,4))
plt.bar(["R","G","B"], global_var, color=["red","green","blue"])
plt.title("Varianza global por canal")
plt.ylabel("Varianza")
plt.savefig(FIG_DIR / "global_variance.png", dpi=200)
plt.close()

print("\n=== VARIANZA GLOBAL ===")
print(f"R={global_var[0]:.6f}, G={global_var[1]:.6f}, B={global_var[2]:.6f}")


# =====================
# 2. VARIANZA POR CLASE
# =====================
class_variances = {}

for c in range(num_classes):
    imgs_c = images[labels == c]
    flat_c = imgs_c.reshape(imgs_c.shape[0], -1, 3)
    class_variances[c] = np.var(flat_c, axis=(0,1))

# GRAFICO: Barras por clase
plt.figure(figsize=(10,6))
width = 0.25

x = np.arange(num_classes)
plt.bar(x - width, [class_variances[c][0] for c in range(num_classes)], width, label="R", color="red")
plt.bar(x       , [class_variances[c][1] for c in range(num_classes)], width, label="G", color="green")
plt.bar(x + width, [class_variances[c][2] for c in range(num_classes)], width, label="B", color="blue")

plt.xticks(x, [f"Clase {c}" for c in range(num_classes)])
plt.title("Varianza por canal en cada clase")
plt.legend()
plt.savefig(FIG_DIR / "variance_by_class.png", dpi=200)
plt.close()

print("\n=== VARIANZA POR CLASE ===")
for c in range(num_classes):
    print(f"Clase {c}: R={class_variances[c][0]:.6f}, G={class_variances[c][1]:.6f}, B={class_variances[c][2]:.6f}")


# =====================
# 3. COLORFULNESS POR CLASE
# =====================
colorfulness_by_class = {}
for c in range(num_classes):
    imgs_c = images[labels == c][:300]  # muestreo
    colorfulness_by_class[c] = np.mean([colorfulness_metric(img) for img in imgs_c])

plt.figure(figsize=(8,5))
plt.bar([f"C{c}" for c in range(num_classes)],
        [colorfulness_by_class[c] for c in range(num_classes)],
        color="purple")
plt.title("Colorfulness (Hasler & Süsstrunk) por clase")
plt.ylabel("Colorfulness")
plt.savefig(FIG_DIR / "colorfulness_by_class.png", dpi=200)
plt.close()

print("\n=== COLORFULNESS ===")
for c in range(num_classes):
    print(f"Clase {c}: {colorfulness_by_class[c]:.4f}")


# =====================
# 4. PCA POR CANAL
# =====================
pca_results = {}
for i, channel in enumerate(["R","G","B"]):
    channel_data = flattened[..., i].reshape(images.shape[0], -1)
    pca = PCA(n_components=1)
    pca.fit(channel_data)
    pca_results[channel] = pca.explained_variance_ratio_[0]

plt.figure(figsize=(6,4))
plt.bar(["R","G","B"], [pca_results["R"], pca_results["G"], pca_results["B"]],
        color=["red","green","blue"])
plt.ylabel("Varianza explicada (PC1)")
plt.title("PCA por canal - Importancia de información")
plt.savefig(FIG_DIR / "pca_by_channel.png", dpi=200)
plt.close()

print("\n=== PCA POR CANAL ===")
for k,v in pca_results.items():
    print(f"{k}: {v:.4f}")


# =====================
# 5. RANKING FINAL
# =====================
ranking = {
    "R": global_var[0] + pca_results["R"],
    "G": global_var[1] + pca_results["G"],
    "B": global_var[2] + pca_results["B"]
}

print("\n=== RANKING FINAL DE IMPORTANCIA ===")
for ch,score in sorted(ranking.items(), key=lambda x: -x[1]):
    print(f"{ch}: {score:.4f}")

print("\nFiguras guardadas en:", FIG_DIR)
print("\n=== COMPLETADO 5.2 ===")
