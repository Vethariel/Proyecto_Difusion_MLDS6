"""
TDSP - RAW → INTERMEDIATE TRANSFORMATION
Convierte sprites.npy y sprites_labels.npy a un archivo .npz optimizado.

INPUT:
    data/raw/sprites.npy
    data/raw/sprites_labels.npy

OUTPUT:
    data/intermediate/pixel_art_data.npz
        - images (float32, normalizadas)
        - labels (int32)
"""

import os
import numpy as np
from datetime import datetime

RAW_DIR = "data/raw"
INTERMEDIATE_DIR = "data/intermediate"
OUTPUT_FILE = os.path.join(INTERMEDIATE_DIR, "pixel_art_data.npz")


def ensure_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Carpeta creada: {path}")


def load_raw_arrays():
    """Carga sprites.npy y sprites_labels.npy desde data/raw."""
    sprites_path = os.path.join(RAW_DIR, "sprites.npy")
    labels_path = os.path.join(RAW_DIR, "sprites_labels.npy")

    if not os.path.exists(sprites_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("[ERROR] Falta sprites.npy o sprites_labels.npy en data/raw")

    print("[INFO] Cargando sprites.npy ...")
    images = np.load(sprites_path)

    print("[INFO] Cargando sprites_labels.npy ...")
    labels = np.load(labels_path)

    print(f"[INFO] Imágenes cargadas: {images.shape}")
    print(f"[INFO] Etiquetas cargadas: {labels.shape}")

    return images, labels


def normalize_images(images: np.ndarray):
    """
    Asegura formato 16x16x3, convierte a float32 y normaliza a 0–1.
    """
    # Si vienen como (N, 3, 16, 16), convertir a (N, 16, 16, 3)
    if images.shape[1] == 3:
        print("[INFO] Detectado formato CHW. Reordenando a HWC...")
        images = images.transpose(0, 2, 3, 1)

    print("[INFO] Convertiendo a float32...")
    images = images.astype(np.float32)

    print("[INFO] Normalizando imágenes a rango 0–1...")
    images /= 255.0

    return images

def convert_labels(labels_one_hot: np.ndarray):
    """Convierte matriz one-hot (N×5) a vector entero (N,)."""
    if labels_one_hot.ndim != 2:
        raise ValueError("[ERROR] labels_one_hot debe tener forma (N, 5)")

    print("[INFO] Convirtiendo etiquetas one-hot a clases enteras...")
    labels = np.argmax(labels_one_hot, axis=1).astype(np.int32)

    print(f"[INFO] Etiquetas enteras: shape {labels.shape}, únicas: {np.unique(labels)}")
    return labels

def save_npz(images, labels, labels_one_hot):
    """Guarda el archivo .npz final."""
    print(f"[INFO] Guardando archivo final en {OUTPUT_FILE} ...")

    np.savez_compressed(
        OUTPUT_FILE,
        images=images,
        labels=labels,
        labels_one_hot=labels_one_hot,
        metadata=f"Creado {datetime.utcnow()} UTC"
    )

    print("[SUCCESS] Archivo guardado:", OUTPUT_FILE)


def main():
    print("\n=== TDSP RAW → INTERMEDIATE ===\n")

    ensure_directory(INTERMEDIATE_DIR)

    # 1. Cargar datasets crudos
    images, labels_one_hot = load_raw_arrays()

    # 2. Normalizar imágenes
    images = normalize_images(images)

    # 3. Convertir etiquetas
    labels = convert_labels(labels_one_hot)

    # 4. Guardar en formato NPZ
    save_npz(images, labels, labels_one_hot)

    print("\n=== COMPLETADO ===")


if __name__ == "__main__":
    main()
