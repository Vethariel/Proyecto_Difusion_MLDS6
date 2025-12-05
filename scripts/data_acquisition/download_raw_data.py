"""
TDSP - RAW DATA ACQUISITION
Este script descarga el dataset de Kaggle usando KaggleHub y lo mueve
a la carpeta data/raw/ siguiendo prácticas de MLOps.

OUTPUT:
    data/raw/
        images/
        labels.csv
        sprites.npy
        sprites_labels.npy
"""

import kagglehub
import shutil
import os
from datetime import datetime

RAW_DIR = "data/raw"
DATASET_NAME = "ebrahimelgazar/pixel-art"


def ensure_directory(path: str):
    """Crea carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Carpeta creada: {path}")
    else:
        print(f"[INFO] Carpeta ya existe: {path}")


def download_dataset():
    """Descarga el dataset usando KaggleHub."""
    print("[INFO] Descargando dataset desde KaggleHub...")
    path_local = kagglehub.dataset_download(DATASET_NAME)
    print(f"[INFO] Dataset descargado en caché: {path_local}")
    return path_local


def copy_to_raw(src_path: str, dst_path: str):
    """Copia todo el dataset descargado hacia data/raw/."""
    print("[INFO] Copiando archivos a data/raw/ ...")
    try:
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        print(f"[SUCCESS] Archivos copiados a: {dst_path}")
    except Exception as e:
        print(f"[ERROR] Fallo al copiar archivos: {e}")
        raise e


def log_metadata(dst_path: str, src_path: str):
    """Guarda un archivo metadata.txt con información del dataset."""
    metadata_path = os.path.join(dst_path, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("RAW DATA INGESTION - METADATA\n")
        f.write("==============================\n\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Fecha de descarga: {datetime.utcnow()} UTC\n")
        f.write(f"Fuente: KaggleHub\n")
        f.write(f"Ruta de caché descargada: {src_path}\n\n")
        f.write("Descripción: Datos crudos sin modificación.\n")

    print(f"[INFO] Metadata registrada en: {metadata_path}")


def main():
    """Pipeline completo de adquisición."""
    print("\n=== TDSP - RAW DATA INGESTION ===\n")

    # 1. Asegurar carpeta raw
    ensure_directory(RAW_DIR)

    # 2. Descargar dataset
    cache_path = download_dataset()

    # 3. Copiar al repositorio (data/raw)
    copy_to_raw(cache_path, RAW_DIR)

    # 4. Registrar metadata
    log_metadata(RAW_DIR, cache_path)

    # 5. Mostrar contenido básico
    print("\n[INFO] Archivos en data/raw/")
    try:
        print(os.listdir(RAW_DIR))
    except Exception:
        pass

    print("\n=== COMPLETADO ===")


if __name__ == "__main__":
    main()
