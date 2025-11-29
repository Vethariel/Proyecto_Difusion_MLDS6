import numpy as np

NPZ_PATH = "data/intermediate/pixel_art_data.npz"

def check_npz(npz_path):
    print(f"=== Inspeccionando {npz_path} ===\n")

    data = np.load(npz_path)
    keys = list(data.keys())

    print("[INFO] Claves encontradas:", keys, "\n")

    if "images" in keys:
        images = data["images"]
        print("→ images")
        print("  shape:", images.shape)
        print("  dtype:", images.dtype)
        print("  min/max:", images.min(), "/", images.max())
        print("  ejemplo de pixel:", images[0,0,0], "\n")

    if "labels" in keys:
        labels = data["labels"]
        print("→ labels")
        print("  shape:", labels.shape)
        print("  dtype:", labels.dtype)
        print("  valores únicos:", np.unique(labels)[:20], "\n")

    if "metadata" in keys:
        print("→ metadata")
        print(" ", data["metadata"], "\n")

    print("=== Fin de inspección ===")

check_npz(NPZ_PATH)
