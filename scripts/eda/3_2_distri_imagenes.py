import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# --- CONFIGURACIÓN DE RUTAS ---
rutas = {
    "Originales": r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images",
    "Únicas": r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\imagen_unica"
}

def analyze_folder_stats(folder_path, max_images=2000):
    """
    Calcula los valores RGB de todas las imágenes y la imagen promedio.
    max_images: Límite de seguridad para no saturar memoria si hay miles de archivos.
    """
    r_vals, g_vals, b_vals = [], [], []
    sum_img = np.zeros((16, 16, 3), dtype=np.float32)
    count = 0
    
    if not os.path.exists(folder_path):
        print(f"Advertencia: La ruta no existe -> {folder_path}")
        return None, None, None, None

    print(f"Analizando ruta: {folder_path}...")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    # Cargar, convertir a RGB y asegurar 16x16
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((16, 16))
                    arr = np.array(img)
                    
                    # Muestreo de píxeles para histogramas (aplanamos)
                    # Tomamos todos los píxeles de la imagen
                    r_vals.extend(arr[:, :, 0].flatten())
                    g_vals.extend(arr[:, :, 1].flatten())
                    b_vals.extend(arr[:, :, 2].flatten())
                    
                    # Acumulador para el promedio visual
                    sum_img += arr
                    count += 1
                    
                    if count >= max_images:
                        break
                except Exception as e:
                    pass
        if count >= max_images:
            break
            
    if count == 0:
        return None, None, None, None

    # Calcular imagen promedio
    mean_img = sum_img / count
    mean_img = np.clip(mean_img, 0, 255).astype(np.uint8)
    
    print(f"  -> Procesadas {count} imágenes.")
    return np.array(r_vals), np.array(g_vals), np.array(b_vals), mean_img

# --- VISUALIZACIÓN COMPARATIVA ---
# 2 Filas (Original vs Unicas) x 2 Columnas (Histograma vs Promedio)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.2)

row_idx = 0
for nombre, ruta in rutas.items():
    r_data, g_data, b_data, avg_img = analyze_folder_stats(ruta)
    
    if r_data is not None:
        # --- Columna 1: Histograma RGB ---
        ax_hist = axes[row_idx, 0]
        sns.histplot(r_data, color='red', element="step", stat="density", fill=False, label='R', ax=ax_hist)
        sns.histplot(g_data, color='green', element="step", stat="density", fill=False, label='G', ax=ax_hist)
        sns.histplot(b_data, color='blue', element="step", stat="density", fill=False, label='B', ax=ax_hist)
        
        ax_hist.set_title(f"Distribución RGB - {nombre}")
        ax_hist.set_xlabel("Intensidad (0=Negro, 255=Brillante)")
        ax_hist.set_ylabel("Densidad")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.2)
        
        # --- Columna 2: Imagen Promedio ---
        ax_img = axes[row_idx, 1]
        ax_img.imshow(avg_img)
        ax_img.set_title(f"Imagen Promedio - {nombre}\n(Centroide Visual)")
        ax_img.axis('off')
        
    else:
        # Si falla la carga, mostramos mensaje en el plot
        axes[row_idx, 0].text(0.5, 0.5, "Sin datos", ha='center')
        axes[row_idx, 1].text(0.5, 0.5, "Sin datos", ha='center')
        
    row_idx += 1

print("\nGenerando gráficos comparativos...")
plt.show()