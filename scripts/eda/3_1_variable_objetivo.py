# Anteriormente se observo la posibilidad de que cada una de las imagenes puede identificarse con un hash el cual se puede utilizar para determinar
# que imagenes se parecen entre si, con lo que se pudieron determinar conjuntos de imagenes parecidas entre si, por lo que queremos ver como es el
# comportamiento de la distribución de intensidad tanto del conjunto original de imagenes como el de imagenes únicas. Al final de la ejecución
# podemos observar el histograma de la distribución de cada conjunto, junto a una imagen muy probable dentro del conjunto y otra de ruido poco
# probable dentro del mismo.

# Los graficos de densidades son muy similares entre si, siendo los valores de intensidad cercanos a cero los que acumulan la mayor parte de la
# densidad, dandonos a entender que en todas las imagenes el color negro o cercanos a este predomina sobre los demás.

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

# --- CONFIGURACIÓN DE RUTAS ---
# Usamos r"" para evitar problemas con los backslash de Windows
rutas = {
    "Originales": r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images",
    "Únicas": r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\imagen_unica"
}

def load_images_as_array(folder_path, max_images=500):
    """Carga imágenes de una carpeta y las convierte en vectores aplanados."""
    image_list = []
    count = 0
    
    if not os.path.exists(folder_path):
        print(f"Advertencia: La ruta no existe -> {folder_path}")
        return np.array([])

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    # Abrir, convertir a RGB y asegurar tamaño 16x16
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((16, 16)) # Aseguramos dimensión
                    img_array = np.array(img)
                    image_list.append(img_array.flatten()) 
                    count += 1
                    if count >= max_images:
                        break
                except Exception as e:
                    print(f"Error cargando {file}: {e}")
        if count >= max_images:
            break
            
    return np.array(image_list)

# --- VISUALIZACIÓN ---
# Creamos una figura con 2 filas (una por dataset) y 2 columnas (Histo y Comparación)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4) # Espacio vertical entre filas

for i, (nombre_dataset, ruta) in enumerate(rutas.items()):
    print(f"Procesando: {nombre_dataset}...")
    data_matrix = load_images_as_array(ruta)
    
    # Validar si encontramos imágenes
    if data_matrix.size == 0:
        print(f"No se encontraron imágenes en {nombre_dataset}. Saltando gráfica.")
        continue

    # --- Columna 1: Histograma de Distribución ---
    ax_hist = axes[i, 0]
    sns.histplot(data_matrix.flatten(), bins=50, kde=True, color='teal' if i==0 else 'orange', stat="density", ax=ax_hist)
    ax_hist.set_title(f'Distribución de Píxeles - {nombre_dataset}\n(La "huella" de los datos)')
    ax_hist.set_xlabel('Intensidad (0-255)')
    ax_hist.set_ylabel('Densidad')
    ax_hist.grid(True, alpha=0.3)

    # --- Columna 2: Real vs Ruido ---
    ax_img = axes[i, 1]
    
    # Tomamos la primera imagen real disponible
    real_img = data_matrix[0].reshape(16, 16, 3).astype(np.uint8)
    
    # Generamos ruido aleatorio del mismo tamaño
    noise_img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    
    # Concatenamos horizontalmente
    comparison = np.hstack((real_img, noise_img))
    
    ax_img.imshow(comparison)
    ax_img.set_title(f'{nombre_dataset}: Real vs Ruido Aleatorio\n($x \sim P_{{data}}$ vs $x \sim P_{{ruido}}$)')
    ax_img.axis('off')

print("Generando gráficos...")
plt.show()