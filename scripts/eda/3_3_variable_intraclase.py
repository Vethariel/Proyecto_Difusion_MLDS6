import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# --- CONFIGURACIÓN DE RUTAS ---
path_images = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images"
path_labels = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\sprites_labels.npy"

# Fijar la semilla para reproducibilidad (Punto clave de la solicitud)
np.random.seed(18) 

def compare_intraclass_mosaics(img_folder, lbl_file):
    # 1. Cargar y convertir etiquetas (manejo de OHE)
    if not os.path.exists(lbl_file):
        print(f"Error: No se encuentra el archivo en {lbl_file}")
        return

    labels_ohe = np.load(lbl_file)
    integer_labels = np.argmax(labels_ohe, axis=1) # Convierte OHE a índice (0, 1, 2...)
    
    # 2. Obtener clases únicas y conteos
    unique_classes, counts = np.unique(integer_labels, return_counts=True)
    
    # 3. Filtrar y Seleccionar 2 Clases Aleatorias
    
    # Solo consideramos clases con al menos 25 imágenes para llenar el mosaico 5x5 completo
    valid_classes = unique_classes[counts >= 25]
    
    if len(valid_classes) < 2:
        print(f"⚠️ Error: Solo hay {len(valid_classes)} clases con 25 o más imágenes. No se puede hacer una comparación 5x5.")
        print("Asegúrate de que hay al menos 2 categorías bien pobladas.")
        return

    # Seleccionamos 2 clases diferentes al azar (gracias a la semilla 18, siempre serán las mismas)
    target_classes = np.random.choice(valid_classes, 2, replace=False)
    
    print(f"Seleccionadas dos clases al azar (Semilla 18): Clase {target_classes[0]} y Clase {target_classes[1]}")

    # Listar archivos (Asumiendo orden alfabético para corresponder con índices)
    files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Validar consistencia
    limit = min(len(files), len(integer_labels))
    files = files[:limit]
    integer_labels = integer_labels[:limit]
    
    # 4. Generar Mosaico Comparativo (5 filas x 10 columnas para dos mosaicos 5x5 lado a lado)
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(f"Comparación de Variabilidad Intra-Clase (Semilla 18)\nMosaico Izquierda: Clase {target_classes[0]} | Mosaico Derecha: Clase {target_classes[1]}", fontsize=16)

    for mosaic_index, target_class in enumerate(target_classes):
        
        # Filtramos los índices de la clase actual
        indices_class = np.where(integer_labels == target_class)[0]
        
        # Seleccionamos 25 índices al azar de esa clase
        selected_indices = np.random.choice(indices_class, 25, replace=False)
        
        # Plotting
        for i, idx in enumerate(selected_indices):
            
            # Cálculo de la posición en la cuadrícula 5x10:
            row = i // 5            # Fila 0 a 4
            col = (i % 5) + (mosaic_index * 5) # Columna 0 a 4 (izq) o 5 a 9 (der)
            
            # Subplot
            ax = fig.add_subplot(5, 10, (row * 10) + col + 1)
            
            img_name = files[idx]
            img_path = os.path.join(img_folder, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((16, 16), resample=Image.NEAREST)
                
                ax.imshow(img)
                ax.axis('off')
                
                # Etiqueta de clase en la parte superior del mosaico
                if i == 2: # Etiqueta central de la fila superior
                    ax.set_title(f"Clase {target_class}", fontsize=10, pad=10)
                    
            except Exception as e:
                ax.text(0.5, 0.5, "Error", ha='center', va='center')
                ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

# Ejecutar
compare_intraclass_mosaics(path_images, path_labels)