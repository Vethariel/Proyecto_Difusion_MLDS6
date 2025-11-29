import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN DE RUTAS ---
path_images = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images"
path_labels = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\sprites_labels.npy"

# Se asume que el tamaño es 16x16x3 = 768
N_COMPONENTS_MAX = 768

def load_and_preprocess_images(folder_path, max_samples=2000):
    """Carga imágenes, asegura 16x16 y las aplana a un vector de 768."""
    image_list = []

    # Listar archivos (asumiendo orden para consistencia)
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i, file in enumerate(files[:max_samples]):
        img_path = os.path.join(folder_path, file)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((16, 16), resample=Image.NEAREST)
            image_list.append(np.array(img).flatten()) # Vector de 768
        except Exception as e:
            # print(f"Error al cargar {file}: {e}")
            pass

    data_matrix = np.array(image_list, dtype=np.float32)
    print(f"Datos cargados: {data_matrix.shape}")

    return data_matrix

def analyze_pca_variance(data_matrix):

    # 1. Estandarización de los datos (Paso crucial para PCA)
    # Es necesario que las componentes tengan media 0 y desviación estándar 1
    print("Estandarizando datos...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)

    # 2. Aplicar PCA (Calculamos hasta el máximo de componentes posibles)
    print(f"Aplicando PCA con {N_COMPONENTS_MAX} componentes...")
    pca = PCA(n_components=N_COMPONENTS_MAX)
    pca.fit(scaled_data)

    # 3. Calcular la Varianza Explicada Acumulada
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)

    # 4. Encontrar los puntos de corte de varianza
    # Buscamos el número de componentes necesarios para explicar el 80% y 95%
    n_80 = np.argmax(cumulative_variance >= 0.80) + 1
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1

    print("\n--- Resultados de Varianza Explicada ---")
    print(f"El 80% de la varianza se explica con {n_80} componentes.")
    print(f"El 95% de la varianza se explica con {n_95} componentes.")
    print(f"Las primeras 10 componentes explican el {cumulative_variance[9]*100:.2f}% de la varianza total.")

    # 5. Visualización del Gráfico de Codo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_COMPONENTS_MAX + 1), cumulative_variance, marker='.', linestyle='--', color='blue')
    plt.title('Varianza Explicada Acumulada por Componentes Principales')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Acumulada Explicada')

    # Dibujar líneas de referencia
    plt.axhline(y=0.80, color='red', linestyle='-', label='80% de Varianza')
    plt.axvline(x=n_80, color='red', linestyle='--')
    plt.axhline(y=0.95, color='green', linestyle='-', label='95% de Varianza')
    plt.axvline(x=n_95, color='green', linestyle='--')

    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.xlim(0, max(n_95 + 10, 100)) # Limita el eje X para no mostrar las 768 si no son necesarias
    plt.show()


# --- Ejecución ---
image_data = load_and_preprocess_images(path_images)

if image_data.size > 0:
    analyze_pca_variance(image_data)
else:
    print("No hay datos para analizar.")