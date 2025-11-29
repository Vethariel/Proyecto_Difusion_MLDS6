import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================

# !!! MODIFICAR ESTA RUTA A TU RUTA LOCAL REAL !!!
# La ruta que indicaste es: D:\documentos\unal\diplomados\...\images\images
IMAGENES_PATH = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images"
IMAGEN_SIZE = (16, 16)
MAX_IMAGES_SAMPLE = 1000  # Limitar el muestreo si el dataset es muy grande
OUTPUT_DIR = "analisis_4_1_rgb"

# Crear directorio de salida si no existe
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Carpeta de salida creada: {OUTPUT_DIR}")

# ==============================================================================
# 2. RECOLECCIÓN Y PROCESAMIENTO DE DATOS
# ==============================================================================

def collect_image_data(base_path):
    """Busca archivos de imagen e infiere etiquetas a partir del nombre de la carpeta."""
    image_data = []
    all_files = glob.glob(os.path.join(base_path, '**', '*.jpg'), recursive=True)
    all_files.extend(glob.glob(os.path.join(base_path, '**', '*.jpeg'), recursive=True))

    if not all_files:
        print(f"\n¡ERROR! No se encontraron imágenes en la ruta: {base_path}")
        print("Verifica que la variable IMAGENES_PATH sea correcta y que haya archivos .jpg/.jpeg.")
        sys.exit(1)

    print(f"Se encontraron {len(all_files)} imágenes.")

    for file_path in all_files:
        label = os.path.basename(os.path.dirname(file_path))
        if label == "":
             label = os.path.basename(base_path)
        
        image_data.append({'path': file_path, 'label': label})

    df = pd.DataFrame(image_data)
    
    if len(df) > MAX_IMAGES_SAMPLE:
        print(f"Muestreando {MAX_IMAGES_SAMPLE} imágenes para agilizar el análisis.")
        df = df.sample(MAX_IMAGES_SAMPLE, random_state=42).reset_index(drop=True)

    return df

def load_and_extract_rgb(image_path, size=IMAGEN_SIZE):
    """Carga una imagen, la estandariza a RGB (16x16) y extrae los canales."""
    try:
        img = Image.open(image_path)
        img = img.resize(size)
        img_rgb = img.convert('RGB')
        np_img = np.array(img_rgb)
        
        R = np_img[:, :, 0].flatten()
        G = np_img[:, :, 1].flatten()
        B = np_img[:, :, 2].flatten()

        return R, G, B

    except Exception:
        return None, None, None

# Ejecutar recolección
df_files = collect_image_data(IMAGENES_PATH)

# Cargar y extraer píxeles
all_r, all_g, all_b, all_labels = [], [], [], []

print("\nExtrayendo valores de píxeles...")
for index, row in tqdm(df_files.iterrows(), total=len(df_files)):
    R, G, B = load_and_extract_rgb(row['path'])
    if R is not None:
        all_r.extend(R)
        all_g.extend(G)
        all_b.extend(B)
        all_labels.extend([row['label']] * len(R))

pixel_df = pd.DataFrame({
    'R': all_r,
    'G': all_g,
    'B': all_b,
    'Label': all_labels
})

print(f"\nNúmero total de píxeles procesados: {len(pixel_df)}.")

# ==============================================================================
# 3. ANÁLISIS Y GENERACIÓN DE SALIDAS
# ==============================================================================

print("\n--- 4.1 Canales RGB: Análisis en curso ---")

# --- 3.1 Histograma por canal (Global) ---
print("1. Generando Histogramas Globales...")
# Creamos la primera figura
plt.figure(figsize=(15, 4))
sns.set_style("whitegrid")

for i, channel in enumerate(['R', 'G', 'B']):
    plt.subplot(1, 3, i + 1)
    sns.histplot(pixel_df[channel], bins=50, kde=True, color=channel.lower(), stat='density')
    plt.title(f'Histograma de Valores del Canal {channel} (Global)')
    plt.xlabel(f'Valor de Píxel ({channel})')
    plt.ylabel('Densidad')
    plt.xlim(0, 255)

plt.tight_layout()
global_hist_path = os.path.join(OUTPUT_DIR, 'rgb_global_histograms.png')
plt.savefig(global_hist_path) # Guardamos la imagen
print(f"-> Histograma global guardado en: {global_hist_path}")


# --- 3.2 Distribución de valores RGB por clase ---
class_analysis_possible = len(pixel_df['Label'].unique()) > 1

if class_analysis_possible:
    print("2. Generando Distribuciones por Clase...")
    top_n_classes = pixel_df['Label'].value_counts().nlargest(5).index.tolist()
    df_top_classes = pixel_df[pixel_df['Label'].isin(top_n_classes)]
    df_melt = df_top_classes.melt(id_vars='Label', value_vars=['R', 'G', 'B'], var_name='Channel', value_name='Value')

    # Creamos la segunda figura (FacetGrid)
    g = sns.FacetGrid(df_melt, col="Channel", hue="Label", col_wrap=3, height=5, aspect=1.2, palette="deep")
    g.map_dataframe(sns.kdeplot, x="Value", clip=(0, 255), linewidth=2)
    g.add_legend(title="Clase")
    g.set_axis_labels("Valor de Píxel", "Densidad")
    g.set_titles(col_template="Distribución de Valores del Canal {col_name} por Clase")
    
    plt.tight_layout()
    class_dist_path = os.path.join(OUTPUT_DIR, 'rgb_class_distributions.png')
    g.savefig(class_dist_path) # Guardamos la imagen
    print(f"-> Distribución por clase (Top 5) guardada en: {class_dist_path}")
else:
    print("2. Distribución por Clase omitida.")

# --- 3.3 Estadísticas descriptivas por canal ---
print("3. Calculando Estadísticas Descriptivas...")

# Estadísticas globales
global_stats = pixel_df[['R', 'G', 'B']].describe().T
global_stats['Canal'] = ['R', 'G', 'B']

# CORRECCIÓN: 'max' debe estar en minúscula para coincidir con la salida de .describe().T
col_mapping_global = {'count': 'Conteo', 'mean': 'Media', 'std': 'StdDev', 
                      'min': 'Min', 'max': 'Max', '25%': 'Q1', '50%': 'Mediana', '75%': 'Q3'}
global_stats.rename(columns=col_mapping_global, inplace=True)
global_stats = global_stats[['Canal', 'Media', 'StdDev', 'Min', 'Q1', 'Mediana', 'Q3', 'Max']] # AHORA 'Max' EXISTE
global_stats_csv_path = os.path.join(OUTPUT_DIR, 'rgb_global_stats.csv')
global_stats.to_csv(global_stats_csv_path, index=False, float_format='%.2f')
print(f"-> Estadísticas globales guardadas en: {global_stats_csv_path}")

# Estadísticas por clase (Guardar en CSV)
if class_analysis_possible:
    class_stats = pixel_df.groupby('Label')[['R', 'G', 'B']].agg(['mean', 'std', 'min', 'max'])
    class_stats.columns = ['_'.join(col).strip() for col in class_stats.columns.values]
    class_stats.reset_index(inplace=True)
    
    col_mapping_class = {
        'R_mean': 'R_Media', 'R_std': 'R_StdDev', 'R_min': 'R_Min', 'R_max': 'R_Max',
        'G_mean': 'G_Media', 'G_std': 'G_StdDev', 'G_min': 'G_Min', 'G_max': 'G_Max',
        'B_mean': 'B_Media', 'B_std': 'B_StdDev', 'B_min': 'B_Min', 'B_max': 'B_Max',
    }
    class_stats.rename(columns=col_mapping_class, inplace=True)
    
    class_stats_display = class_stats[class_stats['Label'].isin(top_n_classes)]
    class_stats_display = class_stats_display[['Label', 'R_Media', 'R_StdDev', 'R_Min', 'R_Max',
                                               'G_Media', 'G_StdDev', 'G_Min', 'G_Max',
                                               'B_Media', 'B_StdDev', 'B_Min', 'B_Max']]

    class_stats_csv_path = os.path.join(OUTPUT_DIR, 'rgb_class_stats_top5.csv')
    class_stats_display.to_csv(class_stats_csv_path, index=False, float_format='%.2f')
    print(f"-> Estadísticas por clase (Top 5) guardadas en: {class_stats_csv_path}")
else:
    print("-> Estadísticas por clase omitidas.")

# ==============================================================================
# 4. VISUALIZACIÓN EN TERMINAL Y VENTANAS EMERGENTES
# ==============================================================================

print("\n" + "="*70)
print("     ✅ RESULTADOS DEL ANÁLISIS RGB GLOBAL (4.1) ✅")
print("="*70)

## Mostrar Estadísticas Globales
print("\n## 📊 Estadísticas Descriptivas Globales (Canales RGB)")
# Usamos to_string() para evitar la dependencia de 'tabulate'
print(global_stats.to_string(index=False, float_format="%.2f"))

print("\n---")

## Mostrar Gráficos
print("## 🖼️ Visualización de Gráficos")
print("Se han generado y guardado los archivos PNG.")
print("A continuación, se mostrarán los gráficos en ventanas emergentes interactivas.")

# Esta línea bloquea el script hasta que cierres las ventanas de los gráficos.
plt.show()

print("\n¡Análisis 4.1 completado! Cierre las ventanas de los gráficos para continuar.")