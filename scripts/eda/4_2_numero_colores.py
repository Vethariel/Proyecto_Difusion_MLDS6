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
IMAGENES_PATH = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images"
IMAGEN_SIZE = (16, 16)
MAX_PIXELS = IMAGEN_SIZE[0] * IMAGEN_SIZE[1]  # 256 píxeles máximos
MAX_IMAGES_SAMPLE = 8000
TEMP_FILENAME = "4_2_stats_temp.csv" # Usamos un nombre de archivo temporal en caso de guardar

# ==============================================================================
# 2. RECOLECCIÓN Y CONTEO DE COLORES ÚNICOS
# ==============================================================================

def collect_image_data(base_path):
    """Busca archivos de imagen e infiere etiquetas."""
    image_data = []
    all_files = glob.glob(os.path.join(base_path, '**', '*.jpg'), recursive=True)
    all_files.extend(glob.glob(os.path.join(base_path, '**', '*.jpeg'), recursive=True))

    if not all_files:
        print(f"\n¡ERROR! No se encontraron imágenes en la ruta: {base_path}")
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

def count_unique_colors(image_path):
    """Carga la imagen, estandariza a 16x16 RGB y cuenta los colores únicos."""
    try:
        img = Image.open(image_path)
        img = img.resize(IMAGEN_SIZE)
        img_rgb = img.convert('RGB')

        # Usamos getcolors() con maxcolors=256 (el máx. de 16x16)
        colors = img_rgb.getcolors(MAX_PIXELS + 1)
        
        # Si la imagen está completamente vacía o hubo un error, colors puede ser None o 0
        return len(colors) if colors else 0

    except Exception:
        return 0

df_files = collect_image_data(IMAGENES_PATH)

# Aplicar el conteo de colores a todo el DataFrame
print("\nContando colores únicos por imagen...")
df_files['unique_colors'] = [count_unique_colors(p) for p in tqdm(df_files['path'], total=len(df_files))]

# Eliminar filas con 0 colores (posibles errores de carga)
df_files = df_files[df_files['unique_colors'] > 0].reset_index(drop=True)

print(f"\nNúmero total de imágenes válidas para análisis: {len(df_files)}.")

# ==============================================================================
# 3. ANÁLISIS 4.2: PALETA DE COLORES
# ==============================================================================

print("\n--- 4.2 Número de Colores: Análisis en curso ---")

# --- 3.1 & 3.2 Conteo, Estadísticas y Distribución (Global y por Etiqueta) ---

# Estadísticas Descriptivas Globales
global_color_stats = df_files['unique_colors'].describe().to_frame().T
global_color_stats.index = ['Global']
global_color_stats = global_color_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
global_color_stats.rename(columns={'count': 'Conteo_Imgs', 'mean': 'Media_Colores', 'std': 'StdDev', 
                                    'min': 'Min_Colores', 'max': 'Max_Colores', 
                                    '50%': 'Mediana_Colores'}, inplace=True)

# Estadísticas Descriptivas por Etiqueta (Top 5 clases)
class_analysis_possible = len(df_files['label'].unique()) > 1
if class_analysis_possible:
    top_n_classes = df_files['label'].value_counts().nlargest(5).index.tolist()
    class_color_stats = df_files[df_files['label'].isin(top_n_classes)].groupby('label')['unique_colors'].describe()
    class_color_stats.rename(columns={'count': 'Conteo_Imgs', 'mean': 'Media_Colores', 'std': 'StdDev', 
                                        'min': 'Min_Colores', 'max': 'Max_Colores', 
                                        '50%': 'Mediana_Colores'}, inplace=True)

# --- 3.3 Clasificación por Paleta ---

# Definición de umbrales (máximo 256 colores para 16x16)
LOW_PALETTE_MAX = 32
MID_PALETTE_MAX = 128

def classify_palette(count):
    if count <= LOW_PALETTE_MAX:
        return 'Low_Palette (<=32)'
    elif count <= MID_PALETTE_MAX:
        return 'Mid_Palette (33-128)'
    else:
        return 'High_Palette (>128)'

df_files['palette_class'] = df_files['unique_colors'].apply(classify_palette)

# Conteo por clase de paleta
palette_counts = df_files['palette_class'].value_counts().sort_index()
palette_counts_df = palette_counts.to_frame(name='Conteo').reset_index()
palette_counts_df.rename(columns={'index': 'Tipo_Paleta'}, inplace=True)


# ==============================================================================
# 4. VISUALIZACIÓN EN TERMINAL Y VENTANAS EMERGENTES
# ==============================================================================

print("\n" + "="*70)
print("     ✅ RESULTADOS DEL ANÁLISIS DE PALETA DE COLORES (4.2) ✅")
print("="*70)

## Mostrar Clasificación por Paleta
print("\n## 🎨 Distribución de Imágenes por Tipo de Paleta")
print(palette_counts_df.to_string(index=False))

print("\n---")

## Mostrar Estadísticas Globales
print("\n## 📊 Estadísticas Globales de Colores Únicos por Imagen")
print(global_color_stats[['Conteo_Imgs', 'Media_Colores', 'StdDev', 'Min_Colores', 'Mediana_Colores', 'Max_Colores']].to_string(float_format="%.2f"))

if class_analysis_possible:
    print("\n## 📈 Estadísticas de Colores Únicos por Clase (Top 5)")
    print(class_color_stats[['Conteo_Imgs', 'Media_Colores', 'StdDev', 'Min_Colores', 'Mediana_Colores', 'Max_Colores']].to_string(float_format="%.2f"))

print("\n---")

## Gráfico 1: Histograma de Colores Únicos (Global)
plt.figure(figsize=(8, 6))
sns.histplot(df_files['unique_colors'], bins=20, kde=True, color='purple')
plt.title('Distribución Global del Número de Colores Únicos por Imagen')
plt.xlabel('Número de Colores Únicos (Máx 256)')
plt.ylabel('Frecuencia (Conteo de Imágenes)')
plt.axvline(x=LOW_PALETTE_MAX, color='r', linestyle='--', label=f'Low/Mid Cutoff ({LOW_PALETTE_MAX})')
plt.axvline(x=MID_PALETTE_MAX, color='g', linestyle='--', label=f'Mid/High Cutoff ({MID_PALETTE_MAX})')
plt.legend()
plt.tight_layout()


## Gráfico 2: Relación entre Colores Únicos y Etiqueta (Top 5)
if class_analysis_possible:
    plt.figure(figsize=(10, 6))
    df_top_classes = df_files[df_files['label'].isin(top_n_classes)]
    sns.boxplot(x='label', y='unique_colors', data=df_top_classes, palette="viridis")
    plt.title(f'Relación Colores Únicos vs. Etiqueta (Top {len(top_n_classes)} Clases)')
    plt.xlabel('Clase (Etiqueta)')
    plt.ylabel('Número de Colores Únicos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


print("## 🖼️ Visualización de Gráficos")
print("Se mostrarán los gráficos de Distribución Global y Relación por Clase en ventanas emergentes.")

# Muestra todas las figuras creadas
plt.show()

print("\n¡Análisis 4.2 completado! Cierre las ventanas de los gráficos para continuar.")