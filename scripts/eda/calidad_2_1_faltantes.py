# EL objetivo de este script es validar que todas las imagenes tengan una etiqueta asignada, y que a su vez todas las etiquetas esten asignadas
# a imagenes existentes. El código arroja la cantidad de etiquetas sin imagen y viceversa.

import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np

#=============================================
# Comparación de nombres
#=============================================
# --- RUTAS ---
RUTA_CARPETA = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images"
RUTA_CSV = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\new_labels.csv"

def generar_listas_nombres():
    # --- 1. Obtener lista de los nombres en la CARPETA ---
    if os.path.exists(RUTA_CARPETA):
        lista_img = os.listdir(RUTA_CARPETA)
    else:
        lista_img = []
        print("❌ Error: No se encontró la carpeta de imágenes.")

    # --- 2. Obtener lista del CSV ---
    if os.path.exists(RUTA_CSV):
        df = pd.read_csv(RUTA_CSV)
        # Limpieza: path/to/imagen.jpg -> imagen.jpg
        rutas_crudas = df['Image Path'].tolist()
        lista_csv = [os.path.basename(ruta.replace("\\", "/")) for ruta in rutas_crudas]
    else:
        lista_csv = []
        print("❌ Error: No se encontró el archivo CSV.")

    return lista_img, lista_csv

def comparar_listas(imagenes_en_disco, imagenes_en_csv):
    print("\n--- RESULTADOS DEL ANÁLISIS DE CALIDAD (2.1) ---")
    
    # Usamos SETS (conjuntos) para encontrar diferencias rápidamente
    set_disco = set(imagenes_en_disco)
    set_csv = set(imagenes_en_csv)

    # 1. IMÁGENES SIN ENTRADA EN CSV (Están en disco - Están en csv)
    sobran_en_disco = set_disco - set_csv
    
    # 2. RUTAS INVÁLIDAS (Están en csv - Están en disco)
    faltan_en_disco = set_csv - set_disco

    # 3. DUPLICADOS EN CSV (Lógica de listas, no de sets)
    # Comparamos el largo de la lista original vs el set de valores únicos
    num_duplicados = len(imagenes_en_csv) - len(set_csv)

    # --- REPORTE ---
    print(f"Total imágenes en carpeta: {len(imagenes_en_disco)}")
    print(f"Total registros en CSV:    {len(imagenes_en_csv)}")
    print("-" * 40)

    if len(sobran_en_disco) == 0:
        print("✅ TODAS las imágenes de la carpeta tienen su etiqueta en el CSV.")
    else:
        print(f"❌ Imágenes sin entrada en CSV: {len(sobran_en_disco)}")
        print(f"   Ejemplos: {list(sobran_en_disco)[:3]}")

    if len(faltan_en_disco) == 0:
        print("✅ TODAS las rutas del CSV apuntan a imágenes existentes.")
    else:
        print(f"❌ Rutas inválidas en CSV (no existen en disco): {len(faltan_en_disco)}")
        print(f"   Ejemplos: {list(faltan_en_disco)[:3]}")

    if num_duplicados == 0:
        print("✅ No hay nombres duplicados en el CSV.")
    else:
        print(f"⚠️ Se encontraron {num_duplicados} registros duplicados en el CSV.")

#=============================================
# Validación de rutas y calidad de imagenes
#=============================================

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Usamos rutas crudas (r"...") para evitar problemas con las barras invertidas en Windows
RUTA_CARPETA = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images\images"
RUTA_CSV = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\new_labels.csv"

def auditoria_tecnica():
    print("--- INICIANDO AUDITORÍA TÉCNICA (RUTAS Y LECTURA) ---\n")

    # ---------------------------------------------------------
    # PASO A: CARGA DE DATOS
    # ---------------------------------------------------------
    if not os.path.exists(RUTA_CARPETA) or not os.path.exists(RUTA_CSV):
        print("❌ Error Crítico: No se encuentra la carpeta de imágenes o el CSV.")
        return

    # 1. Obtener archivos reales en disco (Set para búsqueda rápida)
    archivos_en_disco = set(os.listdir(RUTA_CARPETA))
    print(f"📂 Archivos físicos encontrados: {len(archivos_en_disco)}")

    # 2. Obtener nombres esperados según el CSV
    df = pd.read_csv(RUTA_CSV)
    if 'Image Path' not in df.columns:
        print("❌ Error: Columna 'Image Path' no encontrada en CSV.")
        return

    # Limpiamos las rutas del CSV para quedarnos solo con el nombre del archivo
    # Ejemplo: 'path/to/img_01.jpg' -> 'img_01.jpg'
    nombres_en_csv = [os.path.basename(ruta.replace("\\", "/")) for ruta in df['Image Path']]
    set_csv = set(nombres_en_csv)
    print(f"📄 Registros leídos del CSV: {len(set_csv)}")

    # ---------------------------------------------------------
    # PASO B: DETECCIÓN DE RUTAS INVÁLIDAS
    # ---------------------------------------------------------
    print("\n--- 1. VERIFICANDO RUTAS INVÁLIDAS ---")
    
    # Rutas en CSV que NO están en el disco (Resta de conjuntos)
    rutas_invalidas = set_csv - archivos_en_disco
    
    if len(rutas_invalidas) == 0:
        print("✅ ÉXITO: Todas las rutas del CSV apuntan a archivos existentes.")
    else:
        print(f"❌ FALLO: Se encontraron {len(rutas_invalidas)} rutas inválidas en el CSV.")
        print(f"   (Archivos que el CSV dice que existen, pero no están en la carpeta)")
        print(f"   Ejemplos: {list(rutas_invalidas)[:3]}")

    # ---------------------------------------------------------
    # PASO C: DETECCIÓN DE ERRORES DE LECTURA (PIL & OPENCV)
    # ---------------------------------------------------------
    print("\n--- 2. VERIFICANDO INTEGRIDAD DE ARCHIVOS (LECTURA) ---")
    print("   (Esto puede tomar unos momentos dependiendo de la cantidad de imágenes...)")

    archivos_corruptos = []
    lista_archivos_disco = list(archivos_en_disco) # Convertimos a lista para iterar
    total = len(lista_archivos_disco)

    for i, nombre_archivo in enumerate(lista_archivos_disco):
        ruta_completa = os.path.join(RUTA_CARPETA, nombre_archivo)
        error_detectado = None

        # --- PRUEBA 1: PIL (Python Imaging Library) ---
        # Detecta archivos truncados o cabeceras rotas rápidamente
        try:
            with Image.open(ruta_completa) as img:
                img.verify() # Verifica la estructura del archivo sin decodificarlo todo (rápido)
        except (IOError, SyntaxError, UnidentifiedImageError) as e:
            error_detectado = f"PIL Error: {str(e)}"

        # --- PRUEBA 2: OpenCV (cv2) ---
        # Si PIL pasó, hacemos doble chequeo con OpenCV para asegurar que se puede procesar como matriz
        if error_detectado is None:
            try:
                img_cv = cv2.imread(ruta_completa)
                if img_cv is None:
                    error_detectado = "OpenCV Error: No se pudo decodificar (None)"
                # Opcional: Verificar si la imagen tiene dimensiones válidas (no es 0x0)
                elif img_cv.size == 0:
                    error_detectado = "OpenCV Error: Imagen vacía (size 0)"
            except Exception as e:
                error_detectado = f"OpenCV Exception: {str(e)}"

        # Si hubo algún error, lo registramos
        if error_detectado:
            archivos_corruptos.append((nombre_archivo, error_detectado))

        # Barra de progreso simple en consola
        if i % 100 == 0:
            print(f"   Procesando: {i}/{total} imágenes...", end="\r")

    print(f"   Procesamiento finalizado: {total}/{total}       ")

    # --- REPORTE FINAL DE LECTURA ---
    if len(archivos_corruptos) == 0:
        print("✅ ÉXITO: Todas las imágenes son legibles (PIL y OpenCV).")
    else:
        print(f"☠️ CRÍTICO: Se encontraron {len(archivos_corruptos)} imágenes corruptas.")
        print("   Ejemplos de errores:")
        for nombre, error in archivos_corruptos[:5]:
            print(f"   - {nombre}: {error}")

# --- PUNTO DE ENTRADA DE LAS REVISIONES ---
if __name__ == "__main__":
    # --- REVISIÓN DE LOS NOMBRES ---
    print(f"\n{'='*60}")
    # 1. Generamos las listas
    l_img, l_csv = generar_listas_nombres()

    # 2. Si ambas listas tienen datos, procedemos a comparar
    if len(l_img) > 0 and len(l_csv) > 0:
        comparar_listas(l_img, l_csv)
    else:
        print("No se pudo realizar la comparación por falta de datos.")
    print(f"\n{'='*60}")
    # --- EJECUCIÓN DE LA AUDITORIA DE IMAGENES ---
    auditoria_tecnica()
    print(f"\n{'='*60}")