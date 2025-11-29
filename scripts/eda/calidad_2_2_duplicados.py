# El siguiente código revisa los posibles duplicados utilziando la libreria imagehash, de esta revisión se crea una carpeta en la cual se
# almacenan los representantes de cada una de las clases identificadas al momento de comparar el parecido a trvés del hash. También se
# crea un archivo JSON el cual almacena en forma de diccionario en el cual las claves son el primer hash idnetificado y los valores son
# la lista de nombres de los archivos que resultan iguales a ese hash

import os
import shutil
import json
from PIL import Image
import imagehash

# --- CONFIGURACIÓN DE RUTAS ---
# Ruta Base dada (apunta a .../datos_proyecto/images)
BASE_PATH = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\images"

# Ruta de Origen (donde están todas las imágenes)
RUTA_ORIGEN = os.path.join(BASE_PATH, "images")

# Ruta de Destino (donde quedarán las únicas)
RUTA_DESTINO = os.path.join(BASE_PATH, "imagen_unica")

# Ruta para el archivo JSON (Lo guardaremos un nivel arriba, en 'datos_proyecto')
DIR_PROYECTO = os.path.dirname(BASE_PATH) 
RUTA_JSON = os.path.join(DIR_PROYECTO, "listados_hash.json")

def filtrar_y_copiar_unicos():
    print("--- 1. GENERANDO HASHES Y DETECTANDO DUPLICADOS ---")
    
    if not os.path.exists(RUTA_ORIGEN):
        print("❌ Error: No se encuentra la carpeta de origen.")
        return

    # Obtenemos lista de archivos
    archivos = [f for f in os.listdir(RUTA_ORIGEN) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_archivos = len(archivos)
    print(f"📂 Total imágenes a analizar: {total_archivos}")

    # Diccionario: { "hash": ["img1.png", "img1_copia.png"] }
    hashes_dict = {}
    
    # --- FASE 1: Llenado del Diccionario ---
    for i, archivo in enumerate(archivos):
        ruta_completa = os.path.join(RUTA_ORIGEN, archivo)
        
        try:
            with Image.open(ruta_completa) as img:
                # Calculamos el hash
                img_hash = str(imagehash.phash(img))
                
                # Agregamos al diccionario
                if img_hash in hashes_dict:
                    hashes_dict[img_hash].append(archivo)
                else:
                    hashes_dict[img_hash] = [archivo]
                    
        except Exception as e:
            print(f"⚠️ Error leyendo {archivo}: {e}")

        # Barra de progreso simple
        if i % 100 == 0:
            print(f"   Analizando: {i}/{total_archivos}...", end="\r")

    print(f"   Análisis completado.                         \n")

    # --- NUEVA FASE: Exportar Diccionario a JSON ---
    print(f"💾 Guardando reporte de hashes en: {RUTA_JSON}")
    try:
        with open(RUTA_JSON, 'w', encoding='utf-8') as f:
            json.dump(hashes_dict, f, indent=4)
        print("✅ Archivo 'listados_hash.json' generado correctamente.\n")
    except Exception as e:
        print(f"❌ Error al guardar el JSON: {e}\n")

    # --- FASE 2: Copiado de Imágenes Únicas ---
    print("--- 2. COPIANDO IMÁGENES ÚNICAS A CARPETA NUEVA ---")
    
    # Crear carpeta destino si no existe
    if not os.path.exists(RUTA_DESTINO):
        try:
            os.makedirs(RUTA_DESTINO)
            print(f"✅ Carpeta creada: {RUTA_DESTINO}")
        except OSError as e:
            print(f"❌ Error crítico creando carpeta: {e}")
            return
    else:
        print(f"ℹ️  La carpeta destino ya existe. Se agregarán archivos allí.")

    total_unicas = len(hashes_dict)
    copiadas = 0
    
    # Iteramos sobre el diccionario
    # key = hash, value = lista de archivos [img1, img2, img3]
    for hash_val, lista_archivos in hashes_dict.items():
        
        # Tomamos SOLO el primer elemento de la lista (índice 0)
        imagen_elegida = lista_archivos[0]
        
        origen = os.path.join(RUTA_ORIGEN, imagen_elegida)
        destino = os.path.join(RUTA_DESTINO, imagen_elegida)
        
        try:
            # copy2 preserva metadatos (fechas de creación, etc.)
            shutil.copy2(origen, destino)
            copiadas += 1
        except Exception as e:
            print(f"❌ Error copiando {imagen_elegida}: {e}")
            
        if copiadas % 100 == 0:
            print(f"   Copiando: {copiadas}/{total_unicas}...", end="\r")

    print(f"   Copiado finalizado.                          ")

    # --- REPORTE FINAL ---
    print("-" * 40)
    print("RESUMEN DE LA OPERACIÓN")
    print("-" * 40)
    print(f"Imágenes originales procesadas: {total_archivos}")
    print(f"Imágenes únicas identificadas:  {total_unicas}")
    print(f"Imágenes descartadas (duplicados): {total_archivos - total_unicas}")
    print(f"📄 Reporte de hashes guardado en: {RUTA_JSON}")
    print(f"✅ Archivos copiados exitosamente a:\n{RUTA_DESTINO}")

if __name__ == "__main__":
    filtrar_y_copiar_unicos()