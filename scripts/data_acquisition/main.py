import kagglehub
import shutil
import os

# 1. Descargar el dataset (Esto baja todas las imágenes a la caché de Kagglehub)
# Devuelve la ruta donde se descargaron los archivos
print("Descargando dataset desde Kaggle...")
path_descarga = kagglehub.dataset_download("ebrahimelgazar/pixel-art")

print("Dataset descargado en la caché:", path_descarga)

# 2. Solicitar la ruta de destino al usuario
# .strip() elimina espacios accidentales al inicio o final
# EL mensaje aparecera en la terminal
print("\n--- Configuración de Destino ---")
path_input = input("Ingresa la ruta de la carpeta donde quieres guardar las imágenes: ").strip()

# Convertimos el input a una ruta absoluta del sistema para evitar errores
destination_path = os.path.abspath(path_input)
print(f"Los archivos se guardarán en: {destination_path}")

# Verificar si la carpeta de destino existe, si no, crearla
if not os.path.exists(destination_path):
    try:
        os.makedirs(destination_path)
        print(f"Carpeta creada: {destination_path}")
    except OSError as e:
        print(f"Error crítico al crear la carpeta: {e}")
        exit() # Detiene el script si no puede crear la carpeta

# Copiar el contenido (puedes usar move o copytree según prefieras)
try:
    print("Copiando archivos...")
    shutil.copytree(path_descarga, destination_path, dirs_exist_ok=True)
    print(f"¡Éxito! Archivos copiados a: {destination_path}")
except Exception as e:
    print(f"Error al copiar archivos: {e}")

# 3. Listar archivos para verificar
if os.path.exists(destination_path):
    print("Archivos en tu carpeta de proyecto:", os.listdir(destination_path)[:5])