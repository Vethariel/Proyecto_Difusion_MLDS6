# El archivo .csv tiene las etiquetas de todas las imagenes, pero los nombres de los archivos respecto a los del .csv tiene un desface en la enumeración, ya que en el .csv se inicia con
# image_1.jpg y termina con image_89400.jpg; pero en los nombres de las imagenes se image_0.jpg y termina con image_89399.jpg. Por este motivo, se hace el cambio de los nombres dentro del
# archivo .csv, para que coincidan con los nombres de las imagenes tanto en los números como en el formato. De esta manera podemos garantizar la posibilidad de validar mas adelante, que
# cada una de las imagenes tenga una etiqueta.
#

import pandas as pd
import re

# Ruta del archivo original
ruta_original = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\labels.csv"

# Ruta del archivo nuevo
ruta_nuevo = r"D:\documentos\unal\diplomados\diplomado_ml_ds\mod6_metodologias_agiles_desarrollo_proyectos_ml\proyecto\datos_proyecto\new_labels.csv"

# Leer el archivo CSV
df = pd.read_csv(ruta_original)

# Función para decrementar el número en el path de la imagen y cambiar extensión a .JPEG
def decrementar_numero_imagen(path):
    # Buscar el patrón image_X.jpg o image_X.jpeg
    match = re.search(r'image_(\d+)\.(jpg|jpeg|JPEG|JPG)', path)
    if match:
        numero_actual = int(match.group(1))
        numero_nuevo = numero_actual - 1
        # Reemplazar el número y cambiar la extensión a .JPEG
        nuevo_path = re.sub(r'image_\d+\.(jpg|jpeg|JPEG|JPG)',
                            f'image_{numero_nuevo}.JPEG',
                            path)
        return nuevo_path
    return path

# Aplicar la función a la columna 'Image Path'
df['Image Path'] = df['Image Path'].apply(decrementar_numero_imagen)

# Guardar el nuevo archivo CSV
df.to_csv(ruta_nuevo, index=False)

print(f"Archivo creado exitosamente: {ruta_nuevo}")
print(f"\nPrimeras 5 filas del nuevo archivo:")
print(df.head())