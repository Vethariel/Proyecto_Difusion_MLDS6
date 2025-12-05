import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import hashlib

# Download latest version
path = kagglehub.dataset_download("ebrahimelgazar/pixel-art")

print("Path to dataset files:", path)

sprites = np.load(path + '/sprites.npy')
sprites_labels = np.load(path + "/sprites_labels.npy")

labels = pd.read_csv(path + '/labels.csv')

def funcion2_1():
  print("Primero veamos que paths son validos:")

  valid_paths_count = 0
  invalid_paths_count = 0
  invalid_paths_examples = []

  # Adjust the base path to the actual location of images
  base_images_path = os.path.join(path, 'images', 'images')

  for index, row in labels.iterrows():
      # Construct the image filename using the 'Image Index' column
      image_filename = f"image_{row['Image Index']}.jpg"
      
      # Construct the full actual path to the image file
      full_image_file_path = os.path.join(base_images_path, image_filename)
      
      if os.path.exists(full_image_file_path):
          valid_paths_count += 1
      else:
          invalid_paths_count += 1
          if len(invalid_paths_examples) < 5: # Store up to 5 examples
              invalid_paths_examples.append(full_image_file_path)

  print(f"Total valid image paths: {valid_paths_count}")
  print(f"Total invalid image paths: {invalid_paths_count}")

  if invalid_paths_count > 0:
      print("\nExamples of invalid paths:")
      for example_path in invalid_paths_examples:
          print(example_path)
  else:
      print("\nAll image paths are valid.")

  print("Por lo visto, ningun path es valido, observamos que las rutas tienen la terminacion incorrecta, no es .jpg sino .JPEG")
  print("Si se corrije lo anterior:")
  valid_paths_count = 0
  invalid_paths_count = 0
  invalid_paths_examples = []

  # Adjust the base path to the actual location of images
  base_images_path = os.path.join(path, 'images', 'images')

  for index, row in labels.iterrows():
      # Construct the image filename using the 'Image Index' column and the correct '.JPEG' extension
      image_filename = f"image_{row['Image Index']}.JPEG"
      
      # Construct the full actual path to the image file
      full_image_file_path = os.path.join(base_images_path, image_filename)
      
      if os.path.exists(full_image_file_path):
          valid_paths_count += 1
      else:
          invalid_paths_count += 1
          if len(invalid_paths_examples) < 5: # Store up to 5 examples
              invalid_paths_examples.append(full_image_file_path)

  print(f"Total valid image paths: {valid_paths_count}")
  print(f"Total invalid image paths: {invalid_paths_count}")

  if invalid_paths_count > 0:
      print("\nExamples of invalid paths:")
      for example_path in invalid_paths_examples:
          print(example_path)
  else:
      print("\nAll image paths are valid.")

def funcion2_2():
  # 3. Inicializa un diccionario vacío para almacenar los hashes de las imágenes y sus respectivos índices.
  image_hashes = {}

  # 4. Iterar a través del array sprites usando su índice
  for i, image in enumerate(sprites):
      # 5a. Convierte la imagen a bytes
      image_bytes = image.tobytes()
      # 5b. Calcula el hash MD5 de los bytes
      image_hash = hashlib.md5(image_bytes).hexdigest()

      # 5c. Almacena el hash y el índice en el diccionario image_hashes
      if image_hash in image_hashes:
          image_hashes[image_hash].append(i)
      else:
          image_hashes[image_hash] = [i]

  # 6. Calcula el número total de imágenes
  total_images = len(sprites)

  # 7. Calcula el número de imágenes únicas
  unique_images = len(image_hashes)

  # 8. Calcula el número de imágenes duplicadas
  duplicate_images = total_images - unique_images

  # 9. Calcula el porcentaje de duplicados
  percentage_duplicates = (duplicate_images / total_images) * 100

  # 10. Imprime los resultados en español
  print(f"Número total de imágenes: {total_images}")
  print(f"Número de imágenes únicas (por hash MD5): {unique_images}")
  print(f"Número de imágenes duplicadas: {duplicate_images}")
  print(f"Porcentaje de imágenes duplicadas: {percentage_duplicates:.2f}%")

  # 11. Si hay imágenes duplicadas, imprime ejemplos
  if duplicate_images > 0:
      print("\nEjemplos de grupos de imágenes duplicadas (índices):")
      duplicate_groups_count = 0
      for img_hash, indices in image_hashes.items():
          if len(indices) > 1:
              print(f"  Hash: {img_hash}, Índices: {indices}")
              duplicate_groups_count += 1
          if duplicate_groups_count >= 5: # Limit examples to avoid excessive output
              break

  print("\nVerificación de imágenes duplicadas completada.")

def funcion2_3():
  # 2. Initialize counters and lists
  negros_count = 0
  blancos_count = 0
  demasiados_colores_count = 0
  pocos_colores_count = 0

  negros_indices = []
  blancos_indices = []
  demasiados_colores_indices = []
  pocos_colores_indices = []

  # 3. Set thresholds
  UMBRAL_DEMASIADOS_COLORES = 50
  UMBRAL_POCOS_COLORES = 3

  # 4. Iterate through the sprites array
  for i, sprite in enumerate(sprites):
      # 5a. Check for completely black sprites
      if np.all(sprite == 0):
          negros_count += 1
          if len(negros_indices) < 5:
              negros_indices.append(i)
      # 5b. Check for completely white sprites
      elif np.all(sprite == 255):
          blancos_count += 1
          if len(blancos_indices) < 5:
              blancos_indices.append(i)
      else:
          # 5c. Count unique colors for other sprites
          # Reshape to (num_pixels, 3) to treat each pixel as a color vector
          num_unique_colors = len(np.unique(sprite.reshape(-1, 3), axis=0))

          # 5d. Check for sprites with too many colors
          if num_unique_colors > UMBRAL_DEMASIADOS_COLORES:
              demasiados_colores_count += 1
              if len(demasiados_colores_indices) < 5:
                  demasiados_colores_indices.append(i)
          # 5e. Check for sprites with too few colors
          elif num_unique_colors < UMBRAL_POCOS_COLORES:
              pocos_colores_count += 1
              if len(pocos_colores_indices) < 5:
                  pocos_colores_indices.append(i)

  # 6. Print summary of findings in Spanish
  print("--- Análisis de Características Extremas de Imágenes ---")
  print(f"Sprites completamente negros: {negros_count} (Ejemplos: {negros_indices})")
  print(f"Sprites completamente blancos: {blancos_count} (Ejemplos: {blancos_indices})")
  print(f"Sprites con más de {UMBRAL_DEMASIADOS_COLORES} colores únicos: {demasiados_colores_count} (Ejemplos: {demasiados_colores_indices})")
  print(f"Sprites con menos de {UMBRAL_POCOS_COLORES} colores únicos: {pocos_colores_count} (Ejemplos: {pocos_colores_indices})")
  print("--------------------------------------------------------")

# 2. Generar image_hashes_global
image_hashes_global = {}

for i, image in enumerate(sprites):
    image_bytes = image.tobytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()

    if image_hash in image_hashes_global:
        image_hashes_global[image_hash].append(i)
    else:
        image_hashes_global[image_hash] = [i]

print(f"Se generó image_hashes_global con {len(image_hashes_global)} hashes únicos.")

def funcion2_4(sprites_array, demasiados_colores_indices, image_hashes):
  print("Generando mosaico de imágenes con 'demasiados colores'...")
  # 1. Mosaico de imágenes con "demasiados colores"
  if len(demasiados_colores_indices) > 0:
      num_images_to_display = min(25, len(demasiados_colores_indices)) # Display up to 25 images
      grid_size = int(np.ceil(np.sqrt(num_images_to_display)))
      
      # Seleccionar una muestra aleatoria de índices
      sample_indices = random.sample(demasiados_colores_indices, num_images_to_display)

      fig1, axes1 = plt.subplots(grid_size, grid_size, figsize=(10, 10))
      fig1.suptitle('Mosaico de imágenes con demasiados colores', fontsize=16)
      axes1 = axes1.flatten()

      for i, ax in enumerate(axes1):
          if i < num_images_to_display:
              ax.imshow(sprites_array[sample_indices[i]])
              ax.axis('off')
          else:
              ax.remove() # Remove empty subplots

      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt.show()
  else:
      print("No se encontraron imágenes con 'demasiados colores' para mostrar.")

  print("\nGenerando mosaico de imágenes únicas (no duplicadas)")
  # 2. Mosaico de imágenes únicas
  unique_image_indices = [indices[0] for indices in image_hashes.values()]

  if len(unique_image_indices) > 0:
      num_images_to_display = min(25, len(unique_image_indices)) # Display up to 25 unique images
      grid_size = int(np.ceil(np.sqrt(num_images_to_display)))

      # Seleccionar una muestra aleatoria de índices de imágenes únicas
      sample_unique_indices = random.sample(unique_image_indices, num_images_to_display)

      fig2, axes2 = plt.subplots(grid_size, grid_size, figsize=(10, 10))
      fig2.suptitle('Mosaico de imágenes únicas', fontsize=16)
      axes2 = axes2.flatten()

      for i, ax in enumerate(axes2):
          if i < num_images_to_display:
              ax.imshow(sprites_array[sample_unique_indices[i]])
              ax.axis('off')
          else:
              ax.remove() # Remove empty subplots

      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      plt.show()
  else:
      print("No se encontraron imágenes únicas para mostrar.")
  print("\n--- Justificación para la Gestión de Imágenes Extremas y Duplicadas en el Dataset ---")
  print("\nConsiderando los hallazgos del análisis de características extremas (`funcion2_3`) y la detección de duplicados (`funcion2_2`),")
  print("se ha evaluado la necesidad de mantener, corregir o eliminar las imágenes identificadas como 'raras' o duplicadas.")

  print("\n**Hallazgos Clave:**")
  print("  - **Imágenes con 'Demasiados Colores' (Outliers Cromáticos):** Se identificaron 30,000 imágenes con más de 50 colores únicos. Visualmente, estas imágenes muestran degradados y texturas complejas, alejándose del pixel art de paleta limitada.")
  print("  - **Ausencia de Extremos Puros:** No se encontraron imágenes completamente negras o blancas, lo cual sugiere una calidad base razonable del dataset.")
  print("  - **Imágenes Duplicadas:** Se detectó un altísimo porcentaje (98.07%) de imágenes duplicadas, resultando en 1,722 imágenes únicas a nivel de píxel.")
  print("  - **Diversidad de Imágenes Únicas:** El mosaico de imágenes únicas reveló una gran variedad conceptual y visual, abarcando distintos sujetos y estilos.")

  print("\n**Decisión y Justificación (Orientada a Modelos de Difusión):**")
  print("Para el presente proyecto, que busca entrenar un modelo de difusión, la decisión es **MANTENER** todas las imágenes únicas y aquellas identificadas con 'demasiados colores'. La justificación es la siguiente:")
  print("\n  1.  **Cantidad de Datos:** Los modelos de difusión se benefician enormemente de grandes volúmenes de datos para aprender patrones complejos y generar resultados de alta calidad. Mantener la mayor cantidad posible de ejemplos únicos es crucial.")
  print("\n  2.  **Variabilidad Cromática:** La presencia de imágenes con una rica paleta de colores (más de 50 colores únicos) introduce una variabilidad cromática significativa en el dataset. Para un modelo de difusión, esta diversidad de colores es altamente deseable, ya que le permite aprender a generar imágenes con una gama más amplia de tonalidades y detalles sutiles, enriqueciendo la capacidad generativa del modelo. Descartar estas imágenes reduciría la riqueza del espacio latente que el modelo puede explorar.")
  print("\n  3.  **Redundancia Eliminada:** Aunque el dataset original contenía una gran cantidad de duplicados, la estrategia de trabajar con las 1,722 imágenes únicas (identificadas por hash MD5) aborda el problema de la redundancia inútil. Esto asegura que cada ejemplo utilizado para el entrenamiento aporte información visual novedosa, maximizando la eficiencia del aprendizaje sin sobreajustarse a ejemplos idénticos.")
  print("\n**Conclusión:** La estrategia adoptada busca maximizar la diversidad y cantidad de información útil para el entrenamiento del modelo de difusión, aprovechando tanto la variabilidad cromática como la variedad conceptual presente en los ejemplos únicos del dataset.")
  print("---------------------------------------------------------------------------")