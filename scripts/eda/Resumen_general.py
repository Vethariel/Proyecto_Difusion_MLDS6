import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Download latest version
path = kagglehub.dataset_download("ebrahimelgazar/pixel-art")

print("Path to dataset files:", path)

sprites = np.load(path + '/sprites.npy')
sprites_labels = np.load(path + "/sprites_labels.npy")

labels = pd.read_csv(path + '/labels.csv')

def funcion1_1():
  print("Tamaño de sprites:")
  print(sprites.shape)
  print("Tamaño de sprites_labels:")
  print(sprites_labels.shape)
  print("Tamaño de labels:")
  print(labels.shape)
  print("Numero de imagenes en images")
  print(len(os.listdir(path+"/images"+"/images")))
  print("Los archivos son consistentes entre si, todos cuentan con 89400 ejemplos")

def funcion1_2():
  print("El tamaño de las imagenes es de 16x16x3:")
  print(sprites.shape)
  print()
  unique_labels = np.unique(sprites_labels, axis=0)

  print(f"Valores unicos en 'sprites_labels':\n{unique_labels}")
  print(f"Numero de etiquetas unicas: {len(unique_labels)}\n")

  print(f"La ruta de los archivos es:\n{path}")

def funcion1_3():
  sprites_dtype = sprites.dtype
  print(f"Tipo de dato del array 'sprites': {sprites_dtype}")

  min_pixel_value = np.min(sprites)
  max_pixel_value = np.max(sprites)

  print(f"Valor minimo de pixel en 'sprites': {min_pixel_value}")
  print(f"Valor maximo de pixel en 'sprites': {max_pixel_value}")
  print()

  sprites_labels_dtype = sprites_labels.dtype
  print(f"Tipo de dato del array 'sprites_labels': {sprites_labels_dtype}")

def funcion1_4():
  print("Informacion sobre el csv labels:")
  print(labels.info())
  print("No hay datos faltantes\n")
  print("Anteriormente vimos que no hay imagenes\n con valores de RGB imposibles, de modo que en principio no hay imagenes corruptas")
  
def funcion1_5():
  print("A continuacion, hacemos un mosaico con 300 imagenes aleatorias para verificar si hay imagenes dañadas")
  num_images_to_display = 300 # You can change this (e.g., 200, 300)
  grid_size = int(np.sqrt(num_images_to_display)) # For a square grid

  # Ensure we don't try to display more images than available
  if num_images_to_display > len(sprites):
      num_images_to_display = len(sprites)
      grid_size = int(np.sqrt(num_images_to_display))

  # Randomly select indices of images to display
  random_indices = random.sample(range(len(sprites)), num_images_to_display)

  # Create a figure and a grid of subplots
  fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10)) # Adjust figsize as needed
  fig.suptitle(f'Random Mosaic of {num_images_to_display} Images', fontsize=16)

  # Flatten the axes array for easy iteration
  axes = axes.flatten()

  # Iterate through selected images and plot them
  for i, ax in enumerate(axes):
      if i < num_images_to_display:
          ax.imshow(sprites[random_indices[i]])
          ax.axis('off') # Hide axes for cleaner display
      else:
          ax.remove() # Remove empty subplots if num_images_to_display is not a perfect square

  plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
  plt.show()