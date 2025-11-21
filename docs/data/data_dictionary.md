# Diccionario de datos

## Base de datos 

**Contamos con un dataset proveniente de Kaggle que contiene 89400 imágenes de distintos objetos, con un tamaño de 16x16 pixeles con 3 canales RGB  en formato .jpg que pertenecen a un juego online. Contamos con los siguientes archivos:

images: Carpeta con las imagenes.
labels.csv: Este archivo contiene las etiquetas de las imagenes.
sprites.npy: Este archivo tiene objetos de tipo numpy array con los sprites.
sprites_labels.npy: Este archivo contiene las etiquetas en formato numpy.

En la fuente de los datos, se observa que el archivo labels.csv cuenta con con las siguientes columnas

| Image Index | ID de cada imagen | int | 1-89400 |
| Image Path | Camino a cada imagen | str | - |
| Label | Tipo de la imagen, no hay documentacion sobre esto, hay que explorarlo | list | lista con cinco valores, cuatro con 0 y uno con 1 |
