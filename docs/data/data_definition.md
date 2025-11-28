# Definición de los datos

## Origen de los datos

- [ ] Especificar la fuente de los datos y la forma en que se obtuvieron.

El dataset fue obtenido de la plataforma `Kaggle`, específicamente del repositorio público "**Pixel Art**" creado por el usuario Ebrahim Elgazar. El conjunto de datos contiene 89,000 imágenes de pixel art que representan varios personajes y objetos `Kaggle`.

Detalles técnicos de la fuente:

    * Plataforma: Kaggle (https://www.kaggle.com)
    * Identificador del dataset: ebrahimelgazar/pixel-art
    * URL de acceso: https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art/data
    * Autor/Propietario: Ebrahim Elgazar
    * Fecha de publicación: Febrero de 2024
    * Cantidad de imagenes: $89,000$
    * Tipo de contenido: Imágenes de pixel art en formato JPEG
    * Liciencia de uso: Apache 2.0, la cual permite procesar la información y sus derivados con total libertad.

La obtención de los datos para su posterior manejo se realizo utilizando el \script úbicado en `.\scripts\data_acquisition\main.py`, el cual permite obtenere las imagenes (carpeta `imagenes`), las eiquetas correspondientes a cada una de las imagenes (`labels.csv`), juntoa dos archivos `.npy`el primero con una representación vectorial de las imagenes y el segundo con la repesectación vectorial de las eiquetas de las imagenes, en este último caso la fuente no tiene una referencia al significado de cada una de las etiquetas, lo que en principio implica una dificultad para la el entendimiento de las categorias definidas por los archivos.

## Scripts para la carga de datos

- [ ] Especificar los scripts utilizados para la carga de los datos.

Como se mencionó en la sección anterior, se utilizó el script `main.py` ubicado en `.\scripts\data_acquisition` para automatizar la adquisición del dataset. Este script optimiza el proceso de carga de datos mediante:

    1. Descarga programática desde Kaggle vía `kagglehub`
    2. Gestión validada de rutas de destino
    3. Transferencia automática desde caché temporal a ubicación definitiva.

Esta implementación garantiza reproducibilidad, elimina intervención manual y prepara la estructura de archivos para las fases posteriores de análisis y modelado.

## Referencias a rutas o bases de datos origen y destino

- [ ] Especificar las rutas o bases de datos de origen y destino para los datos.

### Rutas de origen de datos

- [ ] Especificar la ubicación de los archivos de origen de los datos.
- [ ] Especificar la estructura de los archivos de origen de los datos.
- [ ] Describir los procedimientos de transformación y limpieza de los datos.

Los datos se obtuvieron del repositorio público de **Kaggle** identificado como `ebrahimelgazar/pixel-art`. La descarga se ejecutó mediante el script `main.py` ubicado en `.\scripts\data_acquisition`, el cual transfirió los archivos desde la caché de `kagglehub` hacia la ruta de destino configurada en el proyecto.

### Base de datos de destino

- [ ] Especificar la base de datos de destino para los datos.
- [ ] Especificar la estructura de la base de datos de destino.
- [ ] Describir los procedimientos de carga y transformación de los datos en la base de datos de destino.
