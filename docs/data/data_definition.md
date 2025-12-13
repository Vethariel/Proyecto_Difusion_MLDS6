# Definición de los datos

Este documento describe el origen del dataset, cómo se adquiere en el repositorio y cómo se transforma al formato intermedio usado por EDA, entrenamiento y evaluación.

---

## 1. Origen de los datos

El dataset se obtiene desde Kaggle: **`ebrahimelgazar/pixel-art`** (Pixel Art), publicado por **Ebrahim Elgazar**.

- Plataforma: Kaggle
- Dataset ID: `ebrahimelgazar/pixel-art`
- URL: `https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art/data`
- Tipo de contenido: imágenes pixel art + artefactos NumPy (`.npy`) + etiquetas en CSV
- Licencia: Apache 2.0 (según el dataset en Kaggle)

En este repositorio trabajamos con **89,400** imágenes de **16×16×3** (RGB) y **5 clases** (etiquetas one-hot en raw; enteros `0..4` en el dataset intermedio).

---

## 2. Estructura de datos (artefactos)

### 2.1 Datos crudos (`data/raw/`)

Después de la descarga, el repositorio espera los siguientes artefactos en `data/raw/`:

- `data/raw/images/images/` (carpeta con `image_<idx>.JPEG`)
- `data/raw/labels.csv` (metadatos + etiqueta one-hot serializada como texto)
- `data/raw/sprites.npy` (array con forma `(89400, 16, 16, 3)`, `uint8`)
- `data/raw/sprites_labels.npy` (one-hot con forma `(89400, 5)`)
- `data/raw/metadata.txt` (bitácora de ingesta)

### 2.2 Datos intermedios (`data/intermediate/`)

El formato de trabajo recomendado para modelado es:

- `data/intermediate/pixel_art_data.npz`
  - `images`: `(89400, 16, 16, 3)` `float32`, normalizado a `[0, 1]`
  - `labels`: `(89400,)` `int32`, valores `0..4`
  - `labels_one_hot`: `(89400, 5)` `float64`
  - `metadata`: string con timestamp de creación

---

## 3. Scripts de adquisición (raw)

### 3.1 Script estándar: `scripts/data_acquisition/download_raw_data.py`

Este es el script documentado como estándar para adquirir el dataset en este repositorio. Implementa un flujo de MLOps simple:

1. Descarga el dataset desde Kaggle usando `kagglehub`.
2. Copia el contenido descargado al directorio del repositorio `data/raw/`.
3. Registra un `metadata.txt` con fecha de descarga y ruta de caché.

Ejecución:

```bash
.\env\Scripts\python.exe scripts/data_acquisition/download_raw_data.py
```

Requisitos:

- Credenciales de Kaggle configuradas en el equipo (archivo `kaggle.json`).
- Acceso a red (descarga inicial).

### 3.2 Script legado: `scripts/data_acquisition/main.py`

Existe un script anterior (`main.py`) que también descarga con `kagglehub`, pero solicita una ruta destino por consola y no está alineado con la estructura del repositorio (`data/raw/`). Se conserva por trazabilidad, pero el flujo recomendado es `download_raw_data.py`.

---

## 4. Nota sobre índices y nombres de archivos

Hay una diferencia típica entre:

- `labels.csv` → `Image Index` empieza en **1**
- nombres en disco → `image_<idx>.JPEG` empieza en **0**

Mapeo recomendado:

`Image Index = k  →  image_{k-1}.JPEG`

Adicionalmente, en el dataset original el `Image Path` del CSV puede venir con extensiones `.jpg/.jpeg`, mientras que en el repositorio las imágenes quedan como `.JPEG`.

El script `scripts/data_acquisition/correccion_nombres_imagenes.py` documenta la motivación de esta corrección (desfase + extensión). En la práctica, para validaciones locales conviene usar el mapeo anterior o generar una versión “corregida” del CSV (sin perder el `Image Path` original como referencia).

---

## 5. Transformación a dataset intermedio

Una vez descargados los artefactos raw, se genera el dataset intermedio con:

- `scripts/preprocessing/raw_to_npz.py`

Este script:

1. Carga `data/raw/sprites.npy` y `data/raw/sprites_labels.npy`.
2. Normaliza las imágenes a `float32` en `[0, 1]`.
3. Convierte one-hot a etiqueta entera (`argmax`) en `labels` (`int32`).
4. Guarda `data/intermediate/pixel_art_data.npz` (comprimido).

Ejecución:

```bash
.\env\Scripts\python.exe scripts/preprocessing/raw_to_npz.py
```

Implicación: `pixel_art_data.npz` es el artefacto recomendado para EDA, entrenamiento y evaluación por ser compacto, consistente y directamente consumible por modelos.

---

## 6. Reproducibilidad y trazabilidad

- `data/raw/metadata.txt` conserva la fecha de ingesta y la ruta de caché desde la que se copió el dataset.
- `data/intermediate/pixel_art_data.npz` incluye un campo `metadata` con timestamp de creación.
- Los módulos EDA (`scripts/eda/`) registran métricas y rutas de figuras en `reports/eda/eda.json`.
