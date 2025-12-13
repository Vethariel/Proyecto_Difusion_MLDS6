# Diccionario de datos

Este diccionario describe los artefactos de datos usados en el proyecto (raw e intermedio), su estructura, tipos, rangos y cómo se representan las etiquetas.

---

## 1. Artefactos `data/raw/` (datos originales)

### 1.1 `data/raw/images/images/` (imágenes en disco)

- **Tipo:** carpeta con archivos de imagen
- **Cantidad:** 89,400 archivos
- **Formato:** `.JPEG`
- **Convención de nombre:** `image_<idx>.JPEG`
  - `idx` empieza en **0** y llega hasta **89399**
- **Resolución:** 16×16 (RGB)

Uso en el proyecto: principalmente para inspección/validación; el entrenamiento se hace con arrays (`sprites.npy`) o con el dataset intermedio (`pixel_art_data.npz`).

### 1.2 `data/raw/labels.csv` (metadatos y etiquetas)

- **Tipo:** tabla CSV
- **Filas:** 89,400
- **Columnas:**

| Columna | Tipo | Ejemplo | Descripción |
|---|---|---|---|
| `Image Index` | `int` | `1` | Identificador 1-based (mín=1, máx=89400). |
| `Image Path` | `str` | `path/to/image_1.jpg` | Ruta referencial del dataset original (no necesariamente coincide con la ruta local final). |
| `Label` | `str` | `[1. 0. 0. 0. 0.]` | Etiqueta en formato vector one-hot serializado como texto. Hay **5 valores posibles** (5 clases). |

**Nota importante (mapeo index ↔ archivo):** los archivos en disco están numerados desde `image_0.JPEG`, mientras que `Image Index` empieza en 1. El mapeo típico es:

`Image Index = k  →  image_{k-1}.JPEG`

### 1.3 `data/raw/sprites.npy` (tensor de imágenes)

- **Tipo:** `numpy.ndarray`
- **Forma:** `(89400, 16, 16, 3)`
- **Dtype:** `uint8`
- **Rango esperado:** `[0, 255]`
- **Ejes:** `(N, H, W, C)` con `C=3` (RGB)

Uso en el proyecto: fuente principal “raw” para EDA rápido y para algunos pipelines de entrenamiento.

### 1.4 `data/raw/sprites_labels.npy` (etiquetas one-hot)

- **Tipo:** `numpy.ndarray`
- **Forma:** `(89400, 5)`
- **Dtype:** `float64`
- **Contenido:** one-hot estricto (cada fila suma 1)

Uso en el proyecto: conditioning por clase (5 clases) y validación de distribución por clase.

### 1.5 `data/raw/metadata.txt` (metadatos de ingesta)

- **Tipo:** texto plano
- **Contenido:** dataset origen, fecha/hora de descarga, herramienta de descarga, ruta de caché, nota de “datos crudos sin modificación”.

---

## 2. Artefactos `data/intermediate/` (dataset listo para modelado)

### 2.1 `data/intermediate/pixel_art_data.npz`

- **Tipo:** archivo `.npz` (contenedor NumPy)
- **Llaves:** `images`, `labels`, `labels_one_hot`, `metadata`

#### `images`

- **Forma:** `(89400, 16, 16, 3)`
- **Dtype:** `float32`
- **Rango:** `[0.0, 1.0]`
- **Descripción:** imágenes normalizadas para consumo directo por modelos.

#### `labels`

- **Forma:** `(89400,)`
- **Dtype:** `int32`
- **Rango:** `0..4`
- **Descripción:** etiqueta por imagen en formato entero (más cómodo para entrenamiento y evaluación).

#### `labels_one_hot`

- **Forma:** `(89400, 5)`
- **Dtype:** `float64`
- **Descripción:** versión one-hot (equivalente a `sprites_labels.npy`) para compatibilidad con pipelines que lo requieran.

#### `metadata`

- **Forma:** escalar (`shape=()`)
- **Dtype:** string Unicode
- **Descripción:** marca de tiempo de creación del `.npz` (por ejemplo: `Creado 2025-12-04 ... UTC`).

---

## 3. Definición de etiqueta (clases)

El proyecto trabaja con **5 clases**. Las etiquetas aparecen en tres formatos:

1. **Texto one-hot en `labels.csv`**: `Label` es un string con un vector one-hot.
2. **One-hot numérico en `sprites_labels.npy` y `labels_one_hot`**: array `(N, 5)` con una única posición en 1.
3. **Entero en `pixel_art_data.npz` (`labels`)**: array `(N,)` con valores `0..4`.

---

## 4. Notas de uso (implicaciones prácticas)

- Para **entrenamiento/evaluación**, se recomienda usar `data/intermediate/pixel_art_data.npz` por tener normalización y etiquetas enteras listas.
- Para **EDA rápido** o validaciones de integridad en raw, `sprites.npy`/`sprites_labels.npy` evitan I/O de miles de archivos.
