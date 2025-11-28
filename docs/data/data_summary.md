# Reporte de Datos

Header

## 1. Resumen general de los datos

### ✔️ 1.1 Número total de observaciones
- Contar **89.000 imágenes**.
- Contar **20 categorías**.
- Verificar consistencia entre `labels.csv`, `sprites.npy` e imágenes en carpeta.

### ✔️ 1.2 Variables presentes
Para imágenes, estas “variables” son:
- Matriz de píxeles: **16×16×3**
- Etiqueta (**entero 0–19**)
- Ruta del archivo (`path`)

### ✔️ 1.3 Tipos de variables
- **Variables numéricas:** valores RGB (0–255)
- **Variable categórica:** etiqueta
- **Variable de texto:** path

### ✔️ 1.4 Verificación de faltantes
- Confirmar que no falten rutas en `labels.csv`.
- Revisar que no haya imágenes dañadas o corruptas.
- Validar que `sprites_labels.npy` coincide en dimensión con `sprites.npy`.

### ✔️ 1.5 Distribución general de las imágenes
- Mosaico aleatorio de **100–300 imágenes**.
- Verificar tamaño uniforme (16×16).
- Verificar distribución de categorías (≈4.470 por clase).

## 2. Resumen de calidad de los datos

### ✔️ 2.1 Presencia de valores faltantes
Reporte exacto de:
- Imágenes sin entrada en CSV.
- Rutas inválidas.
- Errores de lectura (PIL, OpenCV).

### ✔️ 2.2 Duplicados
- Detectar duplicados por hash (MD5 o perceptual hash).
- Reportar porcentaje de duplicados encontrados.

### ✔️ 2.3 Valores extremos o inconsistencias
Ejemplos de casos a detectar:
- Sprites completamente negros.
- Sprites completamente blancos.
- Sprites con ruido aleatorio.
- Sprites con demasiados colores (>50 únicos).
- Sprites con muy pocos colores (<3 únicos).

### ✔️ 2.4 Outliers visuales
- Mostrar ejemplos “raros”.
- Justificar si se deben:
- Mantener
- Corregir
- Eliminar

### ✔️ 2.5 Acciones tomadas
- Redimensionamiento uniforme.
- Conversión a `float32`.
- Normalización (0–1 o –1 a 1).
- Eliminación de imágenes corruptas (si aplica).
- Eliminación de duplicados (si existen).

## 3. Variable objetivo

*(Adaptación del concepto para modelos de difusión)*
En modelos generativos **la variable objetivo es la propia imagen**:
> x ~ p_data

### ✔️ 3.1 Explicar por qué la variable objetivo es la imagen
- No clasificamos.
- No predecimos.
- Buscamos **modelar la distribución completa de los datos**.

### ✔️ 3.2 Distribución global de las imágenes
- Histogramas promedio de colores.
- Distribución de intensidades por canal RGB.
- Visualización del “promedio” por clase.

### ✔️ 3.3 Variabilidad intra-clase
- Mosaicos 5×5 por clase.
- Comparación visual de ejemplos dentro de una misma categoría.

### ✔️ 3.4 Variabilidad global
- **PCA** sobre vectores flattenizados (16×16×3 → 768 componentes).
- **t-SNE** para clusters naturales.

## 4. Variables individuales

*(Adaptado a análisis de imágenes)*

### ✔️ 4.1 Canales RGB como variables
- Histograma por canal.
- Distribución de valores RGB por clase.
- Estadísticas descriptivas por canal.

### ✔️ 4.2 Número de colores por imagen
- Conteo de colores únicos.
- Relación entre número de colores y etiqueta.
- Clasificación por:
- Low palette
- Mid palette
- High palette

### ✔️ 4.3 Estructura espacial
- Verificar centrado del sprite.
- Análisis de espacio vacío vs contenido.
- Heatmap de densidad de píxeles por clase.

### ✔️ 4.4 Posibles transformaciones
- Normalización.
- Estandarización de paleta.
- Pixel-shuffle (opcional).
- Augmentations razonables:
- Flip horizontal
- Shift pequeño
- Rotación mínima (<15°)
- Jitter de color

### ✔️ 4.5 Relación con etiqueta
- Mapas de calor por clase.
- Imagen promedio por clase.
- Modos de color por clase.

## 5. Ranking de variables

*(Justificación metodológica adaptada)*

### ✔️ 5.1 PCA
- Variancia explicada por los primeros 10 componentes.
- Visualización de reconstrucciones PCA (k = 10, 20, 30).

### ✔️ 5.2 Importancia del color
- Determinar qué canales aportan mayor variabilidad.
- Comparaciones entre clases.

### ✔️ 5.3 Separabilidad entre clases
- t-SNE coloreado por etiqueta.
- Silhouette score (opcional).
- Clustering k-means sobre embeddings.

Esto permite analizar:
- Viabilidad de un modelo **class-conditional**.
- Complejidad necesaria del UNet.
- Necesidad de conditioning adicional.

## 6. Relación entre variables explicativas y variable objetivo


- Ratio de píxeles vacíos vs clase.

### ✔️ 6.3 Mini-modelo auxiliar
Entrenar un clasificador CNN pequeño para evaluar:
- Si las clases son distinguibles.
- Qué tan fuerte es el signal visual.
- Si es viable un **modelo condicionado por clase**.

Esto refuerza decisiones en la fase 3 del modelamiento.
