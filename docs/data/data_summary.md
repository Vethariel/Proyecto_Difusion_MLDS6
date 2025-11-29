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

<!-- *(Adaptación del concepto para modelos de difusión)*
En modelos generativos **la variable objetivo es la propia imagen**:
> x ~ p_data -->
Los script que a los que se hace referencia en cada uno de los item analizados se encuentran dentro de `.\scripts\eda`

### ✔️ 3.1 Explicar por qué la variable objetivo es la imagen
<!-- - No clasificamos.
- No predecimos.
- Buscamos **modelar la distribución completa de los datos**. -->

Esta parte se analizo con ayuda del script `3_1_variable_objetivo.py`.

Anteriormente se observo la posibilidad de que cada una de las imagenes puede identificarse con un hash el cual se puede utilizar para determinar que imagenes se parecen entre si, con lo que se pudieron determinar conjuntos de imagenes parecidas entre si, por lo que queremos ver como es el comportamiento de la distribución de intensidad tanto del conjunto original de imagenes como el de imagenes únicas. Al final de la ejecución podemos observar el histograma de la distribución de cada conjunto, junto a una imagen muy probable dentro del conjunto y otra de ruido poco probable dentro del mismo.

Los graficos de densidades son muy similares entre si, siendo los valores de intensidad cercanos a cero los que acumulan la mayor parte de la densidad, dandonos a entender que en todas las imagenes el color negro o cercanos a este predomina sobre los demás. Además, la simiitud entre los histogramas implica que apezar de quedarse únicamente con las imagenes diferentes, la estructura probabilistica de la intensidad se mantiene entre los conjuntos. Esto nos permite mejorar los tiempos de procesamiento trabajando con el conjunto de imagenes únicas, habiendo pasado de $84.000$ a $1.665$.

### ✔️ 3.2 Distribución global de las imágenes
<!-- - Histogramas promedio de colores.
- Distribución de intensidades por canal RGB.
- Visualización del “promedio” por clase. -->

Como resultado de la ejecución del script `3_2_distri_imagenes.py`, podemos concluir lo siguiente.

Los histogramas muestran que al utilizar las imagenes únicas, hay una disminución en la densidad de la intensidad representada por el negro, lo que es coherente con los resultados obtenidos en el caso anteior. También podemos ver que las imagenes promedio son similares para ambos conjuntos, ya que la disminución se da en una zona de las imagenes donde domina el negro.

### ✔️ 3.3 Variabilidad intra-clase
<!-- - Mosaicos 5×5 por clase.
- Comparación visual de ejemplos dentro de una misma categoría. -->
Como resultado de la ejecución del script `3_3_variable_interclase.py`, podemos concluir lo siguiente.

Con el fin de comprender  las caracteristicas visuales de algunas de las categorias se compara el contenidos de dos mosaicos de $5\times5$ para dos categorias ecogidas al azar. La falta de homogeneidad de los objetos observados dentro de cada uno de los mosaicos, dejan ver que hay algo de `lable noise` dentro de todas las categorias, aumentando la variabilidad intra clases en todas las categorias y exigiendo a futuro mayor capacidad por parte de los modelos que se vayan a implementar.

### ✔️ 3.4 Variabilidad global
<!-- - **PCA** sobre vectores flattenizados (16×16×3 → 768 componentes).
- **t-SNE** para clusters naturales. -->

Como resultado de la ejecución del script `3_4_variabilidad_global.py`, podemos concluir lo siguiente.

PCA nos permitirá descomponer las imágenes (vectores de 768 dimensiones) y entender qué dimensiones (o combinaciones de píxeles) explican la mayor parte de la variación en tu dataset de pixel art. Esto es crucial para la eficiencia, ya que, si el 99% de la varianza se explica con solo 50 componentes, podemos reducir drásticamente la dimensionalidad para ciertos entrenamientos o análisis posteriores. Por esta razón utilizaremos el criterio del umbral, identificando la cantidad de componentes en donde se tiene el 80% y el 90% de la varianza explicada, la ejecución del script muestra la cantidad de componentes necesarias para cada porcentaje escogido.

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

### 5.1 — Análisis PCA del Dataset Pixel Art

Este análisis utiliza PCA para evaluar cuánta información visual se puede comprimir sin pérdida significativa en las imágenes del dataset (89400 sprites de 16×16×3). El objetivo es entender la estructura latente del dataset, la redundancia cromática y la complejidad real de sus variaciones espaciales.

---

#### Variancia explicada

La matriz aplanada resultante tiene dimensión 768 por imagen. Al aplicar PCA, observamos cómo la varianza total se acumula a medida que se agregan componentes principales:

- **PC1:** 12.13 % de la variabilidad total  
- **Primeras 10 componentes:** 46.88 %  
- **Primeras 20 componentes:** 59.59 %  
- **Primeras 30 componentes:** ~70 %  

La siguiente figura muestra la curva completa de varianza acumulada:

![Variance curve](../../reports/figures/eda/pca/variance_ratio.png)

La curva crece con fuerza en los primeros componentes y luego se aplana, lo que indica que gran parte de la información está concentrada en pocas direcciones latentes. Esto revela un dataset altamente estructurado, con patrones visuales consistentes y poca variabilidad caótica.

---

#### Reconstrucciones por número de componentes

Para evaluar el poder reconstructivo del PCA, se reconstruyó una imagen del dataset utilizando diferentes cantidades de componentes. Esto permite visualizar cuánta información se pierde al reducir dimensionalidad.

##### Reconstrucción con 10 componentes

El resultado conserva la forma general pero pierde detalle fino. Los colores se agrupan en bloques y la silueta es apenas perceptible:

![k10](../../reports/figures/eda/pca/reconstruction_k10.png)

##### Reconstrucción con 20 componentes

La forma, proporciones y colores principales se restauran con mayor precisión. Se observan contornos más claros y sombras más coherentes:

![k20](../../reports/figures/eda/pca/reconstruction_k20.png)

##### Reconstrucción con 30 componentes

A partir de 30 componentes, la reconstrucción es visualmente estable y muy cercana al original. El nivel de detalle recuperado es suficiente para preservar la identidad del sprite:

![k30](../../reports/figures/eda/pca/reconstruction_k30.png)

---

#### Interpretación técnica

El comportamiento del PCA revela varias características clave del dataset:

- **Alta compresibilidad:** Un subconjunto pequeño de componentes explica más del 50 % de la variación visual.
- **Estructura visual consistente:** La similitud entre sprites (formas redondeadas, paletas suaves, simetría) reduce la necesidad de dimensiones adicionales.
- **Información dominada por patrones globales:** Los cambios importantes provienen de grandes bloques de color y no de texturas locales complejas.
- **Latent space compacto:** Para modelos generativos posteriores (CNN, autoencoders, diffusion) basta un espacio latente de baja dimensionalidad; no es necesario trabajar directamente con los 768 píxeles originales.

---

#### Conclusión

El análisis PCA demuestra que el dataset de pixel art es altamente estructurado y presenta redundancia visual significativa. Con solo 20–30 componentes ya es posible reconstruir imágenes con fidelidad considerable. Esto indica que:

1. Los modelos de aprendizaje pueden entrenar rápidamente sobre este dominio.  
2. Es viable trabajar con representaciones latentes comprimidas.  
3. La estructura visual es lo suficientemente coherente como para permitir modelos condicionados por clase.

Este punto del análisis confirma que el dataset es ideal para métodos generativos basados en representaciones compactas y controlables.


### 5.2 — Importancia del Color en el Dataset Pixel Art

Este análisis evalúa cómo los canales de color (R, G, B) contribuyen a la variabilidad del dataset de pixel art. Se analiza su varianza global, su comportamiento por clase, su colorfulness perceptual y la cantidad de información que retiene cada uno mediante PCA. Este estudio es fundamental para comprender la estructura estilística del dataset y para orientar el diseño de modelos generativos condicionados por color.

---

#### Varianza global por canal

El análisis de varianza global muestra cuánta variabilidad aporta cada canal a través de todo el dataset. Los valores obtenidos son:

- **R:** 0.1303  
- **G:** 0.1267  
- **B:** 0.1772  

El canal **B** emerge como el más variable y, por tanto, el más informativo. Esto sugiere que la mayor parte del contraste y cambio visual se encuentra en la dimensión azul del espacio RGB, probablemente debido al uso intensivo de tonos púrpuras, rosados y sombreados fríos característicos del dataset.

![Varianza global](../../reports/figures/eda/color/global_variance.png)

---

#### Varianza por clase

Al segmentar por clase, la variabilidad adquiere mayor significado:

- **Clase 3** presenta la mayor variación en los tres canales.  
- **Clase 2** es la más homogénea, lo que indica paletas más restringidas.  
- En todas las clases, el canal **B sigue siendo dominante**, confirmando su rol estructural en el estilo visual.

Esto respalda la hipótesis de que cada clase agrupa sprites provenientes de **diferentes fuentes o estilos artísticos**.

![Varianza por clase](../../reports/figures/eda/color/variance_by_class.png)

---

#### Colorfulness por clase

La métrica de **Hasler & Süsstrunk** aproxima la percepción humana del color basado en contrastes RG y YB. Los promedios obtenidos:

- **Clase 1:** 0.3807 (la más saturada)  
- **Clase 2:** 0.3051  
- **Clases 0, 3, 4:** entre 0.22 y 0.23  

La Clase 1 destaca como el estilo más vibrante, mientras que las demás se mantienen más neutras o uniformes en saturación.

![Colorfulness por clase](../../reports/figures/eda/color/colorfulness_by_class.png)

---

#### Importancia de los canales mediante PCA

Se aplica PCA por canal para medir cuánta varianza captura el primer componente principal (PC1) de cada uno:

- **R:** 0.1741  
- **G:** 0.1564  
- **B:** 0.1848  

Nuevamente, el canal **B** es el que más información concentra, lo que coincide con todos los análisis anteriores.

![PCA por canal](../../reports/figures/eda/color/pca_by_channel.png)

---

#### Ranking integrado de importancia cromática

Combinando:

- Varianza global  
- Varianza explicada por PCA  

el puntaje final queda:

- **B:** 0.3620  
- **R:** 0.3044  
- **G:** 0.2831  

El orden es:

**B > R > G**

Esto confirma que el azul es el eje cromático dominante del dataset.

---

#### Conclusiones

El análisis del color revela:

1. El **canal azul (B)** es el que mayor información aporta en todos los niveles evaluados.  
2. Las clases muestran firmas cromáticas diferentes, lo que apunta a **diferencias estilísticas entre las fuentes del pixel art**.  
3. La Clase 1 es la más saturada y visualmente intensa; la Clase 3 es la más variable; la Clase 2 es la más uniforme.  
4. El color es un atributo altamente discriminativo en el dataset, lo que será clave para modelos de clasificación, generación y condicionamiento.

La importancia estructural del color, especialmente del canal azul, sugiere que los modelos generativos pueden beneficiarse de arquitecturas que traten explícitamente la información cromática —ya sea mediante embeddings condicionados, espacios latentes separados o módulos para manejo de estilo.

---



### 5.3 — Separabilidad entre clases

La separabilidad entre clases en un dataset visual como este determina qué tan “objetiva” es la etiqueta para un modelo. Aunque cada sprite tiene una resolución mínima (16×16×3), sus variaciones cromáticas, posicionales y temáticas pueden generar un espacio continuo más que uno discreto. Esta sección evalúa ese fenómeno desde la estructura visual, estadísticas de color, embeddings reducidos y métodos no supervisados.

---

#### **Visualización directa por clase (mosaicos)**

Los mosaicos permiten observar la coherencia temática interna de cada etiqueta. Las clases humanoides mantienen proporciones y poses similares; las criaturas exhiben variaciones de color vibrante; frutas y vegetales presentan patrones redondeados; los ítems se distinguen por contornos geométricos y simetrías.

![Class 0](../../reports/figures/eda/class_separability/label_grid_class0.png)
![Class 1](../../reports/figures/eda/class_separability/label_grid_class1.png)
![Class 2](../../reports/figures/eda/class_separability/label_grid_class2.png)
![Class 3](../../reports/figures/eda/class_separability/label_grid_class3.png)
![Class 4](../../reports/figures/eda/class_separability/label_grid_class4.png)

---

#### **Imágenes promedio por clase**

El promedio condensa las regiones cromáticas dominantes. Las clases humanoides (0 y 4) colapsan en siluetas simétricas; las criaturas (1) muestran masas difusas y verdes/azules; ítems (3) generan formas circulares sin detalle; frutas (2) forman manchas cálidas, coherentes con su paleta.

![Mean 0](../../reports/figures/eda/class_separability/label_mean_class0.png)
![Mean 1](../../reports/figures/eda/class_separability/label_mean_class1.png)
![Mean 2](../../reports/figures/eda/class_separability/label_mean_class2.png)
![Mean 3](../../reports/figures/eda/class_separability/label_mean_class3.png)
![Mean 4](../../reports/figures/eda/class_separability/label_mean_class4.png)

---

#### **Colorimetría por clase**

Las medias RGB reflejan tendencias claras:

- Clases **2** (frutas) → paletas cálidas y valores altos en rojo y verde.
- Clases **0/4** (humanos) → colores neutros, dominancia marrón/gris.
- Clase **1** (criaturas) → saturación elevada en verdes y azules.
- Clase **3** (ítems) → dispersión alta debido a variabilidad temática.

Pese a esto, las desviaciones estándar son amplias en todas las clases, anticipando una fuerte superposición en espacios de color puros.

---

#### **t-SNE: proyección del espacio visual**

La proyección t-SNE confirma la intuición: las clases no forman grupos compactos. Los puntos se mezclan formando un gradiente continuo donde todos los tipos de sprites coexisten sin fronteras nítidas. Las clases sólo se distinguen en zonas muy pequeñas del espacio.

![t-SNE](../../reports/figures/eda/class_separability/tsne_labels.png)

Esta estructura dispersa indica que **la etiqueta de clase no está codificada linealmente en los píxeles**. Cualquier modelo que busque separar clases deberá aprender rasgos altamente no lineales.

---

#### **Silhouette score**

El puntaje silhouette cuantifica la cohesión intraclase y separación interclase.  
Los resultados son negativos tanto en el espacio crudo como en PCA-50:

![Silhouette](../../reports/figures/eda/class_separability/silhouette_scores.png)

Valores:
- Raw pixels: **–0.051**
- PCA-50: **–0.034**

Un valor negativo implica que las instancias están más cerca de otras clases que de la propia. En términos prácticos: **el dataset no presenta clusters naturales para estas cinco etiquetas**.

---

#### **K-means (5 clusters)**

K-means se ejecutó para `k=5` sin usar etiquetas. La matriz de confusión entre predicción de cluster y clase real confirma el solapamiento:

![K-means confusion](../../reports/figures/eda/class_separability/confusion_clusters.png)

Los clusters no corresponden a las clases originales. Algunas clases se dividen en varios clusters, y varios clusters contienen instancias múltiples de distintas etiquetas. Las métricas no supervisadas lo ratifican:

- Adjusted Rand Index: **0.057**
- Normalized Mutual Information: **0.194**

Ambas cercanas a 0 → **alineamiento casi aleatorio**.

---

### **Conclusión técnica**

Este análisis deja claro que las clases del dataset **no son separables de forma lineal ni semicompacta** en su espacio visual original. Aunque cada categoría tiene coherencia estética superficial, las estructuras internas se traslapan profundamente: poses similares, paletas similares, contornos redondeados, saturación inconsistente.

En consecuencia:

1. **Métodos no supervisados no recuperan la estructura real.**  
2. **Un modelo supervisado debe aprender rasgos altamente específicos**: contornos, proporciones, siluetas y microtexturas.  
3. **La etiqueta no es trivial**: requiere redes convolucionales capaces de extraer invariancias espaciales.  
4. **La mezcla visual sugiere que la dificultad del dataset no está en la finura del arte sino en su similitud estructural.**

Este comportamiento explica por qué arquitecturas simples pueden fallar, mientras que modelos convolucionales moderados (o autoencoders previos) logran capturar las señales necesarias.

## 6. Relación entre variables explicativas y variable objetivo


- Ratio de píxeles vacíos vs clase.

### 6.1 — Clasificador Auxiliar CNN (Separabilidad Real entre Clases)

Este experimento entrena una CNN pequeña para evaluar si las clases del dataset contienen un *signal visual fuerte*, es decir, si es posible distinguirlas a partir de sus patrones cromáticos y espaciales sin un modelo profundo.  
El objetivo no es obtener un modelo final, sino medir la **separabilidad visual real** del dataset.

---

### Resultados obtenidos

#### Precisión por época

El modelo alcanza **≈100 % de accuracy en validación** desde muy temprano, lo que indica que las clases poseen patrones visuales extremadamente consistentes.

![Accuracy](../../reports/figures/eda/aux_classifier/accuracy_curve.png)

---

#### Pérdida por época

La pérdida cae a casi cero en solo 1–2 épocas, reforzando el comportamiento de separabilidad fuerte entre clases.

![Loss](../../reports/figures/eda/aux_classifier/loss_curve.png)

- **Loss final:** 0.00007  
- **Accuracy final:** 100 %  

---

### Matriz de confusión

La CNN clasifica *todas* las imágenes de validación correctamente.  
La matriz es diagonal perfecta:

![Confusion matrix](../../reports/figures/eda/aux_classifier/confusion_matrix.png)

Esto solo ocurre cuando los clusters visuales están extremadamente bien definidos.

---

### Interpretación técnica

Los resultados permiten extraer varias conclusiones clave:

#### **1. Separabilidad absoluta**
Una CNN mínima identifica cada clase con precisión perfecta.  
Esto sugiere que:

- dentro de cada clase hay **muy baja variabilidad**,  
- entre clases hay **diferencias visuales claras y robustas**.

#### **2. Las etiquetas no son aleatorias**
El modelo no podría converger así si las labels fueran ruido.  
Las clases parecen representar:

- estilos visuales,
- fuentes gráficas diferentes,
- pipelines/artistas distintos,
- o familias de sprites con estructuras muy similares.

#### **3. Es viable un modelo condicionado por clase**
Dado el comportamiento perfecto:

- los modelos generativos pueden usar conditioning estable,
- se pueden generar estilos diferenciados fácilmente,
- no habrá mezclas espurias entre clases.

---

### Conclusión

Este análisis confirma que:

1. Las clases poseen **identidad visual fuerte**.  
2. El dataset es **limpio, estructurado y altamente separable**.  
3. La arquitectura del modelo generativo puede incorporar conditioning por clase sin riesgo.  

El clasificador auxiliar funciona como evidencia empírica de que la estructura latente observada en PCA, análisis de color, t-SNE y UMAP también se refleja en un modelo discriminativo simple.

---

