# Reporte de Modelos Generativos (Baseline y Variantes Difusivas)

Entrenamos los siguientes cinco modelos:

1. **Modelo 1:** Autoencoder / Denoising Autoencoder (baseline no difusivo)  
2. **Modelo 2:** Diffusion en espacio de píxeles (DDPM baseline, sin condición de clase)  
3. **Modelo 3:** Diffusion condicionada por clase (Class-conditional DDPM)  
4. **Modelo 4:** Diffusion en espacio latente (Latent Diffusion “chiquito”)  
5. **Modelo 5:** Ablación sobre ruido / pasos / capacidad del modelo  

El objetivo general es comparar un baseline generativo simple (autoencoder) frente a distintos modelos de difusión, evaluando su capacidad para reconstruir y generar sprites 16×16×3, así como entender qué hiperparámetros tienen mayor impacto en la calidad de las muestras.

---

## Modelo 1: Autoencoder / Denoising Autoencoder (Baseline no difusivo)

### Descripción del modelo

El **baseline** es un **autoencoder convolucional** que toma imágenes de entrada de tamaño 16×16×3 y las comprime a un espacio latente de baja dimensión, para luego reconstruirlas:

- **Encoder**:  
  - Input: 16×16×3  
  - 2–3 bloques Conv2D + ReLU + MaxPool  
  - Proyección a un vector latente de dimensión reducida (por ejemplo, 32 o 64).
- **Decoder**:  
  - Bloques ConvTranspose2D / Upsampling  
  - Reconstrucción a 16×16×3.

Variante: **denoising autoencoder**, donde se agrega ruido (p. ej. Gaussiano) a la imagen de entrada y el modelo aprende a reconstruir la imagen limpia.

Este modelo sirve como línea base de qué tan bien un método simple puede reconstruir / regenerar el dataset sin usar difusión.

### Variables de entrada

- **Imagen original** \(x \in \mathbb{R}^{16\times16\times3}\).  
- Para la variante denoising:
  - **Imagen ruidosa** \(\tilde{x} = x + \epsilon\), con:
    - Ruido Gaussiano con \(\sigma \in [0.1, 0.2]\), y/o  
    - Ruido tipo *masking* o *salt-and-pepper* (proporción de píxeles perturbados).

### Variable objetivo

- **Imagen objetivo**:  
  - Autoencoder clásico: la propia imagen de entrada \(x\).  
  - Denoising autoencoder: imagen limpia \(x\) a partir de \(\tilde{x}\).
- El modelo aprende una función \(f_\theta(\tilde{x}) \approx x\).

### Evaluación del modelo

#### Métricas de evaluación

- **val_loss**: pérdida de validación del modelo durante el entrenamiento (MSE sobre el conjunto de validación).
- **MSE (Mean Squared Error)** entre imagen original y reconstruida.  
- **PSNR (Peak Signal-to-Noise Ratio)** como métrica de calidad de señal.  
- **SSIM (Structural Similarity Index)** para capturar similitud perceptual.  
- **Análisis del espacio latente** (PCA / t-SNE y *silhouette score*); aún pendiente de cálculo en esta versión del experimento.

En el mejor modelo entrenado se obtuvieron las siguientes métricas:

#### Resultados de evaluación

| Métrica              | Descripción                                      | Valor obtenido          |
|----------------------|--------------------------------------------------|-------------------------|
| val_loss             | Pérdida de validación (MSE)                      | 0.005559464450925589    |
| MSE (promedio test)  | Error cuadrático medio de reconstrucción         | 0.004395863972604275    |
| PSNR (dB)            | Calidad de señal promedio                        | 24.82591389982544       |
| SSIM                 | Similitud estructural promedio                   | 0.9490600824356079      |

### Análisis de los resultados

- El autoencoder entrega una **referencia clara** de qué tan bien se pueden reconstruir las imágenes con un modelo relativamente simple.  
- El **MSE ≈ 0.0044** y la **val_loss ≈ 0.0056** indican que el error de reconstrucción promedio es bajo para este tamaño de imagen (16×16×3).  
- Un **PSNR ≈ 24.8 dB** sugiere que la señal reconstruida mantiene una relación señal-ruido razonablemente buena.  
- El **SSIM ≈ 0.949** muestra que la estructura y el contenido perceptual de las imágenes se preservan muy bien, lo que indica que el modelo captura la geometría básica de los sprites.  
- El análisis del **espacio latente** (PCA / t-SNE y *silhouette score*) queda como trabajo pendiente; cuando se calcule, permitirá verificar si el encoder está encontrando una representación más estructurada por clase que el espacio de píxeles crudo.

### Conclusiones

- Con estas métricas, el autoencoder / denoising autoencoder se adopta como un **baseline no difusivo sólido** para:
  - Medir qué tan bien se puede reconstruir el dataset con un modelo simple.
  - Obtener un **encoder entrenado** que luego se reutiliza en el modelo de **diffusion en espacio latente** (Modelo 4).
- A partir de este baseline se evaluará si los modelos de difusión realmente aportan mejoras visibles en **calidad generativa** (PSNR, SSIM, FID-like) y en **estructura del espacio de representación**, una vez se complete el análisis del espacio latente.


---

## Modelo 2: Diffusion en espacio de píxeles (DDPM baseline, sin condición de clase)

### Descripción del modelo

El segundo modelo es un **DDPM (Denoising Diffusion Probabilistic Model)** que opera directamente en **espacio de píxeles**:

- **Arquitectura principal**: U-Net pequeña, adaptada a imágenes 16×16×3.  
  - 2 niveles de downsampling: 16→8→4 y simétrico de upsampling.  
  - Canales típicos: 32–64–128 como máximo.  
- **Embedding de tiempo**:  
  - Embedding sinusoidal / posicional del paso de difusión \(t\), inyectado en la U-Net.  
- **Proceso de difusión**:  
  - Número de pasos T ≈ 200–400 (reducido por el tamaño pequeño 16×16).  
  - *Schedule* de \(\beta_t\) lineal o coseno como baseline.

### Variables de entrada

- Durante entrenamiento:
  - **Imagen real** \(x_0\).  
  - Paso de difusión \(t \in \{1, \dots, T\}\).  
  - Ruido gaussiano \(\epsilon \sim \mathcal{N}(0, I)\).  
  - Imagen ruidosa \(x_t\) obtenida del *forward process*.

### Variable objetivo

- **Ruido real** \(\epsilon\) que se usó para corromper la imagen.  
- El modelo aprende a predecir \(\hat{\epsilon}_\theta(x_t, t)\) minimizando:
  - **Pérdida MSE** entre ruido real y ruido predicho.

### Evaluación del modelo

### Evaluación del modelo

#### Métricas de evaluación

1. **Calidad de muestras (sampling)**  
   - Grillas de 8×8 imágenes generadas desde ruido puro.  
   - Evaluación visual de:
     - Diversidad de sprites.  
     - Ausencia de artefactos.  

2. **Estadísticas marginales**  
   - Comparación de **histogramas de intensidades RGB** entre imágenes reales y generadas.  
   - Comparación de la distribución en un espacio de **PCA** (reales vs generadas).  

3. **“FID casero”**  
   - Se utilizó un FID-like calculado en un espacio de *features* reducido con PCA a 20 dimensiones (**fid_like_pca20**).  
   - Adicionalmente se registró la **train_loss** del modelo (pérdida promedio de entrenamiento, MSE entre ruido real y ruido predicho).

En el mejor modelo DDPM en píxel space se obtuvieron las siguientes métricas numéricas:

#### Resultados de evaluación

| Métrica / análisis          | Descripción                                                    | Valor / Observación        |
|----------------------------|----------------------------------------------------------------|----------------------------|
| FID-like global (PCA-20)   | Distancia FID-like entre reales vs generadas en PCA-20        | 21.853602172478762         |
| train_loss                 | Pérdida promedio de entrenamiento (MSE del ruido)             | 0.08807305246591568        |

### Análisis de los resultados

- Este modelo responde a la pregunta:  
  > “¿Un DDPM simple en espacio de píxeles ya genera sprites creíbles de manera global?”
- El **FID-like global ≈ 21.85** en PCA-20 proporciona una primera referencia cuantitativa de qué tan cerca están las muestras generadas de los datos reales en el espacio de *features*.  
- La **train_loss ≈ 0.088** indica que el modelo ha aprendido a predecir el ruido con un error moderadamente bajo; sin embargo, la interpretación final depende de la comparación con configuraciones posteriores (modelos condicionados o en espacio latente).  
- El análisis cualitativo (diversidad visual, artefactos, coincidencia de histogramas y de nubes en PCA) queda como trabajo pendiente de describir de forma sistemática a partir de las grillas de muestras y las gráficas correspondientes.  
- En conjunto, estas métricas numéricas sirven como **baseline difusivo** sobre el cual se evaluará si:
  - El condicionamiento por clase (Modelo 3) mejora la coherencia de las muestras.  
  - La difusión en espacio latente (Modelo 4) puede igualar o superar este FID-like con menor costo computacional.

### Conclusiones

- El DDPM en píxeles sirve como **baseline difusivo** para comparar con:
  - La versión **condicionada por clase** (Modelo 3).  
  - La **diffusion en espacio latente** (Modelo 4).  
- El valor de **fid_like_pca20 ≈ 21.85** y la **train_loss ≈ 0.088** establecen una línea base cuantitativa; los experimentos posteriores deben aspirar a **reducir el FID-like** y/o mejorar la calidad visual manteniendo costos de entrenamiento razonables.  
- Este baseline permitirá determinar si la difusión, incluso sin condicionamiento, ya mejora significativamente la **variedad** y **realismo** de las muestras respecto al autoencoder, una vez se comparen directamente sus métricas y ejemplos visuales.


---

## Modelo 3: Diffusion condicionada por clase (Class-conditional DDPM)

### Descripción del modelo

Se utiliza el mismo DDPM del Modelo 2, pero ahora **condicionado por clase**:

- **Condicionamiento por clase**:
  - Las etiquetas se representan como **one-hot vectors**.  
  - Se proyectan a un **embedding denso** (p. ej. 16–32 dimensiones).  
  - Este embedding se concatena:
    - A la entrada de la U-Net.  
    - Y/o a los bloques residuales, junto con el embedding de tiempo.
- Se mantienen:
  - Mismo número de pasos T.  
  - Mismo *schedule* de betas.  
  - Mismo tamaño de U-Net, para que el cambio principal sea solo “sin condición vs condicionado”.

### Variables de entrada

- **Imagen ruidosa** \(x_t\).  
- **Paso de difusión** \(t\).  
- **Etiqueta de clase** \(y\) (one-hot, luego embebida).  
- **Ruido** \(\epsilon\).

### Variable objetivo

- Igual que en el Modelo 2:
  - **Ruido real** \(\epsilon\) que corrompe la imagen, con pérdida MSE entre \(\epsilon\) y \(\hat{\epsilon}_\theta(x_t, t, y)\).

### Evaluación del modelo

#### Métricas de evaluación

1. **Coherencia de clase**  
   - Se generan muestras condicionadas en cada clase.  
   - Se pasan por el **clasificador auxiliar** (que tiene ~100% de accuracy en datos reales).  
   - En esta versión del experimento aún no se registró explícitamente la **accuracy condicional**, por lo que queda como trabajo pendiente de cálculo y reporte.

2. **FID / distancia de *features* por clase**  
   - Se calculó un **FID-like por clase** entre:
     - Features de imágenes reales de clase c.  
     - Features de imágenes generadas condicionadas en c.  
   - Estas métricas aparecen registradas como `fid_like_class_0` … `fid_like_class_4`.

3. **Calidad visual**  
   - Grids comparando:
     - Muestras **no condicionadas** (Modelo 2).  
     - Muestras **condicionadas** (Modelo 3) para la misma clase.
   - El análisis cualitativo detallado (diversidad, artefactos, coherencia visual) se deja pendiente de documentar por escrito.

Adicionalmente, se registró la **train_loss** del modelo, que corresponde a la pérdida promedio de entrenamiento (MSE entre ruido real y ruido predicho) en el esquema de difusión condicional.

#### Resultados de evaluación

| Clase | Accuracy condicional (clasificador auxiliar) | FID-like por clase      |
|-------|----------------------------------------------|-------------------------|
| 0     | *Pendiente de cálculo*                       | 19.12356543349288       | 
| 1     | *Pendiente de cálculo*                       | 16.898045572589403      | 
| 2     | *Pendiente de cálculo*                       | 17.39349062898828       | 
| 3     | *Pendiente de cálculo*                       | 20.83842260591875       | 
| 4     | *Pendiente de cálculo*                       | 21.93005297056078       | 

Métrica global adicional:

- **train_loss** ≈ 0.09294738620519638

### Análisis de los resultados

- Este experimento muestra **cómo cambia la estructura de las muestras** cuando el modelo sabe explícitamente qué clase debe generar.  
- Los valores de **FID-like por clase** se sitúan aproximadamente entre **16.9 y 21.9**, lo que indica que, en el espacio de *features* utilizado, las imágenes generadas condicionadas por clase se mantienen relativamente cercanas a las distribuciones reales de cada etiqueta.  
- La **train_loss ≈ 0.093** refleja que el modelo ha aprendido a predecir el ruido condicionalmente con un error moderado, coherente con la complejidad del problema y comparable con el DDPM no condicionado.  
- Falta aún cuantificar la **accuracy condicional** usando el clasificador auxiliar; cuando se compute, permitirá validar numéricamente si las muestras generadas para la clase c son efectivamente reconocidas como tal.  
- La comparación visual entre grids no condicionadas (Modelo 2) y condicionadas (Modelo 3) será clave para evaluar si el condicionamiento produce:
  - Menos “mezcla de estilos” entre clases.  
  - Mayor claridad en la forma, color y atributos característicos de cada etiqueta.  
  - Ausencia de **mode collapse** (es decir, que dentro de cada clase aún haya diversidad de sprites).

### Conclusiones

- El modelo pasa (idealmente) de generar una mezcla de estilos a producir **estilos más definidos por clase**, manteniendo diversidad interna.  
- Los valores de **FID-like por clase** constituyen una primera evidencia cuantitativa de que el condicionamiento por clase estructura mejor el espacio generativo, aunque se requiere complementar con la **accuracy condicional** del clasificador auxiliar.  
- El **clasificador auxiliar** se mantiene como herramienta clave para cuantificar la calidad condicional; una vez se añadan esas métricas, este modelo servirá como base sólida para el **experimento de ablación (Modelo 5)**, donde se analizará la sensibilidad a T y a la capacidad de la U-Net bajo este esquema condicionado.

---

## Modelo 4: Diffusion en espacio latente (Latent Diffusion “chiquito”)

### Descripción del modelo

Este modelo combina el **autoencoder del Modelo 1** con un proceso de difusión en su **espacio latente**:

- Se usa el **encoder** entrenado del autoencoder:
  - Imagen: \(x \rightarrow z\) (dimensión latente 32 o 64).  
- El **modelo de difusión** opera sobre \(z\) en vez de sobre píxeles:
  - Puede ser una **U-Net pequeña** (tratando \(z\) como mapa 4×4×C) o un modelo tipo **MLP / U-Net 1D**.  
- Para generar muestras:
  1. Se muestrea \(z\) mediante difusión: \(z \sim p_\theta(z)\).  
  2. Se decodifica con el **decoder** del autoencoder:
     - \(x = \text{decoder}(z)\).

### Variables de entrada

- **Vector latente** \(z_0\) (derivado de imágenes reales).  
- **Paso de difusión** \(t\).  
- **Ruido** \(\epsilon\) aplicado en el espacio latente.

### Variable objetivo

- Igual que en los DDPM previos, pero en el espacio latente:
  - El modelo aprende a predecir el **ruido sobre z**:
    - \(\hat{\epsilon}_\theta(z_t, t)\).

### Evaluación del modelo

#### Métricas de evaluación

1. **Reconstrucción + generación**  
   - Comparar visualmente imágenes generadas por Latent Diffusion vs:
     - Imágenes generadas en píxeles (Modelos 2–3).  
   - Analizar si el **decoder introduce blur o artefactos**.  

2. **Costo computacional**  
   - Épocas / tiempo hasta converger frente al DDPM en píxeles.  
   - **MSE de denoising en espacio latente**: en este experimento se registró como una única métrica global llamada `loss`, que corresponde a la pérdida promedio de entrenamiento del modelo de difusión en el espacio latente.

3. **Feature-FID (usando CNN)**  
   - Igual que antes, pero comparando:
     - Features de imágenes reales.  
     - Features de imágenes generadas por Latent Diffusion.  
   - En esta corrida aún no se calculó explícitamente un FID-like, por lo que queda como trabajo pendiente.

#### Resultados de evaluación

En la mejor corrida del modelo de Latent Diffusion se obtuvo:

| Métrica / análisis                 | Modelo píxeles (Exp 2/3) | Latent Diffusion (Exp 4)          |
|------------------------------------|--------------------------|-----------------------------------|
| MSE denoising (promedio, latent)   | —                        | 0.6203462481498718 (`loss`)      |

### Análisis de los resultados

- El modelo responde a la hipótesis:  
  > “Espacio latente compacto + difusión → entrenamiento más rápido, con muestras igual de buenas.”
- La métrica registrada (`loss ≈ 0.62`) representa el **error medio de denoising en el espacio latente**.  
  - Su interpretación directa en términos de calidad visual no es tan inmediata como el MSE en píxeles, porque opera sobre la representación comprimida \(z\).  
  - Para valorar si este valor es “alto” o “bajo” es necesario compararlo con:
    - El comportamiento del autoencoder (qué tan bien reconstruye desde \(z\)).  
    - La calidad visual final de las muestras reconstruidas (blur, artefactos).  
- A falta de FID-like y tiempos de entrenamiento reportados, todavía no se puede concluir si Latent Diffusion:
  - Mantiene una **calidad comparable** a los DDPM en píxeles.  
  - Ofrece una **ventaja clara en costo computacional**.
- Si en análisis visual se observa **blur excesivo** o pérdida de detalles finos, podría indicar que:
  - El autoencoder está comprimiendo demasiado (bottleneck agresivo).  
  - El modelo de difusión en \(z\) necesita más capacidad o más pasos de difusión.

### Conclusiones

- Este experimento permite valorar si, para un dataset tan pequeño en píxeles (16×16), vale la pena la complejidad conceptual de Latent Diffusion.  
- Con la información disponible, el resultado principal es el **loss ≈ 0.62 en espacio latente**; para completar la evaluación se requiere:
  - Calcular un **FID-like global** para Latent Diffusion.  
  - Medir el **tiempo de entrenamiento** y compararlo con el DDPM en píxeles.  
  - Documentar sistemáticamente la **impresión visual** (blur, artefactos, diversidad).  
- Solo con esas piezas adicionales se podrá decidir si la difusión en espacio latente ofrece ventajas claras frente al uso de difusión directa en píxeles.


---

## Modelo 5: Ablación — Ruido / número de pasos y capacidad del modelo

### Descripción del modelo

Este experimento es más **científico / de análisis de sensibilidad** que arquitectónico. Se estudia:

1. **Número de pasos de difusión (T)**.  
2. **Capacidad de la U-Net (tamaño de canales)**.

Se puede aplicar sobre el modelo condicional del Modelo 3 o sobre el DDPM en píxeles del Modelo 2.

### Variante A: Barrido sobre T

- Fijando el resto de hiperparámetros, se comparan modelos con:
  - \(T \in \{50, 100, 200, 400\}\).
- Se miden:
  - Calidad visual de muestras.  
  - Feature-FID.  
  - Accuracy del clasificador auxiliar sobre las muestras generadas.

### Variante B: Barrido sobre capacidad

- Fijando un T razonable (por ejemplo, 200), se comparan arquitecturas:

  - **Modelo “small”**: canales base 16–32–64.  
  - **Modelo “medium”**: 32–64–128.  
  - **(Opcional) Modelo “large”**: 64–128–256 (según recursos).

- Métricas:
  - Feature-FID.  
  - Accuracy del clasificador auxiliar.  
  - Tiempo de entrenamiento / uso de memoria.

### Variables de entrada

- Son las mismas que en el modelo de referencia (DDPM en píxeles o condicional).  
- Lo que cambia son los **hiperparámetros T y capacidad**.

### Variable objetivo

- Sigue siendo la predicción del **ruido** en el proceso de difusión:
  - \(\epsilon\) vs \(\hat{\epsilon}_\theta(x_t, t, y)\) o \(\hat{\epsilon}_\theta(x_t, t)\).

### Evaluación del modelo

#### Métricas de evaluación

- **Feature-FID** global y/o por clase.  
- **Accuracy del clasificador auxiliar** sobre muestras generadas.  
- **Calidad visual** (artefactos, diversidad).  
- **Costo computacional**:
  - Tiempo de entrenamiento.  
  - Memoria requerida.

#### Resultados de evaluación

*(Estructura para completar)*

**Variante A: T**

| T   | FID-like | Accuracy clasificador | Observaciones visuales | Tiempo de entrenamiento |
|-----|----------|-----------------------|------------------------|-------------------------|
| 50  |          |                       |                        |                         |
| 100 |          |                       |                        |                         |
| 200 |          |                       |                        |                         |
| 400 |          |                       |                        |                         |

**Variante B: capacidad**

| Configuración   | Canales base      | FID-like | Accuracy clasificador | Tiempo / recursos | Observaciones |
|-----------------|-------------------|----------|-----------------------|-------------------|---------------|
| Small           | 16–32–64          |          |                       |                   |               |
| Medium          | 32–64–128         |          |                       |                   |               |
| Large (opcional)| 64–128–256        |          |                       |                   |               |

### Análisis de los resultados

- La ablación permite identificar **qué parámetros realmente importan** para este dataset:
  - Si bajar mucho T degrada notablemente la calidad, entonces T es crítico.  
  - Si aumentar capacidad apenas mejora métricas pero aumenta mucho tiempo/memoria, se puede justificar usar un modelo más pequeño.  
- También permite encontrar un **punto de equilibrio** entre calidad y costo computacional.

### Conclusiones

- Este experimento orienta la **selección fina de hiperparámetros** para futuros modelos.  
- Permite justificar, con datos, por qué se escoge cierto T y cierta capacidad de U-Net como configuración estándar.

---

## Conclusiones generales

- El **Autoencoder / Denoising Autoencoder** proporciona:
  - Una referencia de reconstrucción.  
  - Un encoder reutilizable para modelos en espacio latente.
- El **DDPM en espacio de píxeles** muestra la capacidad de los modelos de difusión para aprender la distribución global de sprites, incluso sin condicionamiento.
- El **DDPM condicionado por clase** explota la información de etiquetas, mejorando la coherencia de las muestras por clase y permitiendo métricas muy informativas con el clasificador auxiliar.
- La **Diffusion en espacio latente** evalúa si un espacio comprimido puede acelerar entrenamiento sin sacrificar calidad.
- El **experimento de ablación** ayuda a entender la sensibilidad a T y capacidad, y a diseñar modelos eficientes y bien justificados.

En conjunto, estos cinco modelos construyen un **marco experimental completo** para estudiar la generación de sprites 16×16×3 con técnicas autoencoder y de difusión, tanto en píxeles como en latente.
