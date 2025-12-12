# Informe de salida del proyecto

## Resumen ejecutivo

El proyecto **PixelGen** tuvo como meta construir un sistema generativo de pixel art capaz de producir nuevos sprites de 16×16 px a partir de un dataset real. Se desarrolló un pipeline completo que incluye adquisición y limpieza de datos, análisis exploratorio, entrenamiento y comparación de varios modelos generativos, selección de un modelo final y despliegue de una interfaz interactiva para generación y comparación visual.

Los principales resultados son:

- Un dataset procesado y documentado de **89.400 sprites RGB 16×16** con etiquetas usadas para condicionamiento.
- Cinco experimentos generativos base (autoencoder, DDPM pixel‑space, DDPM condicional, latent diffusion y ablación de hiperparámetros) más una variante mejorada (AD6).
- Selección del **DDPM condicional (AD3)** como modelo final base por su mejor fidelidad perceptual (FID‑like promedio ~19.2) y coherencia por clase.
- Despliegue de una app Gradio (`app.py`) que permite comparar **AD3 vs AD6** lado a lado y visualizar el proceso de denoising en tiempo real.

Estos entregables cumplen el alcance definido en el Project Charter y dejan una base reproducible para iterar sobre modelos de difusion más avanzados.

## Objetivo y alcance cumplido

El objetivo fue generar sprites tipo pixel art útiles para prototipado y creación de assets en videojuegos, con un modelo de difusion **reproducible, evaluable y desplegable**.  
El alcance comprometido se cumplió mediante:

- Uso del dataset Pixel Art de Kaggle y construcción de un pipeline de datos reproducible.
- EDA completo con análisis de calidad, estructura latente y separabilidad por clase.
- Entrenamiento y comparación de modelos baseline y difusivos.
- Evaluación con métricas proxy tipo FID‑like y revisión cualitativa por clase.
- Demo interactivo en Gradio con visualización del denoising.

Quedaron fuera del alcance entrenamientos a gran escala (resoluciones mayores) o integración en un producto comercial.

## Proceso realizado

### 1) Datos: adquisición, limpieza y EDA

La fuente fue el dataset público **ebrahimelgazar/pixel‑art** (Kaggle, licencia Apache 2.0). La adquisición se automatizó mediante `scripts/data_acquisition/main.py`, garantizando reproducibilidad.

El procesamiento generó `data/intermediate/pixel_art_data.npz`, con:

- **Imágenes** en float32 normalizadas a **[0,1]**, tamaño fijo **16×16×3**.
- **Etiquetas** numéricas usadas como clases de condicionamiento (5 clases en el dataset intermedio).
- Validación de consistencia entre `labels.csv`, arrays `.npy` e imágenes.

En `docs/data/data_summary.md` se documentaron hallazgos clave:

- Dataset limpio, sin faltantes ni corrupción.
- Alta redundancia visual (duplicados identificados), sin afectar la distribución de intensidades tras limpieza.
- Separabilidad no lineal pero fuerte entre clases, validada con un clasificador auxiliar que alcanza 100% accuracy, justificando el condicionamiento por clase.
- PCA mostró estructura latente compacta: gran parte de la varianza se explica con pocas componentes, lo que facilita el aprendizaje generativo.

### 2) Modelamiento: baselines y variantes de difusion

Se entrenaron y documentaron los experimentos AD1–AD5 (`docs/modeling/baseline_models.md`, `docs/modeling/model_report.md`):

- **Exp1 (Autoencoder/DAE):** baseline no difusivo para reconstrucción. Buenas métricas (SSIM ~0.949; PSNR ~24.8; MSE ~0.0043) pero sin capacidad de generar muestras nuevas.
- **Exp2 (DDPM pixel‑space):** modelo generativo estable sin condicionamiento. FID‑like ~21.85.
- **Exp3 (DDPM condicional – AD3):** U‑Net con embedding de clase. Mejor resultado global; FID‑like por clase entre ~16.9 y ~21.9, promedio ~19.2.
- **Exp4 (Latent diffusion):** DDPM sobre latentes de un autoencoder; mostró potencial de eficiencia, pero dependía de la calidad del autoencoder y no tuvo métricas comparables completas.
- **Exp5 (Ablación):** barridos de T y capacidad confirmaron que reducir demasiado T degrada calidad y que aumentar capacidad mejora hasta cierto punto.

Como iteración final se desarrolló **AD6** (`scripts/training/AD6.py`), incorporando recomendaciones basadas en literatura y la comparación con modelos de referencia:

- U‑Net residual con mayor capacidad.
- Conditioning tipo FiLM y **classifier‑free guidance**.
- Cosine schedule y EMA de pesos.

AD6 se entrenó y se guardó en `data/models/` junto con su schedule.

### 3) Evaluación y comparación

La evaluación cuantitativa se apoyó en una métrica proxy **Feature‑FID** basada en PCA (consistente con AD5), y en revisión cualitativa por clase.  
Para comparar directamente AD3 vs AD6 se creó `scripts/evaluation/compare_models.py`, que:

- Genera muestras por clase con ambos modelos.
- Calcula Feature‑FID por clase y promedio.

Este script permite reproducir comparaciones sin interfaz, y queda listo para reportar mejoras numéricas de AD6 a medida que se ejecuten nuevos entrenamientos.

### 4) Despliegue

El despliegue está documentado en `docs/deployment/deploymentdoc.md`. Se implementó `app.py` con Gradio para:

- Generar imágenes nuevas a partir de un control de clase, pasos, ruido y seed.
- Mostrar **dos salidas en paralelo**: AD3 (izquierda) vs AD6 (derecha).
- Visualizar el denoising mediante galerías de frames.
- Mantener pixel art nítido mediante reescalado nearest‑neighbor.

El gif comparativo se dejó en `reports/deployment/AD3 vs AD6.gif` y está embebido en la documentación.

## Resultados finales

- **Modelo final base:** AD3 (DDPM condicional) por mejor fidelidad perceptual y estabilidad entre clases.
- **Mejora exploratoria:** AD6, con arquitectura más robusta y guidance, mostró mejoras cualitativas en nitidez y convergencia; queda abierta la cuantificación completa con el script de comparación.
- **Pipeline completo:** datos → modelamiento → evaluación → despliegue reproducible, con modelos y schedules versionados en `data/models/`.

## Entregables

- **Datos procesados y documentados:** `data/intermediate/pixel_art_data.npz`, `docs/data/*`.
- **Experimentos de entrenamiento:** `scripts/training/AD1.ipynb` … `AD5.ipynb`, `scripts/training/AD6.py`.
- **Modelos entrenados:** `data/models/ddpm_conditional.keras`, `data/models/ddpm_resunet*_ad6*.keras`, `data/models/schedule_ad6.npz`.
- **Evaluación:** `scripts/evaluation/compare_models.py`.
- **App de despliegue:** `app.py` (Gradio).
- **Documentación:** `docs/business_understanding/project_charter.md`, `docs/modeling/*`, `docs/deployment/deploymentdoc.md`, este informe.

## Lecciones aprendidas

- **Condicionamiento es clave:** incorporar clase en DDPM mejora coherencia y reduce FID‑like, validado empíricamente en AD3.
- **Calidad del pipeline de datos domina:** normalización consistente, control de duplicados y verificación de labels facilitaron estabilidad de entrenamiento.
- **Métricas proxy ayudan a iterar:** Feature‑FID (PCA) permitió comparar rápidamente variantes sin depender de modelos pesados.
- **Serialización en Keras:** capas Lambda con funciones anónimas pueden dificultar despliegue; reemplazarlas por capas custom (`SplitScaleShift`) mejora robustez.
- **Costo computacional vs calidad:** T y capacidad deben equilibrarse; reducir pasos acelera pero afecta nitidez del denoising.

## Impacto para negocio / usuarios

Para los stakeholders (estudios de videojuegos y comunidad académica), el sistema permite:

- **Acelerar prototipado visual:** generación inmediata de sprites para explorar estilos y variantes.
- **Consistencia por clase/estilo:** habilita creación de familias coherentes de assets.
- **Herramienta demostrativa y educativa:** la visualización del denoising hace transparente el funcionamiento del modelo y apoya divulgación técnica.

## Limitaciones y trabajo futuro

- **Resolución limitada (16×16):** el estilo pixel art se preserva, pero no cubre necesidades de assets de mayor tamaño.
- **Semántica de etiquetas:** la fuente original no documenta clases; el condicionamiento es útil, pero puede refinarse con un mapeo semántico más claro.
- **Métrica FID‑like:** es un proxy; evaluar FID completo o métricas perceptuales adicionales daría mayor robustez.
- **AD6 requiere benchmarking formal:** ejecutar `compare_models.py` con suficientes muestras y reportar métricas en futuras iteraciones.
- **Mejoras posibles:** U‑Net más profunda/atención ligera, CFG con escalas adaptativas, o latent diffusion apoyada en un VQ‑VAE entrenado in‑repo.

## Agradecimientos

Gracias al equipo del proyecto por la colaboración durante todas las etapas, y al instructor del diplomado por la guía metodológica. También se agradece a la comunidad de Kaggle por la disponibilidad del dataset y a los revisores académicos del diplomado por el feedback recibido.
