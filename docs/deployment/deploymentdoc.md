# Despliegue de la app (AD3 vs AD6)

El objetivo del despliegue fue cerrar el ciclo del proyecto: pasar de modelos entrenados en notebooks a una aplicación interactiva que permitiera **generar sprites nuevos** y, al mismo tiempo, **comparar visualmente** el desempeño del modelo base (AD3) frente al modelo mejorado (AD6) durante el proceso de denoising.  
Como trabajamos con pixel art de 16×16, era importante que la interfaz preservara la estética “pixel‑perfect” al ampliar imágenes y que la evolución paso a paso quedara clara.

La solución se implementó en `app.py` usando Gradio, porque ofrece una vía rápida y ligera para exponer un flujo de inferencia con controles, salidas visuales y servidor HTTP local sin necesidad de backend adicional.

## Artefactos y entorno

El despliegue se apoya en tres piezas:

1. **Pesos de modelos entrenados** (copiados a `data/models/`):
   - AD3: `data/models/ddpm_conditional.keras` (DDPM pixel‑space condicional con U‑Net simple).
   - AD6: `data/models/ddpm_resunet_ad6_ema.keras` o `data/models/ddpm_resunetema.keras` (EMA preferida). Si no existe, se usa `data/models/ddpm_resunet_ad6.keras` como respaldo.
2. **Schedule de difusión de AD6**:
   - `data/models/schedule_ad6.npz` con `betas`, `alphas` y `alphas_cumprod` (cosine schedule).
3. **Dataset auxiliar para clases** (opcional):
   - `data/intermediate/pixel_art_data.npz`, usado para detectar automáticamente las clases disponibles.

El entorno recomendado es el venv `env` con Python 3.10+ y dependencias principales: TensorFlow 2.20.0, Gradio, NumPy y PIL. La GPU es opcional; TensorFlow usará GPU automáticamente si está disponible.

## Proceso de despliegue en `app.py`

El diseño de la app replica fielmente la lógica de muestreo de AD3 y AD6:

- **Carga de modelos.** AD3 se carga al iniciar la app (es liviano). AD6 se carga **solo al primer Generate**, porque es más grande y su carga puede tardar varios segundos. Para garantizar que Keras pueda reconstruir el grafo, se registran las capas personalizadas `TimestepEmbedding` y `SplitScaleShift`, y se carga con `safe_mode=False` y `compile=False`.
- **Muestreo reverse diffusion.**
  - AD3 usa un schedule lineal de 300 pasos.
  - AD6 usa el cosine schedule guardado en `schedule_ad6.npz` y aplica classifier‑free guidance, controlado por el slider `guidance_scale`.
- **Visualización pixel‑perfect.** Cada frame intermedio y el resultado final se convierten a PIL y se reescalan a 256×256 con **nearest‑neighbor**, evitando blur al ampliar sprites pequeños.
- **Comparación lado a lado.** La UI muestra AD3 a la izquierda y AD6 a la derecha, con galerías separadas de frames para apreciar diferencias en convergencia y nitidez.

## Cómo ejecutar la app

1. Activar el entorno:
   - Windows: `.\env\Scripts\activate`
   - Unix: `source env/bin/activate`
2. (Si hace falta) instalar dependencias: `pip install -r requirements.txt`.
3. Ejecutar desde la raíz del repo:
   - `python app.py`  
   - o explícitamente en Windows: `.\env\Scripts\python.exe app.py`
4. Abrir en el navegador `http://127.0.0.1:7860`.
5. Seleccionar clase y parámetros (pasos, ruido, guidance de AD6, seed) y presionar **Generate**.  
   El primer generate puede tardar más por la carga diferida de AD6.

## Resolución de problemas

- Si AD6 no carga, verificar que alguno de estos archivos exista en `data/models/`:  
  `ddpm_resunet_ad6_ema.keras`, `ddpm_resunetema.keras` o `ddpm_resunet_ad6.keras`, y que `schedule_ad6.npz` esté presente.
- Para ver logs completos, ejecutar con salida sin buffer: `python -u app.py`.
- Si falta `pixel_art_data.npz`, la app sigue funcionando; solo mostrará etiquetas genéricas (Class 0…N‑1).

## Mantenimiento y reproducción

- Para regenerar AD6 sin Lambdas inseguras, reentrenar/guardar con `scripts/training/AD6.py` (usa `SplitScaleShift`).
- Para comparar modelos sin interfaz, usar `scripts/evaluation/compare_models.py` con el venv activo y los artefactos en `data/models/`.

## Vista comparativa (AD3 vs AD6)

La siguiente animación captura el denoising de ambos modelos con el mismo seed/clase. AD3 (izquierda) tiende a producir bordes algo más suaves y más grano; AD6 (derecha) converge más rápido y conserva contornos más nítidos, lo que es deseable para pixel art.

![AD3 vs AD6](../../reports/deployment/AD3%20vs%20AD6.gif)
