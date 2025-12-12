# Despliegue de la app (AD3 vs AD6)

Este documento describe el despliegue del demo interactivo del proyecto **PixelGen**. El propósito del despliegue es cerrar el ciclo de vida del modelo: pasar de notebooks experimentales (AD3 y AD6) a una aplicación accesible para usuarios no técnicos, que permita generar sprites nuevos y comparar visualmente el proceso de denoising de ambos modelos.

La interfaz se implementa en `app.py` usando Gradio porque:
- Reduce fricción de despliegue (un solo script Python).
- Permite interacción inmediata con controles de inferencia.
- Facilita la visualización paso a paso, clave para explicar difusión.

## 1. Artefactos requeridos

El despliegue usa artefactos guardados en `data/models/`:

- **AD3 (modelo base, DDPM condicional):**  
  `data/models/ddpm_conditional.keras`
- **AD6 (modelo mejorado, DDPM residual con guidance):**  
  `data/models/ddpm_resunet_ad6_ema.keras` o `data/models/ddpm_resunetema.keras` (EMA preferida)  
  fallback: `data/models/ddpm_resunet_ad6.keras`
- **Schedule AD6 (cosine):**  
  `data/models/schedule_ad6.npz` con `betas`, `alphas` y `alphas_cumprod`
- **Dataset auxiliar para clases (opcional):**  
  `data/intermediate/pixel_art_data.npz` (si no existe, la app infiere el número de clases desde el embedding).

Estos archivos son el “contrato” mínimo para ejecutar el demo.

## 2. Entorno e infraestructura

### 2.1 Entorno local (demo)

El demo está pensado para ejecución local con un entorno virtual:
- **Python:** 3.10+  
- **Librerías:** TensorFlow 2.20.0, Gradio, NumPy, PIL  
- **Hardware:** CPU suficiente; GPU opcional (TensorFlow usa GPU automáticamente).

### 2.2 Infraestructura objetivo (producción)

Para un despliegue productivo (p. ej. en un estudio de videojuegos), el flujo recomendado es:

1. **Empaquetado** en contenedor (Docker) con el venv o requirements congelados.  
2. **Servicio de inferencia** (FastAPI/Gradio “mount”) que expone un endpoint de generación.  
3. **Front-end ligero** (Gradio o web propia) consumiendo el endpoint.  
4. **Persistencia/versionado** de modelos en un bucket (S3/GCS/Azure Blob) con tags por versión.

La naturaleza del modelo (difusión) implica latencia proporcional al número de pasos, por lo que una GPU de entrada mejora sustancialmente la experiencia de usuario.

## 3. Código de despliegue: funcionamiento, eficiencia y escalabilidad

### 3.1 Flujo funcional

`app.py` implementa el pipeline completo de inferencia:

1. **Carga de modelos:**  
   - AD3 se carga al iniciar la app (modelo pequeño).  
   - AD6 se carga *lazy* al primer `Generate` para evitar tiempos largos de arranque.  
   - Se registran las capas personalizadas `TimestepEmbedding` y `SplitScaleShift`; por compatibilidad se carga con `safe_mode=False` y `compile=False`.
2. **Reverse diffusion:**  
   - AD3 usa schedule lineal de 300 pasos, replicando `AD3.ipynb`.  
   - AD6 usa el cosine schedule desde `schedule_ad6.npz` y aplica classifier‑free guidance controlado por `guidance_scale`.
3. **Frames intermedios:** se almacenan cada 10 pasos y se muestran en galerías separadas.
4. **Visualización pixel‑perfect:** salida 16×16 se reescala a 256×256 con `Image.NEAREST` para mantener bordes nítidos.

### 3.2 Eficiencia

Decisiones que mantienen el despliegue eficiente:
- **Schedules precomputados** y en memoria (`betas/alphas` como tensores constantes).  
- **Modelos cacheados** globalmente: no se recargan por request.  
- **Carga diferida de AD6:** evita bloquear el servidor al inicio.  
- **Inferencia en `training=False`** y sin compilación para reducir overhead.

En CPU, una generación completa a 300–400 pasos puede tardar varios segundos; en GPU, el tiempo cae típicamente a sub‑segundos o pocos segundos según hardware.

### 3.3 Escalabilidad

Para escalar más allá del demo local:
- **Batching:** la lógica de muestreo soporta `num_samples>1` con cambios mínimos.  
- **Cola de requests:** Gradio puede habilitarse con `demo.queue()` para manejar concurrencia sin saturar VRAM/CPU.  
- **Separación UI/servicio:** mover el muestreo a una API permite escalar horizontalmente (réplicas con GPU).  
- **Versionado y rollback:** los loaders admiten múltiples nombres/versiones; en producción se recomienda versionar por carpeta/fecha.

## 4. Instrucciones de instalación y ejecución

1. Activar el entorno:
   - Windows: `.\env\Scripts\activate`
   - Unix: `source env/bin/activate`
2. (Si hace falta) instalar dependencias: `pip install -r requirements.txt`.
3. Ejecutar desde la raíz del repo:
   - `python app.py`
   - o en Windows: `.\env\Scripts\python.exe app.py`
4. Abrir `http://127.0.0.1:7860` en el navegador.
5. Elegir clase, pasos, ruido, guidance (AD6) y seed; presionar **Generate**.

## 5. Costos estimados de infraestructura

Los costos dependen del modo de despliegue:

### 5.1 Demo local
- **Costo monetario:** 0 (se usa equipo local).  
- **Costo computacional:** tiempo de CPU/GPU por generación.  
- **Memoria:**  
  - AD3 ~2.1 MB en disco; AD6 EMA ~17.5 MB.  
  - RAM/VRAM requerida: ~1–2 GB para inferencia cómoda.

### 5.2 Producción en nube (referencia)

Para un servicio con baja latencia:
- **GPU básica (p. ej. T4/L4 o equivalente):** suele ser suficiente para 16×16.  
  - Orden de magnitud: **USD 0.20–0.80/hora** según proveedor/región.  
- **CPU-only:** posible, pero con latencias más altas (menos recomendable para usuarios interactivos).
- **Almacenamiento:** modelos + schedules < 50 MB → costo despreciable en un bucket estándar.

Estos valores son aproximados y deben ajustarse al proveedor real (GCP/AWS/Azure/Kaggle/Colab Pro).

## 6. Operación y mantenimiento

Para mantener el despliegue en producción:
- **Monitoreo:** latencia por request, uso de VRAM/CPU, tasa de errores al cargar modelos.
- **Actualización de modelos:** guardar nuevas versiones en `data/models/`/bucket con nombres versionados; actualizar loaders o symlinks para rollout controlado.
- **Reentrenamiento:** AD6 se reentrena con `scripts/training/AD6.py` (sin Lambdas inseguras).  
- **Compatibilidad de entorno:** fijar versiones en `requirements.txt` y/o imagen Docker; evitar cambios de TensorFlow sin prueba.
- **Seguridad básica:** si se expone públicamente, añadir autenticación simple o limitar por red; no aceptar paths externos ni uploads sin validación.

## 7. Vista comparativa (AD3 vs AD6)

La siguiente animación muestra el denoising de ambos modelos con el mismo seed/clase. AD3 (izquierda) tiende a producir bordes más suaves y más grano; AD6 (derecha) converge más rápido y conserva contornos más nítidos, lo que es deseable para pixel art.

![AD3 vs AD6](../../reports/deployment/AD3%20vs%20AD6.gif)
