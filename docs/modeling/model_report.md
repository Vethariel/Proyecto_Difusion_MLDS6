# Reporte del modelo final (AD3 vs AD6)

## Resumen ejecutivo

El proyecto evaluó varias familias de modelos generativos para sprites 16×16 (autoencoder, DDPM no condicional, DDPM condicional, latent diffusion y una ablación). En la batería inicial, el mejor candidato fue el **DDPM condicional (AD3 / Exp3)** por coherencia por clase y estabilidad durante entrenamiento.

Como extensión, se construyó **AD6**: una evolución directa de AD3 inspirada en el estilo de U‑Net usado en el notebook del autor del dataset (referencia: https://www.kaggle.com/code/ebrahimelgazar/diffusion-model-u-net) y en mejoras comunes en difusión (bloques residuales, schedule cosine, EMA y classifier‑free guidance). La comparación objetiva (Feature‑FID basado en PCA) muestra una mejora consistente:

- **AD3 promedio:** 153.92  
- **AD6 promedio:** 38.26  

(fuente: `reports/evaluation/compare_ad3_ad6.json`, menor es mejor).

## Planteamiento del problema

Generar imágenes sintéticas de pixel art requiere aproximar la distribución real de sprites manteniendo:

- Fidelidad (bordes/colores coherentes con el dominio).
- Diversidad (variaciones no triviales).
- Control por clase (coherencia intra‑clase sin mezclar estilos).

En modelos de difusión, esto se traduce en entrenar una red (U‑Net) a predecir el ruido agregado en distintos pasos y usar esa predicción para revertir el proceso desde ruido hasta imagen.

## Modelos evaluados (AD1–AD5)

Resumen (ver detalle en `docs/modeling/baseline_models.md`):

- **Exp1 (Autoencoder/DAE):** excelente reconstrucción, pero no genera muestras nuevas.
- **Exp2 (DDPM pixel‑space no condicional):** baseline generativo estable, pero sin control por clase.
- **Exp3 (AD3: DDPM condicional):** mejor dentro del set inicial; el conditioning por clase mejora coherencia y reduce el error perceptual proxy.
- **Exp4 (Latent diffusion “lite”):** prometedor en eficiencia, pero dependiente de la calidad del autoencoder y sin métricas comparables completas.
- **Exp5 (Ablación):** estudiar T/capacidad confirma el trade‑off calidad vs costo (menos pasos suele degradar nitidez).

## AD3 (Exp3): DDPM condicional base

AD3 es un DDPM condicional en pixel‑space:

- **Schedule:** lineal, **T=300** (betas 1e‑4 → 2e‑2).
- **Modelo:** U‑Net pequeña con convoluciones y ReLU.
- **Condicionamiento:** embedding de `class_id` fusionado con embedding de tiempo e inyectado como sesgo (modulación simple).
- **Objetivo:** MSE entre ruido real y ruido predicho.

Fortalezas: control por clase y estabilidad con pocos componentes.  
Limitación: arquitectura simple (sin residual blocks/normalización) tiende a producir bordes más suaves o “grano” en clases difíciles.

## AD6: mejora sobre AD3 (enfoque didáctico y práctico)

AD6 nace como “AD3 mejorado”: no cambia la formulación DDPM, sino la capacidad del modelo y la calidad del muestreo. Se implementó para acercarse a patrones de referencia (incluyendo la idea general del U‑Net del autor del dataset) y para enseñar por qué ciertas técnicas mejoran el denoising.

### Principales mejoras (AD6 vs AD3)

1) **U‑Net residual (más capacidad, mejor optimización)**
- AD3: Conv+ReLU sin residual blocks ni normalización.
- AD6: bloques residuales + normalización + activación SiLU y upsampling aprendido (ConvTranspose).

Por qué ayuda: los residual blocks facilitan el flujo de gradiente y permiten redes más expresivas sin colapsar; la normalización estabiliza escalas internas y mejora la predicción de ruido.

2) **Condicionamiento más fuerte (FiLM‑style)**
- AD3: inyección aditiva simple.
- AD6: modulación tipo FiLM (scale/shift) derivada de embeddings de tiempo y clase.

Por qué ayuda: la clase controla directamente activaciones internas del denoiser, reduciendo mezclas entre estilos y elevando coherencia por clase.

3) **Schedule cosine (mejor distribución señal/ruido)**
- AD3: betas lineales, T=300.
- AD6: cosine schedule, T=400 (`schedule_ad6.npz`).

Por qué ayuda: en práctica, la progresión de ruido suele ser más “natural”, mejorando estabilidad del denoising en etapas tempranas y finales.

4) **EMA (Exponential Moving Average)**
- AD3: muestreo con pesos finales.
- AD6: checkpoint EMA para muestreo.

Por qué ayuda: EMA suaviza fluctuaciones del entrenamiento y tiende a producir muestras más limpias y consistentes.

5) **Classifier‑Free Guidance (CFG)**
- AD3: solo condicional.
- AD6: entrenamiento con token nulo (dropout de condicion) y muestreo combinando condicional/no‑condicional con `guidance_scale`.

Por qué ayuda: permite aumentar fidelidad a la clase sin un clasificador extra; el guidance actúa como “perilla” de control (más guidance → más fidelidad, potencialmente menos diversidad).

Nota técnica: AD6 reemplaza Lambdas problemáticas por una capa determinística (`SplitScaleShift`) para que el modelo sea más robusto al cargarse en producción.

## Comparación objetiva AD3 vs AD6 (Feature‑FID PCA)

Para hacer la comparación más objetiva se usó `scripts/evaluation/compare_models.py`, que calcula un proxy de FID en un espacio de características reducido por PCA. Interpretación:

- **Menor puntaje = mejor** (más cercanía entre distribuciones real vs generada).
- Es una **métrica proxy**: útil para comparación rápida y consistente entre variantes, pero no reemplaza FID “completo”.

Resultados (fuente: `reports/evaluation/compare_ad3_ad6.json`):

| Clase | AD3 | AD6 |
|------:|----:|----:|
| 0 | 39.72 | 13.36 |
| 1 | 167.41 | 32.37 |
| 2 | 77.96 | 23.74 |
| 3 | 311.38 | 79.45 |
| 4 | 173.12 | 42.37 |
| **Promedio** | **153.92** | **38.26** |

Lectura de los resultados:

- AD6 mejora de forma consistente en todas las clases (no es una ganancia “accidental” en una sola).
- La brecha es mayor en clases difíciles (ej. clase 3), lo que sugiere que mayor capacidad + guidance ayudan a sostener estructura cuando el denoising requiere más “correcciones”.
- Esta mejora cuantitativa coincide con lo que se observa en la app de despliegue (`app.py`): AD6 tiende a producir contornos más nítidos y una convergencia más limpia en los frames.

## Conclusiones y recomendaciones

- AD3 fue el mejor modelo dentro de la batería inicial y sigue siendo un baseline sólido y simple para explicar DDPM condicional.
- AD6 demuestra que, manteniendo el mismo marco DDPM, mejoras de arquitectura (ResUNet + FiLM), schedule (cosine) y muestreo (EMA + CFG) pueden mejorar significativamente la fidelidad (proxy) y la calidad visual.

Siguientes pasos recomendados:

1) Estandarizar un protocolo de comparación (mismo número de pasos efectivos, mismo `noise_level` y reportar latencia CPU/GPU).
2) Barrer `guidance_scale` y reportar el trade‑off fidelidad/diversidad.
3) Si hay recursos, calcular FID completo o añadir una métrica perceptual adicional (por ejemplo LPIPS) para complementar PCA‑FID.
4) Probar una variante “AD6‑lite” (menos canales o menos pasos) para un punto de equilibrio calidad/tiempo de inferencia en despliegue.
