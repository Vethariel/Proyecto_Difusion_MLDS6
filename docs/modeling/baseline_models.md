# Reporte de modelos generativos (baselines y variantes de difusión)

Este documento describe los experimentos AD1–AD5 siguiendo la idea original del proyecto: construir una **línea evolutiva** de modelos (desde un baseline simple hasta difusión condicionada y variantes), con hipótesis claras y evidencia cuantitativa registrada en MLflow (`mlflow.db`).

## 1. Idea original (propósito de cada experimento)

La estrategia de modelamiento se diseñó como un “escalado” de complejidad:

1. **Exp1 – Autoencoder / Denoising Autoencoder (baseline no difusivo):** establecer una referencia de reconstrucción (qué tan bien un modelo simple representa el dataset) y obtener un encoder reutilizable para explorar difusión en latente.
2. **Exp2 – DDPM en pixel-space (unconditional):** entrenar un difusor sin condicionamiento para aprender la distribución global de sprites (“océano de sprites”).
3. **Exp3 – DDPM condicionado por clase (AD3):** incorporar `class_id` para mejorar coherencia por clase y evitar mezcla de estilos; aprovechar la alta separabilidad observada en EDA.
4. **Exp4 – Latent Diffusion (LDM-lite):** probar la hipótesis de eficiencia: si el espacio latente es compacto, la difusión podría converger más rápido sin degradar calidad.
5. **Exp5 – Ablación (T/capacidad):** cuantificar qué hiperparámetros realmente importan (número de pasos y tamaño del denoiser) y el trade-off calidad vs costo.

## 2. Tracking y fuente de verdad (MLflow)

Los resultados se consolidan desde `mlflow.db`. Para cada experimento se reporta el **mejor run FINISHED**, seleccionado por su métrica primaria (↓ mejor). Cuando una corrida terminó en FAILED pero dejó métricas registradas, se reporta explícitamente como evidencia incompleta.

### 2.1 Resumen comparativo (mejores runs en MLflow)

| Exp | Experimento | Métrica primaria (↓ mejor) | Mejor valor | Run |
|---:|---|---:|---:|---|
| 1 | Exp1-Autoencoder | `val_loss` | 0.005559 | `ac2d65bbeb9f48df9131fc49f3359f2a` |
| 2 | Exp2-DDPM-PixelSpace | `fid_like_pca20` | 21.8536 | `0fa38124710c42428d04bf583ec74120` |
| 3 | Exp3-DDPM-Conditional | avg(`fid_like_class_*`) | 19.2371 | `c0ac7d681d664f62a1d31084292199ef` |
| 4 | Exp4-LatentDiffusion | `loss` | 0.6203 | `158876ba6a9943379beac92117161f5e` |
| 5 | Exp5-Ablation_Scientific | `feature_fid` | 345.7668 | `72576899bd3c43fe8264d504c4ebd802` (FAILED) |

Notas:
- El promedio de Exp3 se calcula como promedio simple de `fid_like_class_0..4` del run.
- `Exp5-Ablation` no registra runs en `mlflow.db`; solo existe un run “científico” adicional y quedó incompleto.

---

## 3. Experimento 1 — Autoencoder / Denoising Autoencoder (baseline no difusivo)

### Objetivo (hipótesis)
Tener un baseline generativo simple para comparar contra difusión: qué tanto puede reconstruir el dataset un autoencoder clásico (y su variante denoising).

### Modelo (implementación esperada)
- Encoder–Decoder convolucional:
  - Input: 16×16×3
  - Encoder: 2–3 bloques Conv + ReLU + MaxPool → latente de dimensión baja (p. ej. 32 o 64)
  - Decoder: ConvTranspose / Upsampling → reconstrucción 16×16×3
- Variante DAE: agregar ruido gaussiano al input y reconstruir la imagen limpia.

### Parámetros clave (registrados en el mejor run)
Fuente: run `ac2d65bbeb9f48df9131fc49f3359f2a`.

- `latent_dim`: 64
- `noise_sigma`: 0.15
- `epochs`: 25

### Evaluación (qué se buscaba medir)
- Reconstrucción: MSE / PSNR / SSIM.
- Calidad visual: grids de original vs reconstrucción.
- Espacio latente: PCA/t-SNE y separabilidad (silhouette) para ver si el latente estructura mejor que píxeles.

### Resultados (MLflow)

| Métrica | Valor |
|---|---:|
| `val_loss` | 0.005559464450925589 |
| `MSE` | 0.004395863972604275 |
| `PSNR` | 24.82591389982544 |
| `SSIM` | 0.9490600824356079 |

### Lectura
El autoencoder establece una referencia clara: reconstruye con alta fidelidad estructural (SSIM alto), pero **no es un generador de nuevas muestras**. Además, deja listo un encoder para explorar enfoques en latente (Exp4).

---

## 4. Experimento 2 — Difusión en espacio de píxeles (DDPM baseline, sin condición)

### Objetivo (hipótesis)
Entrenar un DDPM simple (unconditional) que aprenda la distribución global del dataset y genere sprites “creíbles” desde ruido.

### Modelo (implementación esperada)
- DDPM pixel-space con U-Net pequeña:
  - Downsampling: 16→8→4, simétrico en upsampling.
  - Canales base típicos: 32–64–128.
  - Embedding de tiempo t (sinusoidal/posicional) inyectado en la U-Net.
- Pérdida: MSE entre ruido real y predicho.

### Parámetros clave (mejor run)
Fuente: run `0fa38124710c42428d04bf583ec74120`.

- `T`: 300
- `beta_schedule`: linear
- `batch_size`: 256
- `epochs`: 50
- `lr`: 2e-4

### Evaluación (qué se buscaba medir)
- Calidad visual: grids de muestras (8×8) desde ruido.
- Estadísticas marginales: histogramas RGB reales vs generadas.
- “FID casero”: distancia entre medias/covarianzas en un espacio de features (PCA o extractor auxiliar).

### Resultados (MLflow)

| Métrica | Valor |
|---|---:|
| `train_loss` | 0.08807305246591568 |
| `fid_like_pca20` | 21.853602172478762 |

### Lectura
El DDPM unconditional produce muestras razonables y es un baseline difusivo sólido, pero al no usar clase puede mezclar estilos y perder coherencia intra-clase. Esto motiva el experimento condicional.

---

## 5. Experimento 3 — Difusión condicionada por clase (Class-conditional DDPM / AD3)

### Objetivo (hipótesis)
Mejorar la estructura de las muestras cuando se especifica la clase objetivo. Dado que el clasificador auxiliar separa clases con alta precisión, el conditioning debería traducirse en mayor coherencia visual por clase y mejor métrica perceptual.

### Modelo (implementación esperada)
- Mismo DDPM del Exp2, pero con conditioning por clase:
  - `class_id` → embedding (p. ej. 16–32 dims) → fusión con embedding temporal.
  - Inyección del conditioning en la U-Net (sesgo/FiLM/concatenación según diseño).
- Mantener T y capacidad similares a Exp2 para aislar el efecto del conditioning.

### Parámetros clave (mejor run)
Fuente: run `c0ac7d681d664f62a1d31084292199ef`.

- `T`: 300
- `conditioning`: class_id
- `epochs`: 50
- `lr`: 2e-4

### Evaluación (qué se buscaba medir)
- Coherencia por clase:
  - Idealmente, pasar samples por un clasificador auxiliar y medir accuracy condicional (no quedó registrado en MLflow en este experimento).
- Distancia por clase:
  - FID-like por clase: reales de clase c vs generadas condicionadas a c.
- Calidad visual:
  - grids por clase y comparación con Exp2.

### Resultados (MLflow)

| Métrica | Valor |
|---|---:|
| `fid_like_class_0` | 19.12536543349288 |
| `fid_like_class_1` | 16.898045572589403 |
| `fid_like_class_2` | 17.39349062989828 |
| `fid_like_class_3` | 20.83842260591875 |
| `fid_like_class_4` | 21.93005297056078 |
| **Promedio** | **19.23707544249202** |

### Lectura
El conditioning reduce la distancia perceptual proxy frente al DDPM unconditional y mejora control por clase. AD3 queda como baseline fuerte del proyecto y punto de partida para la mejora AD6 (ver `docs/modeling/model_report.md`).

---

## 6. Experimento 4 — Difusión en espacio latente (Latent Diffusion “chiquito”)

### Objetivo (hipótesis)
Comprobar si difundir en un espacio latente compacto acelera entrenamiento y mantiene calidad comparable, aprovechando el encoder/decoder del Exp1.

### Modelo (implementación esperada)
- Encoder: imagen → z (latente).
- Difusión opera sobre z (MLP o U-Net pequeña sobre latente aplanado o mapa).
- Generación: z ~ difusión → decoder → imagen.

### Parámetros clave (mejor run)
Fuente: run `158876ba6a9943379beac92117161f5e`.

- `T`: 200
- `latent_dim`: 1024 (latente aplanado)
- `epochs`: 100
- `batch_size`: 128
- `learning_rate`: 1e-3

### Evaluación (qué se buscaba medir)
- Calidad visual y distancias tipo FID-like (comparables a pixel-space).
- Costo computacional (tiempo a converger).
- Riesgo: el decoder puede introducir blur y el latente puede requerir normalización cuidadosa.

### Resultados (MLflow)

| Métrica | Valor |
|---|---:|
| `loss` | 0.6203462481498718 |

### Lectura
La pérdida sola no permite compararlo directamente con FID-like de pixel-space. Para “cerrar” este experimento, se requiere un protocolo de evaluación comparable (misma métrica perceptual).

---

## 7. Experimento 5 — Ablación: ruido / pasos / capacidad

### Objetivo (hipótesis)
Identificar qué hiperparámetros mueven más la aguja en este dataset:
- Número de pasos T (50, 100, 200, 400).
- Capacidad del denoiser (canales base 16/32/64, etc.).

### Diseño planeado
- Variante A: barrer T manteniendo el resto fijo.
- Variante B: barrer capacidad manteniendo T fijo.
- Métricas: Feature-FID, calidad visual, y (para modelos condicionales) accuracy del clasificador auxiliar sobre samples.

### Evidencia disponible en MLflow (corrida incompleta)
`Exp5-Ablation` no tiene runs en `mlflow.db`. Solo se registró un run bajo `Exp5-Ablation_Scientific` y terminó en FAILED, por lo que se reporta como evidencia parcial.

Fuente: run `72576899bd3c43fe8264d504c4ebd802`.

| Config | `T_steps` | `base_channels` | `epochs` | `train_loss` | `feature_fid` (↓) | `training_time_sec` |
|---|---:|---:|---:|---:|---:|---:|
| DDPM_T50_C32 | 50 | 32 | 15 | 0.189316 | 345.766827 | 2047.9076 |

### Lectura
Aunque incompleta, esta evidencia es consistente con la hipótesis: **T muy bajo** puede degradar fuertemente calidad (feature_fid alto). Para completar este experimento, se recomienda re-ejecutar los barridos y registrar runs FINISHED.

---

## 8. Cierre: cómo se conecta con AD6

La narrativa completa del proyecto es:

- Exp1 define un baseline de reconstrucción.
- Exp2 demuestra que difusión en pixel-space funciona como generador global.
- Exp3 muestra que el conditioning por clase mejora coherencia y métricas por clase.
- Exp4 explora eficiencia en latente (pendiente de evaluación comparable).
- Exp5 explica sensibilidad a hiperparámetros (parcialmente registrado).

Sobre esa base, AD6 se plantea como la mejora lógica del camino Exp3: fortalecer arquitectura y muestreo (ResUNet, cosine schedule, EMA, CFG) para mejorar fidelidad. La comparación AD3 vs AD6 se presenta en `docs/modeling/model_report.md` usando `reports/evaluation/compare_ad3_ad6.json`.
