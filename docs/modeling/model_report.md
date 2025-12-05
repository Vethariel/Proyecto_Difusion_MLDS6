# Reporte del Modelo Final

## Resumen Ejecutivo

- Objetivo: comparar Autoencoders, DDPM en pixel space, DDPM condicional, Latent Diffusion y una ablation (T=50) para elegir el generador con mayor fidelidad perceptual.
- Mejor modelo: DDPM Condicional (Exp3) con FID-like promedio ~19.2, estable entre clases y coherente con la teoria al usar la clase como guia de denoising.
- Decisiones clave: incorporar conditioning por class_id mejora la calidad visual y la convergencia; el DDPM estandar es un buen baseline pero queda detras al no usar informacion semantica.
- Uso final: el modelo seleccionado captura mejor la estructura por clase y se mantiene estable durante el entrenamiento, por lo que se adopta como version final.

## Planteamiento del Problema

Generar imagenes sinteticas de alta calidad requiere aproximar la distribucion de los datos originales, manteniendo coherencia visual, diversidad y consistencia por clase. El reto es elegir un modelo que pueda revertir un proceso de ruido progresivo y recuperar imagenes que se perciban realistas.

### Contexto

El trabajo se enmarca en modelos de difusion, hoy el paradigma dominante frente a GANs por estabilidad y calidad. Se evaluaron diferencias entre reconstruccion (Autoencoder) y generacion (DDPM), el impacto de entrenar en pixel space vs latent space, el rol del conditioning, el numero de pasos de ruido (T) y el comportamiento del loss bajo distintas arquitecturas.

## Modelo Seleccionado: DDPM Condicional (Exp3)

El modelo es un Denoising Diffusion Probabilistic Model con embedding de clase que modula todo el U-Net para guiar la prediccion de ruido.

- Flujo de difusion: T = 300 pasos de ruido gaussiano hasta aproximar una normal estandar.
- Proceso inverso: U-Net entrenada para predecir el ruido en cada paso; su precision determina que tan bien se reconstruye desde ruido puro.
- Conditioning: embedding de clase inyectado en varios niveles del U-Net (modulacion tipo FiLM o concatenacion residual) para generar imagenes especificas por categoria.
- Hiperparametros: learning_rate = 2e-4, batch_size = 256, epochs = 50, conditioning = class_id.
- Objetivo de entrenamiento: MSE sobre la prediccion de ruido.

## Evaluacion Comparativa

Se ejecutaron cinco experimentos principales:

- Exp1 — Autoencoder / Denoising Autoencoder  
  - SSIM: 0.949; PSNR: 24.82; MSE: 0.0043.  
  - Lectura: excelente reconstruccion, pero no genera nuevas imagenes.

- Exp2 — DDPM en Pixel Space  
  - FID-like: 21.85; train_loss: 0.088.  
  - Lectura: generador estable y de buena calidad, pero sin condicionamiento queda por detras.

- Exp3 — DDPM Condicional (modelo final)  
  - FID-like por clase: 0: 19.12, 1: 16.89, 2: 17.39, 3: 20.83, 4: 21.93. Promedio ~19.2.  
  - Lectura: mejor resultado global; el conditioning captura mejor la estructura intra-clase y estabiliza el denoising.

- Exp4 — Latent Diffusion  
  - Loss: 0.6203.  
  - Lectura: el espacio latente es mas complejo; sin metricas comparables no se puede declarar ganador.

- Exp5 — Ablation T=50  
  - train_loss: 0.222 (epoch 6/15).  
  - Lectura: reducir T acelera, pero sacrifica calidad; no supera al DDPM estandar.

### Hallazgos clave

- El conditioning por clase reduce FID-like y mejora la coherencia visual.
- El DDPM estandar es un baseline solido pero insuficiente sin señal semantica.
- Latent Diffusion requiere evaluacion adicional con metricas comparables.
- Acelerar con menos pasos (T=50) degrada perceptual notablemente.

## Conclusiones y Recomendaciones

- El DDPM Condicional ofrece la mejor calidad perceptual y estabilidad entre clases, por lo que se adopta como modelo final.
- Los autoencoders son utiles para reconstruccion, pero no compiten como generadores.
- El conditioning es el factor que marca la diferencia en fidelidad y control por clase.
- Limitaciones: mayor costo computacional y escalabilidad limitada del conditioning simple para datasets grandes.
- Recomendaciones proximas:
  1) Implementar Classifier-Free Guidance para mayor control semantico.  
  2) Probar un U-Net mas profundo para seguir mejorando FID-like.  
  3) Replicar el DDPM Condicional en latent space para comparar eficiencia.  
  4) Evaluar con FID completo (no reducido a PCA) si hay recursos de calculo.
