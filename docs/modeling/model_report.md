# Reporte del Modelo Final

## Resumen Ejecutivo

Este proyecto tuvo como objetivo comparar diferentes arquitecturas generativas —Autoencoders, DDPM en pixel space, DDPM condicional, Latent Diffusion y un experimento de ablation— para identificar cuál modelo produce imágenes sintéticas con la mayor fidelidad perceptual respecto a la distribución original.

Los resultados experimentales muestran que el DDPM Condicional entregó el mejor desempeño global, obteniendo un FID-like promedio cercano a 19.2, superando al DDPM no condicional (21.85) y mostrando estabilidad entre clases. Este comportamiento confirma que incorporar información semántica (class_id) durante el proceso de denoising mejora la calidad visual y la convergencia del modelo.

El DDPM Condicional se selecciona como modelo final, tanto por su desempeño cuantitativo como por su coherencia teórica y su capacidad demostrada para capturar características estructurales de las distintas clases del dataset.

## Descripción del Problema

El problema central consiste en generar imágenes sintéticas de alta calidad que respeten la estructura estadística del conjunto de datos original. Esto requiere modelos capaces de aproximar la distribución de probabilidad de las imágenes, manteniendo coherencia visual, diversidad y fidelidad al dominio.

### Contexto

La generación de imágenes mediante modelos de difusión se ha convertido en el paradigma dominante en inteligencia artificial, desplazando a modelos como GANs en estabilidad y calidad. Estos modelos aplican un proceso progresivo de adición de ruido y un proceso inverso entrenado para revertirlo, generando nuevas imágenes desde ruido puro.

El proyecto se desarrolla dentro de ese marco, evaluando:

diferencias entre reconstrucción (Autoencoders) y generación (DDPMs),

impacto de entrenar en pixel space vs latent space,

rol del conditioning para guiar el proceso de generación,

efecto del número de pasos de ruido (T),

comportamiento del loss durante diferentes configuraciones arquitectónicas.

Objetivos específicos

Desarrollar y entrenar múltiples modelos generativos.

Evaluar su desempeño mediante métricas perceptuales.

Comparar cuantitativa y cualitativamente los resultados.

Seleccionar el mejor modelo según evidencia experimental.

Documentar de manera interpretativa y técnica los hallazgos.

### Justificación

Generar imágenes sintéticas de calidad requiere capturar dependencias complejas entre pixeles y patrones de clase. Elegir el modelo adecuado permite no solo producir mejores resultados, sino también entender los mecanismos que hacen que ciertos modelos generalicen mejor. Esto es clave para aplicaciones futuras como generación condicionada, simulación de datos o estudios de ablation para optimización computacional.

## Descripción del Modelo

El modelo seleccionado fue el DDPM Condicional (Exp3). Se trata de un Denoising Diffusion Probabilístico Model que incorpora un embedding de clase que modula el proceso de denoising a lo largo de toda la arquitectura.

Metodología del modelo

El modelo sigue dos procesos fundamentales:

Difusión hacia adelante (q)
Se añade ruido gaussiano progresivamente durante T = 300 pasos hasta aproximar una distribución normal estándar.

Proceso inverso (p)
Una red neuronal U-Net es entrenada para predecir el ruido añadido en cada paso. Su precisión determina qué tan bien puede reconstruir una imagen desde ruido puro.

Condicionamiento

El condicionamiento se realiza mediante un embedding de clase, que se injerta en diferentes niveles del U-Net, guiando la predicción del ruido. Esto permite que el modelo genere imágenes específicas de cada categoría de manera más precisa que un DDPM no condicional.

Hiperparámetros principales

T = 300

learning_rate = 0.0002

batch_size = 256

epochs = 50

conditioning = class_id

Arquitectura

Backbone: U-Net ligera pero eficiente para datasets de baja resolución.

Mecanismo de conditioning: modulación tipo FiLM o concatenación en bloques residuales.

Objetivo de entrenamiento: error en la predicción del ruido (MSE).

## Evaluación del Modelo

Se evaluaron cinco experimentos principales:

Autoencoder / Denoising Autoencoder (Exp1)

DDPM en Pixel Space (Exp2)

DDPM Condicional (Exp3)

Latent Diffusion (Exp4)

DDPM Ablation T=50 (Exp5)

Resultados cuantitativos

Exp1 – AE/DAE

SSIM: 0.949

PSNR: 24.82

MSE: 0.0043

Interpretación: excelente reconstrucción, pero no genera nuevas imágenes.

Exp2 – DDPM PixelSpace

FID-like: 21.85

train_loss: 0.088

Interpretación: generador estable y de buena calidad, pero todavía superado por modelos condicionados.

Exp3 – DDPM Condicional (modelo final)
FID-like por clase:

clase 0 → 19.12

clase 1 → 16.89

clase 2 → 17.39

clase 3 → 20.83

clase 4 → 21.93

Promedio ≈ 19.2

Interpretación:
Es el mejor resultado entre todos los experimentos. La ganancia en FID-like demuestra que el conditioning permite capturar mejor la estructura intra-clase y estabiliza el proceso de denoising.

Exp4 – Latent Diffusion

Loss: 0.6203

Interpretación:
El loss es más alto porque se entrena en un espacio latente más complejo. No hay métricas comparables, por lo que no puede considerarse ganador.

Exp5 – Ablation T=50

train_loss: 0.222 (epoch 6/15)

Interpretación:
Reducir T acelera el modelo, pero sacrifica calidad. A mitad de entrenamiento ya es evidente que no superará al DDPM estándar.

Conclusión de la evaluación

El DDPM Condicional supera al resto de modelos en:

calidad perceptual,

control sobre la clase generada,

estabilidad del entrenamiento,

métricas FID-like por clase.

## Conclusiones y Recomendaciones

Conclusiones principales

El DDPM Condicional fue el mejor modelo, con el FID-like más bajo y la mejor coherencia entre clases.

Incorporar conditioning mejora notablemente el modelado de la distribución.

Los autoencoders son buenos reconstructores, pero no compiten como generadores.

El DDPM estándar es un baseline sólido pero inferior.

Latent Diffusion requiere evaluación adicional para ser comparable.

Puntos fuertes del modelo final

Calidad perceptual superior.

Mejor capacidad de generación específica por clase.

Estabilidad durante el entrenamiento.

Limitaciones

Mayor costo computacional.

El conditioning por clase no escala bien a datasets grandes sin embeddings más sofisticados.

Se requiere evaluación visual más amplia para asegurar diversidad y ausencia de colapso.

Recomendaciones

Implementar Classifier-Free Guidance para mejorar el control semántico.

Entrenar un U-Net más profundo para mejorar el FID-like.

Replicar el DDPM Condicional en Latent Space para comparar desempeño y eficiencia.

Evaluar con FID real (no reducido a PCA) si se dispone de recursos.

## Referencias

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.

Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis.

Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

Documentación complementaria y repositorios utilizados en el proyecto.
