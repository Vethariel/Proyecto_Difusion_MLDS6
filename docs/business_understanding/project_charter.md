# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

PixelGen: Sistema Generativo de Pixel Art basado en Modelos de Difusión

## Objetivo del Proyecto

Desarrollar un sistema generativo capaz de producir imágenes tipo pixel art, entrenado a partir de un dataset real de 89.000 sprites en 16×16 píxeles. El objetivo es crear un modelo de difusión reproducible, evaluable y desplegable que permita generar nuevos assets visuales para entornos de videojuegos, prototipado visual y diseño retro. El proyecto es importante porque demuestra la capacidad del equipo para diseñar modelos de última generación bajo estándares profesionales de ciencia de datos, versionado, documentación y despliegue.

## Alcance del Proyecto

### Incluye:

- Uso del dataset Pixel Art (Kaggle) con 89,000 imágenes JPEG de 16×16 pixeles y sus respectivas etiquetas.
- Procesos de limpieza, análisis exploratorio, normalización de imágenes y construcción del pipeline de datos.
- Entrenamiento de un modelo de difusión (DDPM o UNet-based) y comparación con al menos un baseline generativo (VAE o GAN).
- Evaluación del modelo con métricas como FID/KID y panel comparativo de muestras generadas.
- Despliegue de un demo interactivo mediante Gradio o similar.
- Documentación técnica, versionado del dataset y del modelo, y presentación ejecutiva.
- Video final explicando la solución y su aplicación empresarial.

### Excluye:

- Entrenamiento de modelos para tareas que no sean generación (clasificación, segmentación, detección).
- Uso de datasets adicionales no autorizados o con restricciones de licencia.
- Creación de un producto comercial final o integración en pipelines de empresas reales.
- Entrenamiento de modelos de difusión a gran escala (imagen > 64×64) que requieran HPC o infraestructura especializada.

## Metodología

El proyecto se desarrollará bajo un enfoque de ciencia de datos aplicada, siguiendo las etapas tradicionales del ciclo de vida de un proyecto de machine learning:

1. Comprensión del negocio: análisis del problema, validación del dataset y definición del valor empresarial.
2. Preprocesamiento y EDA: limpieza, visualización, análisis del estilo y estructura de los datos.
3. Modelamiento: entrenamiento de un modelo baseline y posterior entrenamiento del modelo de difusión, incluyendo selección de hiperparámetros y técnicas de estabilización.
4. Despliegue: implementación de interfaz interactiva, creación de scripts y documentación del pipeline.
5. Evaluación: análisis de métricas, comparación de modelos y entrega final.

Se utilizarán herramientas como Python, PyTorch, Diffusers, MLflow (opcional), Git/GitHub y Gradio.

## Cronograma

| Etapa | Duración Estimada | Fechas |
|------|---------|-------|
| Entendimiento del negocio y carga de datos | 1 semana | del 14 de noviembre al 20 de noviembre |
| Preprocesamiento, análisis exploratorio | 1 semana | del 21 de noviembre al 27 de noviembre |
| Modelamiento y extracción de características | 1 semana | del 28 de noviembre al 4 de diciembre |
| Despliegue | 1 semana | del 5 de diciembre al 11 de diciembre |
| Evaluación y entrega final | 1 semana | del 12 de diciembre al 15 de diciembre |

## Equipo del Proyecto

- David Paloma – Líder de proyecto
- Juan Ayala – Científico de datos / Modelamiento
- Daniel Gracia – Preprocesamiento y despliegue

## Presupuesto

El proyecto se realizará con recursos gratuitos y computacionales proporcionados por plataformas como Google Colab. No se requiere presupuesto monetario adicional. En caso de necesitar GPU premium, se contemplará un gasto opcional y reducido en créditos de Colab Pro o Kaggle.

## Stakeholders

Empresas de videojuegos

- Rol: potenciales usuarios del modelo generativo.
- Interés: conocer herramientas que aceleren la creación de assets y prototipos visuales.
- Expectativas: un modelo capaz de generar pixel art consistente, reproducible y útil para procesos creativos, exploración de estilos y desarrollo temprano de videojuegos.

Compañeros del diplomado

- Rol: comunidad académica que sigue el desarrollo de los diferentes proyectos.
- Interés: observar enfoques alternativos, comparar metodologías y aprender de la aplicación de modelos de difusión en un contexto práctico.
- Expectativas: claridad metodológica, documentación accesible, coherencia técnica y resultados visualmente demostrables que aporten a la discusión académica general.

## Aprobaciones

- Juan Sebastian Malagon – Instructor
- Firma: __________________________
- Fecha de aprobación: __________________
