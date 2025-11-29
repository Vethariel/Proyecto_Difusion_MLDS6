# Pixel Art Diffusion Project

Este repositorio contiene el pipeline completo para construir un modelo generativo basado en *diffusion*, utilizando un dataset de 89.400 imágenes de pixel art (16×16×3). El proyecto implementa un ciclo profesional de ciencia de datos con:

- **DVC** para manejo de datos, caché y versiones del pipeline.
- **EDA profunda** (PCA, t-SNE, análisis cromático, separabilidad de clases, CNN auxiliar).
- **Scripts modulares** para lectura, limpieza, procesado y análisis.
- **Estructura robusta de carpetas** siguiendo el estándar de proyectos ML reproducibles.

---

## 🚀 Objetivo del Proyecto

Construir un *modelo de difusión* capaz de generar pixel art coherente, limpio y controlable por clase.  
El dataset original contiene ruido, duplicados y variaciones estilísticas; por eso se diseñó un pipeline de EDA + procesamiento que permite:

- Detectar duplicados y quedarse con imágenes únicas.  
- Limpieza y normalización del dataset.
- Evaluación de separabilidad real entre clases.
- Exploración de la estructura latente del dominio visual.

Todas las fases están versionadas con **DVC** para garantizar reproducibilidad y trazabilidad.

---

## 📂 Estructura del Proyecto

```
.
├── data/
│   ├── raw/               # Datos originales sin procesar
│   ├── intermediate/      # Resultados generados por scripts (versionados con DVC)
│   └── processed/         # Conjunto final para entrenamiento de la difusión
│
├── scripts/
│   ├── eda/               # Análisis exploratorio modular (3.x, 5.x, 6.x)
│   ├── processing/        # Limpieza, normalización, hashing, uniques
│   └── run_eda.py         # Orquestador unificado del EDA
│
├── reports/
│   ├── figures/           # Gráficas generadas por todos los análisis
│   └── eda/               # Archivos de texto y JSON con resultados
│
├── docs/
│   ├── data_summary.md    # Reporte completo del EDA
│   └── methodology.md     # Diseño metodológico del proyecto
│
├── dvc.yaml               # Pipeline declarativo
├── dvc.lock               # Trazabilidad exacta del experimento
├── README.md              # Documento actual
└── requirements.txt
```

---

## 🧠 Scripts Clave

### `scripts/run_eda.py`
Orquestador general del EDA.  
Ejecuta:

- 3.1 – Variable objetivo  
- 3.2 – Distribución de imágenes  
- 3.3 – Variabilidad intra-clase  
- 3.4 – Variabilidad global  
- 5.1 – PCA  
- 5.2 – Importancia del color  
- 5.3 – Separabilidad entre clases  
- 6.3 – CNN auxiliar

Los resultados se guardan en:

```
reports/eda/eda.json
reports/figures/eda/
```

---

## 📦 Uso del Proyecto

### 1. Clonar el repo
```
git clone https://github.com/usuario/pixel-art-diffusion.git
cd pixel-art-diffusion
```

### 2. Instalar dependencias
```
pip install -r requirements.txt
```

### 3. Descargar los datos con DVC
```
dvc pull
```

### 4. Ejecutar el EDA completo
```
python scripts/run_eda.py
```

### 5. Regenerar datos procesados
```
dvc repro
```

---

## 📊 Resultados principales

- El dataset tiene **altísima redundancia**, reduciendo ~89.400 → 1.665 imágenes únicas.
- La distribución de intensidad se mantiene entre dataset completo y único.
- PCA revela que **20–30 componentes** capturan la mayor parte de la estructura.
- El canal azul **B** es el eje cromático dominante.
- t-SNE y metrics no supervisadas muestran baja separabilidad lineal.
- Una **CNN auxiliar logra 100% accuracy**, evidenciando separabilidad profunda no lineal.

Todos los gráficos están disponibles en:

```
reports/figures/eda/
```

---

## ☁️ DVC y flujo de datos

El pipeline controla:

- Descarga de imágenes crudas.
- Limpieza y hashing.
- Generación de dataset procesado `.npz`.
- EDA completo con sus salidas.

Modificar cualquier script hace que DVC regenere automáticamente la etapa afectada.

Esto garantiza:

- **Reproducibilidad**
- **Trazabilidad**
- **Versionado de datasets y gráficas**
- **Ejecución consistente entre integrantes del equipo**

---

## 🤝 Equipo

Proyecto desarrollado por:

- **David Paloma**
- **Juan Ayala**
- **Daniel Gracia**

Bajo la metodología TDSP aplicada al desarrollo de modelos generativos.

---

## 📌 Estado del Proyecto

✔️ EDA finalizado  
✔️ Pipeline limpio  
✔️ Dataset procesado  
⬜ Entrenamiento del modelo de difusión  
⬜ Evaluación y benchmarks  
⬜ Generación de experimentos condicionados

---

## 🔮 Próximos pasos

1. Construcción de la U-Net para difusión.  
2. Entrenamiento con conditioning por clase.  
3. Evaluación de FID, IS y métricas perceptuales.  
4. Implementación de GUI minimal para generar sprites.

---

## 📄 Licencia

MIT — uso libre para investigación y desarrollo.
