# PixelGen — Diffusion para Pixel Art (AD3 vs AD6)

PixelGen es un proyecto end-to-end para generar sprites de **pixel art (16×16 RGB)** usando modelos de difusión (DDPM). Incluye adquisición y transformación de datos, EDA reproducible, entrenamiento de modelos, evaluación cuantitativa y un demo de despliegue con Gradio para comparar **AD3 vs AD6** lado a lado.

![Demo AD3 vs AD6](<reports/deployment/AD3 vs AD6.gif>)

---

## Qué hay en este repositorio

- **Datos**: `data/raw/` (artefactos originales) y `data/intermediate/pixel_art_data.npz` (formato normalizado listo para modelado).
- **EDA reproducible**: `scripts/eda/run_eda.py` + resultados en `reports/eda/eda.json` y `reports/figures/eda/`.
- **Modelos**:
  - **AD3**: DDPM condicional (baseline fuerte).
  - **AD6**: ResUNet + FiLM + cosine schedule + EMA + classifier-free guidance (mejora sobre AD3).
- **Evaluación**: `scripts/evaluation/compare_models.py` + resultados en `reports/evaluation/`.
- **Despliegue**: `app.py` (Gradio) para generación y visualización del denoising.

---

## Resultados clave

- **Calidad de datos**: sin faltantes/corrupción en las validaciones; alta redundancia (duplicados exactos) en raw.
- **EDA**: separabilidad por clase fuerte en términos no lineales (clasificador auxiliar con 100% accuracy).
- **Modelos**: AD6 mejora consistentemente frente a AD3 en una métrica proxy Feature-FID (PCA); ver `docs/modeling/model_report.md`.

---

## Quickstart

### 1) Crear entorno e instalar dependencias

```bash
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

### 2) Obtener datos (raw)

```bash
.\env\Scripts\python.exe scripts/data_acquisition/download_raw_data.py
```


### 3) Generar dataset intermedio (recomendado para modelado)

```bash
.\env\Scripts\python.exe scripts/preprocessing/raw_to_npz.py
```

### 4) Ejecutar EDA

```bash
.\env\Scripts\python.exe scripts/eda/run_eda.py
```

Salida principal:

- `reports/eda/eda.json`
- `reports/figures/eda/`

---

## Entrenamiento y evaluación

### Entrenar AD6 (opcional)

```bash
.\env\Scripts\python.exe scripts/training/AD6.py
```

Artefactos esperados:

- `artifacts_exp6/ddpm_resunet_ad6.keras`
- `artifacts_exp6/ddpm_resunet_ad6_ema.keras`
- `artifacts_exp6/schedule_ad6.npz`

Para el demo, estos artefactos deben estar en `data/models/` (ver `docs/deployment/deploymentdoc.md`).

### Comparar AD3 vs AD6 (métrica proxy)

```bash
.\env\Scripts\python.exe scripts/evaluation/compare_models.py
```

Salida:

- `reports/evaluation/compare_ad3_ad6.json`

Resumen (ver `docs/modeling/model_report.md`): AD6 mejora consistentemente frente a AD3 en Feature-FID (proxy) basado en PCA.

---

## Demo de despliegue (Gradio)

La app permite generar sprites y comparar el denoising de **AD3 (izquierda)** vs **AD6 (derecha)**, incluyendo galerías de frames intermedios con escalado *pixel-perfect* (nearest-neighbor).

Requisitos:

- Modelos en `data/models/` (AD3, AD6 EMA y schedule AD6).

Ejecutar:

```bash
.\env\Scripts\python.exe app.py
```

Abrir: `http://127.0.0.1:7860`

---

## Documentación (fuente de verdad)

- Negocio: [`docs/business_understanding/project_charter.md`](docs/business_understanding/project_charter.md)
- Datos:
  - [`docs/data/data_definition.md`](docs/data/data_definition.md)
  - [`docs/data/data_dictionary.md`](docs/data/data_dictionary.md)
  - [`docs/data/data_summary.md`](docs/data/data_summary.md)
- Modelamiento:
  - [`docs/modeling/baseline_models.md`](docs/modeling/baseline_models.md)
  - [`docs/modeling/model_report.md`](docs/modeling/model_report.md)
- Despliegue: [`docs/deployment/deploymentdoc.md`](docs/deployment/deploymentdoc.md)
- Cierre: [`docs/acceptance/exit_report.md`](docs/acceptance/exit_report.md)

---

## Estructura (resumen)

```
data/
  raw/
  intermediate/
  models/
scripts/
  data_acquisition/
  preprocessing/
  eda/
  training/
  evaluation/
reports/
  eda/
  figures/
  evaluation/
  deployment/
docs/
```

---

## Equipo

- David Paloma
- Juan Ayala
- Daniel Gracia
