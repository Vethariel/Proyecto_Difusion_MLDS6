# utils/update_eda.py

import json
from pathlib import Path

EDA_JSON_PATH = Path("reports/eda/eda.json")

def update_eda_json(section_name: str, section_data: dict):
    """
    Actualiza o crea eda.json agregando o sobrescribiendo una sección.

    Parameters
    ----------
    section_name : str
        Nombre de la sección (ej: "pca_analysis", "color_importance").
    section_data : dict
        Datos que se guardarán bajo esa sección.
    """

    # Crear carpeta si no existe
    EDA_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Base
    if EDA_JSON_PATH.exists():
        with open(EDA_JSON_PATH, "r") as f:
            try:
                eda = json.load(f)
            except json.JSONDecodeError:
                eda = {}
    else:
        eda = {}

    # Actualizar sección
    eda[section_name] = section_data

    # Guardar
    with open(EDA_JSON_PATH, "w") as f:
        json.dump(eda, f, indent=4)

    print(f"[EDA JSON] Sección '{section_name}' actualizada en {EDA_JSON_PATH}")
