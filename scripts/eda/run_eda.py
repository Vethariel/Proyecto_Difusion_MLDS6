"""
run_eda.py
TDSP - EDA Orchestrator

Ejecuta los análisis EDA en un orden estándar y registra resultados en:
- figures: reports/figures/eda/<modulo>/
- json: reports/eda/eda.json
"""

import sys
from pathlib import Path

# Permite ejecutar/importar este script desde el root del repo u otras ubicaciones
# sin depender del CWD. (Necesario para resolver `modules.*`).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from modules.summary_general import run_summary_general
from modules.data_quality import run_data_quality
from modules.target_variable import run_target_variable_analysis
from modules.intraclass_variability import run_intraclass_variability
from modules.rgb_stats import run_rgb_stats
from modules.palette_stats import run_palette_stats

from modules.pca_analysis import run_pca_analysis
from modules.color_importance import run_color_importance
from modules.class_separability import run_class_separability_analysis
from modules.aux_classifier import run_aux_classifier


def main():
    print("=== Running 1.x Summary General ===")
    run_summary_general()

    print("=== Running 2.x Data Quality ===")
    run_data_quality()

    print("=== Running 3.x Target Variable (real vs noise) ===")
    run_target_variable_analysis()

    print("=== Running 3.3 Intraclass Variability ===")
    run_intraclass_variability()

    print("=== Running 4.1 RGB Stats ===")
    run_rgb_stats()

    print("=== Running 4.2 Palette Stats ===")
    run_palette_stats()

    print("=== Running 5.1 PCA Analysis ===")
    run_pca_analysis()

    print("=== Running 5.2 Color Importance Analysis ===")
    run_color_importance()

    print("=== Running 5.3 Class Separability Analysis ===")
    run_class_separability_analysis()

    print("=== Running 6.3 Auxiliary Classifier Analysis ===")
    run_aux_classifier()


if __name__ == "__main__":
    main()
