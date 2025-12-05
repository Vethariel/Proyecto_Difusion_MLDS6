"""
run_eda.py
TDSP - EDA Orchestrator
Ejecuta todos los análisis EDA en orden.
"""
from modules.pca_analysis import run_pca_analysis
from modules.color_importance import run_color_importance
from modules.class_separability import run_class_separability_analysis
from modules.aux_classifier import run_aux_classifier

def main():
    print("=== Running 5.1 PCA Analysis ===")
    run_pca_analysis()

    print("=== Running 5.2 Color Importance Analysis ===")
    run_color_importance()

    print("=== Running 5.3 Class Separability Analysis ===")
    run_class_separability_analysis()

    print("=== Running 6.1 Auxiliary Classifier Analysis ===")
    run_aux_classifier()

if __name__ == "__main__":
    main()
