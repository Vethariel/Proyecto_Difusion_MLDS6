"""
6.3 - Auxiliary CNN Classifier
TDSP - EDA

Entrena un clasificador CNN pequeño para comprobar:
- Separabilidad real entre clases
- Fuerza del signal visual
- Viabilidad de modelos condicionados por clase

Salida:
    - Gráficas PNG en reports/figures/eda/aux_classifier/
    - confusion_matrix.png
    - accuracy_curve.png
    - loss_curve.png
    - aux_classifier_results.txt
    - Entrada 'aux_classifier' en eda.json
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

from .utils import update_eda_json


# =====================================================================
# CONFIG
# =====================================================================
DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/aux_classifier")
TXT_RESULTS = Path("reports/eda/aux_classifier_results.txt")


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TXT_RESULTS.parent.mkdir(parents=True, exist_ok=True)


# =====================================================================
# CNN MODEL
# =====================================================================
def build_model(num_classes: int):
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(16,16,3)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# =====================================================================
# MAIN RUN FUNCTION
# =====================================================================
def run_aux_classifier():

    print("\n=== 6.3 Auxiliary CNN Classifier ===\n")
    ensure_dirs()

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    data = np.load(DATA_PATH)
    images = data["images"]
    labels = data["labels"]
    num_classes = len(np.unique(labels))

    print(f"[INFO] Dataset: {images.shape}, classes={num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # -----------------------------
    # BUILD & COMPILE
    # -----------------------------
    model = build_model(num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    print("[INFO] Training model (10 epochs)...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=256,
        verbose=2
    )

    # -----------------------------
    # PREDICT
    # -----------------------------
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # -----------------------------
    # 1. PLOT ACCURACY CURVE
    # -----------------------------
    acc_fig = FIG_DIR / "accuracy_curve.png"
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_fig, dpi=200)
    plt.close()

    # -----------------------------
    # 2. PLOT LOSS CURVE
    # -----------------------------
    loss_fig = FIG_DIR / "loss_curve.png"
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=200)
    plt.close()

    # -----------------------------
    # 3. CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_val, y_pred)
    cm_fig = FIG_DIR / "confusion_matrix.png"

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="viridis")
    plt.title("Confusion Matrix - Auxiliary CNN")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_fig, dpi=200)
    plt.close()

    # -----------------------------
    # 4. TXT REPORT
    # -----------------------------
    clf_report = classification_report(y_val, y_pred, digits=4)

    with open(TXT_RESULTS, "w", encoding="utf-8") as f:
        f.write("=== AUXILIARY CLASSIFIER RESULTS ===\n\n")
        f.write(clf_report)

    # -----------------------------
    # 5. BUILD RESULTS DICT (for eda.json)
    # -----------------------------
    results = {
        "n_samples": int(images.shape[0]),
        "n_classes": int(num_classes),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "accuracy_last_epoch": float(history.history["val_accuracy"][-1]),
        "loss_last_epoch": float(history.history["val_loss"][-1]),
        "figure_files": {
            "accuracy_curve": str(acc_fig),
            "loss_curve": str(loss_fig),
            "confusion_matrix": str(cm_fig)
        },
        "classification_report_file": str(TXT_RESULTS),
        "confusion_matrix": cm.tolist(),
    }

    update_eda_json("aux_classifier", results)

    print("\n=== COMPLETED 6.3 ===\n")
    return results


if __name__ == "__main__":
    run_aux_classifier()
