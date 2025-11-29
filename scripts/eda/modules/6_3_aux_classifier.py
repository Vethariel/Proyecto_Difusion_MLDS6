"""
6.3 - Auxiliary CNN Classifier
TDSP - EDA

Pequeño clasificador para evaluar separabilidad real entre clases del dataset.
No busca performance final, solo medir fuerza del "signal visual".

Salida:
    - Gráficas PNG en reports/figures/eda/aux_classifier/
    - confusion_matrix.png
    - accuracy_curve.png
    - loss_curve.png
    - aux_classifier_results.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ============================
# CONFIG
# ============================
DATA_PATH = Path("data/intermediate/pixel_art_data.npz")
FIG_DIR = Path("reports/figures/eda/aux_classifier")
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = Path("reports/eda/aux_classifier_results.txt")
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================
# LOAD DATA
# ============================
data = np.load(DATA_PATH)
images = data["images"]          # (89400,16,16,3) normalizado
labels = data["labels"]          # (89400,)
num_classes = len(np.unique(labels))

print(f"[INFO] Loaded dataset: {images.shape}, classes={num_classes}")

# train/val split
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)


# ============================
# BUILD SMALL CNN
# ============================
def build_model(num_classes):
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


model = build_model(num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ============================
# TRAIN
# ============================
print("\n[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=256,
    verbose=2
)


# ============================
# PREDICTIONS
# ============================
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)


# ============================
# 1. ACCURACY & LOSS CURVES
# ============================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(FIG_DIR / "accuracy_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(FIG_DIR / "loss_curve.png", dpi=200)
plt.close()


# ============================
# 2. CONFUSION MATRIX
# ============================
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=False, fmt="d", cmap="viridis")
plt.title("Confusion Matrix - Auxiliary CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=200)
plt.close()


# ============================
# 3. SAVE TEXT RESULTS
# ============================
report = classification_report(y_val, y_pred, digits=4)
with open(RESULTS_FILE, "w") as f:
    f.write("=== AUXILIARY CLASSIFIER RESULTS ===\n")
    f.write(report)

print("\n[INFO] Results saved to:", RESULTS_FILE)
print("[INFO] Figures saved to:", FIG_DIR)
print("\n=== COMPLETED 6.3 ===\n")
