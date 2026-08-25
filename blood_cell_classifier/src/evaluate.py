"""
Evaluación de un modelo entrenado: matriz de confusión, precisión/recall/F1
y reporte de clasificación completo (útil para el informe académico).
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import STAGE_DATA_DIRS, STAGE_MODEL_PATHS, MODELS_DIR
from src.data_pipeline import load_datasets


def evaluate_stage(stage: str):
    data_dir = STAGE_DATA_DIRS[stage]
    model_path = STAGE_MODEL_PATHS[stage]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe un modelo entrenado en {model_path}. Ejecuta primero src/train.py")

    print(f"Cargando modelo de la etapa '{stage}' desde {model_path}")
    model = tf.keras.models.load_model(model_path)

    _, val_ds, class_names = load_datasets(data_dir, augment=False)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)

    report_path = os.path.join(MODELS_DIR, f"{stage}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Reporte guardado en {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión - etapa {stage}")
    plt.tight_layout()

    cm_path = os.path.join(MODELS_DIR, f"{stage}_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Matriz de confusión guardada en {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa un modelo entrenado de una etapa")
    parser.add_argument("--stage", choices=["coarse", "wbc", "rbc"], required=True)
    args = parser.parse_args()
    evaluate_stage(args.stage)
