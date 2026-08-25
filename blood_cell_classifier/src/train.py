"""
Script de entrenamiento con interfaz de línea de comandos.

Ejemplos:
    python -m src.train --stage coarse --epochs 20
    python -m src.train --stage wbc --epochs 30 --fine_tune
    python -m src.train --stage rbc --epochs 30
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    STAGE_DATA_DIRS, STAGE_MODEL_PATHS, CLASS_NAMES, MODELS_DIR,
    DEFAULT_EPOCHS, FINE_TUNE_LEARNING_RATE,
)
from src.data_pipeline import load_datasets
from src.model import build_model, unfreeze_for_fine_tuning


def plot_history(history, stage: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title(f"Accuracy - etapa {stage}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title(f"Loss - etapa {stage}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    out_path = os.path.join(MODELS_DIR, f"{stage}_history.png")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Gráfica guardada en {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo de una etapa del pipeline progresivo")
    parser.add_argument("--stage", choices=["coarse", "wbc", "rbc"], required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--fine_tune", action="store_true", help="Aplica fine-tuning tras el entrenamiento inicial")
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    args = parser.parse_args()

    data_dir = STAGE_DATA_DIRS[args.stage]
    model_path = STAGE_MODEL_PATHS[args.stage]
    class_names_expected = CLASS_NAMES[args.stage]

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No se encontró la carpeta de datos '{data_dir}'.\n"
            f"Revisa data/README_dataset.md para preparar el dataset de la etapa '{args.stage}'."
        )

    print(f"Cargando datos desde {data_dir} ...")
    train_ds, val_ds, class_names = load_datasets(data_dir)
    print(f"Clases encontradas: {class_names}")

    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune:
        print("Iniciando fine-tuning del backbone...")
        model = unfreeze_for_fine_tuning(model, learning_rate=FINE_TUNE_LEARNING_RATE)
        history_ft = model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.fine_tune_epochs, callbacks=callbacks,
        )
        for k in history.history:
            history.history[k] += history_ft.history[k]

    plot_history(history, args.stage)

    # Guarda el mapeo de índices a nombres de clase (por si el orden difiere del de config.py)
    mapping_path = os.path.join(MODELS_DIR, f"{args.stage}_class_indices.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"Entrenamiento completo. Mejor modelo guardado en {model_path}")
    print(f"Mapeo de clases guardado en {mapping_path}")


if __name__ == "__main__":
    main()
