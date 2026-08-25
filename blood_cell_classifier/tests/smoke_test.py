"""
Smoke test del pipeline de inferencia.

Genera 2 modelos Keras sintéticos (pesos aleatorios, misma forma de
entrada/salida que los modelos reales) y corre la cascada completa
(api Flask incluida) para verificar que todo el software funciona
ANTES de invertir tiempo en entrenar modelos reales.

Uso:
    python -m tests.smoke_test
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import STAGE_MODEL_PATHS, CLASS_NAMES, IMG_SIZE, MODELS_DIR, RBC_FINE_STAGE_AVAILABLE


def _make_dummy_keras_model(num_classes: int, out_path: str, seed: int = 42):
    """Crea un modelo Keras minimal (GlobalAveragePooling2D + Dense softmax) con pesos aleatorios."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(seed)

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.save(out_path)


def generate_dummy_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    stages = [("coarse", 1), ("wbc", 2)]
    if RBC_FINE_STAGE_AVAILABLE:
        stages.append(("rbc", 3))
    for stage, seed in stages:
        num_classes = len(CLASS_NAMES[stage])
        _make_dummy_keras_model(num_classes, STAGE_MODEL_PATHS[stage], seed=seed)
        print(f"  · modelo sintético generado para '{stage}' ({num_classes} clases)")
    if not RBC_FINE_STAGE_AVAILABLE:
        print("  · etapa 'rbc' omitida (sin clases definidas: el dataset PBC no "
              "incluye morfología de eritrocitos maduros, ver src/config.py)")


def run_pipeline_check():
    from src.predict import predict_progressive

    dummy_image = Image.fromarray((np.random.rand(*IMG_SIZE, 3) * 255).astype("uint8"))
    result = predict_progressive(dummy_image)

    assert "coarse_class" in result, "Falta 'coarse_class' en el resultado"
    assert result["coarse_class"] in CLASS_NAMES["coarse"], "Clase gruesa inesperada"
    print(f"  · predicción de cascada OK -> {result['coarse_class']}"
          + (f" / {result['fine_class']}" if result["fine_class"] else ""))
    return result


def run_api_check():
    from api.app import app

    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200, "Endpoint /health falló"
    print(f"  · /health OK -> {r.get_json()}")

    buf_path = "/tmp/_smoke_test_image.png"
    Image.fromarray((np.random.rand(*IMG_SIZE, 3) * 255).astype("uint8")).save(buf_path)
    with open(buf_path, "rb") as f:
        r2 = client.post("/predict", data={"image": (f, "img.png")}, content_type="multipart/form-data")
    assert r2.status_code == 200, f"Endpoint /predict falló: {r2.get_json()}"
    print(f"  · /predict OK -> {r2.get_json()['coarse_class']}")


def main():
    print("1) Generando modelos Keras sintéticos (pesos aleatorios)...")
    generate_dummy_models()

    print("\n2) Probando cascada de inferencia (src/predict.py)...")
    run_pipeline_check()

    print("\n3) Probando servidor Flask (api/app.py)...")
    run_api_check()

    print("\n✅ Smoke test completo: el software funciona de punta a punta.")
    print("   Recordá: estos modelos son ALEATORIOS, solo validan el software, "
          "no el rendimiento real. Reemplazalos entrenando con src/train.py.")


if __name__ == "__main__":
    main()
