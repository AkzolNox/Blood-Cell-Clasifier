"""
Backend de inferencia: carga y ejecuta los modelos entrenados directamente
con tf.keras (formato .keras). Es el único backend del proyecto desde que se
retiró ONNX (ver notas de la versión sin ONNX en README.md / CHANGELOG).

Por qué un solo backend con tf.keras:
- Elimina la dependencia de tf2onnx, que era el verdadero cuello de botella
  de compatibilidad: no soporta Python 3.13+ ni versiones recientes de
  TensorFlow, y obligaba a mantener dos entornos separados (entrenamiento
  y servicio) solo para evitar ese problema.
- TensorFlow por sí solo sí sigue versiones recientes de Python (2.20/2.21
  ya soportan 3.13), así que entrenamiento e inferencia pueden vivir en el
  mismo entorno sin restricciones especiales de versión.
- El costo es que el servidor de inferencia ahora sí necesita TensorFlow
  instalado (más pesado que ONNX Runtime), pero a cambio se gana
  compatibilidad con Python moderno y un solo `requirements.txt`.
"""
import os

import numpy as np
from PIL import Image

from src.config import STAGE_MODEL_PATHS, IMG_SIZE, CLASS_NAMES

_MODEL_CACHE = {}


def _get_model(stage: str):
    import tensorflow as tf  # import perezoso: solo se paga el costo de importar TF si hace falta

    if stage not in _MODEL_CACHE:
        path = STAGE_MODEL_PATHS[stage]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Falta el modelo entrenado de la etapa '{stage}' en {path}. "
                f"Entrena primero con: python -m src.train --stage {stage}"
            )
        _MODEL_CACHE[stage] = tf.keras.models.load_model(path)
    return _MODEL_CACHE[stage]


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_from_array(batch: np.ndarray, stage: str, class_names: list):
    model = _get_model(stage)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), dict(zip(class_names, [float(p) for p in probs]))


def warm_up(stages=("coarse", "wbc", "rbc")):
    """
    Precarga los modelos al iniciar el servidor, para que la primera
    petición del usuario no pague el costo de cargar el modelo en memoria.
    Ignora etapas sin clases definidas (p. ej. 'rbc' con este dataset) y
    avisa (sin fallar) si un modelo aún no fue entrenado/exportado.
    """
    for stage in stages:
        if not CLASS_NAMES.get(stage):
            continue
        try:
            _get_model(stage)
            print(f"[inference] Etapa '{stage}' precargada correctamente.")
        except FileNotFoundError as e:
            print(f"[inference] Aviso: {e}")
