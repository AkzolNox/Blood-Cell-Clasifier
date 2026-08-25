"""
Definición de arquitecturas. Se usa Transfer Learning con EfficientNetB0
(buen balance precisión/velocidad) para las tres etapas del sistema progresivo.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from src.config import IMG_SIZE, LEARNING_RATE


def build_model(num_classes: int, img_size=IMG_SIZE, base_trainable: bool = False) -> tf.keras.Model:
    """
    Construye un clasificador basado en EfficientNetB0 preentrenado en ImageNet.

    num_classes: número de clases de salida de esta etapa
                 (3 para 'coarse', 5 para 'wbc', 5 para 'rbc' por defecto).
    base_trainable: si True, permite fine-tuning de las capas superiores del backbone.
    """
    inputs = layers.Input(shape=(*img_size, 3))

    # Preprocesamiento específico de EfficientNet (escala internamente)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = base_trainable

    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"blood_cell_classifier_{num_classes}c")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def unfreeze_for_fine_tuning(model: tf.keras.Model, num_layers_to_unfreeze: int = 30,
                              learning_rate: float = 1e-5) -> tf.keras.Model:
    """
    Descongela las últimas N capas del backbone para fine-tuning de baja tasa de aprendizaje.
    Se llama típicamente después de un entrenamiento inicial con el backbone congelado.
    """
    # Encuentra el submodelo base (EfficientNetB0) dentro del modelo funcional
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
