"""
Inferencia progresiva (en cascada):

    Imagen -> Modelo Etapa 1 (coarse: WBC / Plaqueta / Precursor eritroide)
           -> si es WBC, Modelo Etapa 2 (subtipo de glóbulo blanco)

Usa el backend único basado en tf.keras (src/inference.py). Este módulo
puede usarse desde línea de comandos o importado por el servidor Flask
(api/app.py).
"""
import argparse
import io

from PIL import Image

from src.config import CLASS_NAMES, CLASS_DESCRIPTIONS, CONFIDENCE_THRESHOLD, RBC_FINE_STAGE_AVAILABLE
from src import inference as backend


def predict_progressive(image: Image.Image) -> dict:
    """
    Ejecuta la clasificación en cascada completa sobre una única imagen PIL.
    Devuelve un diccionario listo para serializar a JSON (usado por la API).
    """
    batch = backend.preprocess_image(image)

    coarse_names = CLASS_NAMES["coarse"]
    coarse_class, coarse_conf, coarse_probs = backend.predict_from_array(batch, "coarse", coarse_names)

    result = {
        "coarse_class": coarse_class,
        "coarse_confidence": round(coarse_conf, 4),
        "coarse_probabilities": {k: round(v, 4) for k, v in coarse_probs.items()},
        "coarse_description": CLASS_DESCRIPTIONS.get(coarse_class, ""),
        "fine_class": None,
        "fine_confidence": None,
        "fine_probabilities": {},
        "fine_description": "",
        "uncertain": coarse_conf < CONFIDENCE_THRESHOLD,
    }

    if coarse_class == "WBC":
        fine_names = CLASS_NAMES["wbc"]
        fine_class, fine_conf, fine_probs = backend.predict_from_array(batch, "wbc", fine_names)
        result.update({
            "fine_class": fine_class,
            "fine_confidence": round(fine_conf, 4),
            "fine_probabilities": {k: round(v, 4) for k, v in fine_probs.items()},
            "fine_description": CLASS_DESCRIPTIONS.get(fine_class, ""),
        })
    elif coarse_class == "Precursor_eritroide" and RBC_FINE_STAGE_AVAILABLE:
        # Solo se ejecuta si en el futuro se agrega un dataset verificado
        # de morfología de eritrocitos maduros (ver src/config.py).
        fine_names = CLASS_NAMES["rbc"]
        fine_class, fine_conf, fine_probs = backend.predict_from_array(batch, "rbc", fine_names)
        result.update({
            "fine_class": fine_class,
            "fine_confidence": round(fine_conf, 4),
            "fine_probabilities": {k: round(v, 4) for k, v in fine_probs.items()},
            "fine_description": CLASS_DESCRIPTIONS.get(fine_class, ""),
        })
    # Si es Plaqueta, o Precursor_eritroide sin etapa fina disponible,
    # el resultado se queda en la Etapa 1 (comportamiento esperado y documentado).

    return result


def predict_from_bytes(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes))
    return predict_progressive(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia progresiva sobre una imagen")
    parser.add_argument("--image", required=True, help="Ruta a la imagen a clasificar")
    args = parser.parse_args()

    img = Image.open(args.image)
    output = predict_progressive(img)

    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))
