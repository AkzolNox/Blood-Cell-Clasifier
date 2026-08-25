"""
Servidor Flask que expone la inferencia progresiva vía API REST.

Endpoints:
    GET  /health           -> chequeo de estado
    POST /predict          -> recibe una imagen (multipart/form-data, campo 'image')
                               y devuelve el resultado de la clasificación en cascada.

Ejecutar:
    python api/app.py
"""
import os
import sys

# Permite importar el paquete src/ estando dentro de api/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.predict import predict_from_bytes

app = Flask(__name__)
CORS(app)  # habilita llamadas desde el frontend servido en otro puerto/origen

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ningún archivo con la clave 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Extensión no permitida. Usa: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    try:
        image_bytes = file.read()
        result = predict_from_bytes(image_bytes)
        return jsonify(result)
    except FileNotFoundError as e:
        # Ocurre si aún no se ha entrenado alguno de los modelos
        return jsonify({"error": str(e)}), 503
    except Exception as e:  # pragma: no cover
        return jsonify({"error": f"Error interno al procesar la imagen: {e}"}), 500


if __name__ == "__main__":
    print("[api] Precargando modelos entrenados (tf.keras)...")
    from src.inference import warm_up
    warm_up()
    app.run(host="0.0.0.0", port=5000, debug=True)
