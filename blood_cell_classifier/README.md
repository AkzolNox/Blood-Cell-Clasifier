# Hematoscope — Clasificación Progresiva de Células Sanguíneas con IA

Proyecto académico de Inteligencia Artificial para la **detección y clasificación jerárquica
(progresiva)** de células sanguíneas a partir de imágenes de microscopía, con interfaz web
de inferencia en tiempo real (carga de archivo o cámara de microscopio en vivo).

---

## 📋 Requisitos

| Requisito | Detalle |
|---|---|
| **Python** | 3.10 o más nuevo, **incluyendo 3.13+** (sin restricciones de versión) |
| **Espacio en disco** | ~1-2 GB libres (TensorFlow + dependencias) |
| **Sistema operativo** | Windows, macOS o Linux |
| **Conexión a internet** | Para instalar dependencias y descargar el dataset |
| **Cuenta de Kaggle (gratis)** | Solo si vas a descargar el dataset con `src/download_data.py` |
| **Cámara USB (opcional)** | Solo si querés usar el modo "cámara en vivo" del frontend (cualquier cámara UVC estándar, p. ej. acoplada a un microscopio) |
| **GPU (opcional)** | Acelera el entrenamiento, pero no es necesaria — funciona en CPU |

No hace falta instalar CUDA, drivers de cámara especiales, ni ningún motor de
inferencia externo: todo corre con TensorFlow/Keras y librerías estándar de Python.

## ⚙️ Instalación

```bash
cd blood_cell_classifier
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Con eso ya está todo instalado (un solo `requirements.txt` para entrenar, evaluar
y correr la API). Para confirmar que quedó bien instalado:

```bash
python -m tests.smoke_test
```

Si termina con `✅ Smoke test completo`, la instalación funciona. Si algo falla,
**[`INSTALL.md`](./INSTALL.md)** tiene la solución a los errores más comunes
(módulo faltante, Mac M1/M2/M3, Windows, TensorFlow lento de instalar, etc.).

## ▶️ Uso

**1) Conseguir los datos** (dataset verificado, descarga y organiza automático):
```bash
python -m src.download_data   # requiere ~/.kaggle/kaggle.json, ver sección Dataset más abajo
```

**2) Entrenar los modelos:**
```bash
python -m src.train --stage coarse --epochs 20
python -m src.train --stage wbc --epochs 30 --fine_tune
```

**3) (Opcional) Evaluar con métricas para tu informe:**
```bash
python -m src.evaluate --stage wbc
```

**4) Levantar la aplicación:**
```bash
python api/app.py                        # backend en http://localhost:5000
python -m http.server 8080 -d frontend   # frontend en http://localhost:8080
```

Abrí `http://localhost:8080` en el navegador. Ahí podés:
- **Subir imagen**: arrastrar o seleccionar cualquier imagen de una célula.
- **Cámara del microscopio**: capturar en vivo desde una cámara USB conectada,
  con opción de análisis automático continuo cada 2.5 segundos.

En ambos casos vas a ver la cascada de clasificación completa (Etapa 1 → Etapa 2),
con el nivel de confianza, las probabilidades por clase y una descripción
morfológica educativa.

> Guía detallada paso a paso, con capturas de lo que deberías ver en cada paso
> y un checklist final: **[`GUIDE.md`](./GUIDE.md)**.

---

## 1. Idea general: ¿por qué "progresiva"?

En lugar de entrenar un único clasificador plano con todas las clases mezcladas, el sistema se organiza
en **dos etapas progresivas**, imitando el razonamiento de un hematólogo:

```
Etapa 1 (Grueso)                          Etapa 2 (Fino)
─────────────────                         ─────────────────────────────
Imagen de célula ──▶ ¿WBC, Plaqueta o     ──▶  Sub-clasificador especializado
                      Precursor eritroide?         │
                     Si es WBC ───────────────────┤──▶ Neutrófilo / Eosinófilo / Basófilo /
                                                    │    Linfocito / Monocito / Granulocito inmaduro
                     Si es Precursor eritroide ────┘──▶ (sin sub-clasificador: el dataset verificado
                                                          usado no incluye morfologías de eritrocitos
                                                          maduros, ver data/README_dataset.md)
```

> **Nota sobre los datos:** este proyecto usa el dataset verificado
> [PBC / Hospital Clínic de Barcelona](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset)
> (17,092 imágenes, 8 clases, anotado por patólogos clínicos). Por eso la
> Etapa 1 distingue "WBC / Plaqueta / Precursor eritroide" en vez de
> "RBC / WBC / Plaqueta": el dataset no contiene glóbulos rojos maduros,
> solo su precursor nucleado (erythroblast). Ver `data/README_dataset.md`
> para el detalle completo y las clases exactas.

Ventajas académicas de este enfoque:
- Cada modelo resuelve un problema más simple y balanceado → mejor accuracy con menos datos.
- Se puede explicar por separado la matriz de confusión de cada etapa (bueno para un informe/tesis).
- Es fácilmente extensible: agregar una tercera etapa (p. ej. detección de anomalías) no rompe el resto.

## 2. Arquitectura técnica

El proyecto usa **un solo entorno** de Python para todo (entrenar, evaluar y
servir la API), basado íntegramente en TensorFlow/Keras:

```
┌───────────────────────────────────────────────────────────┐
│                    UN SOLO ENTORNO                          │
│              Python + TensorFlow/Keras                      │
│                                                              │
│   src/train.py  →  models/*.keras  →  src/inference.py     │
│                                            │                 │
│                                            ▼                 │
│                                     api/app.py (Flask)       │
└───────────────────────────────────────────────────────────┘
```

- **Keras/TensorFlow** se usa para entrenar los modelos (transfer learning con
  EfficientNetB0) y **también** para servirlos en producción (`src/inference.py`
  carga el `.keras` directo con `tf.keras.models.load_model`).
- No hay exportación a ningún formato intermedio ni motor de inferencia
  alternativo: un solo framework, un solo `requirements.txt`.
- Esta simplicidad tiene un costo (el servidor de la API pesa más porque
  necesita TensorFlow instalado) a cambio de compatibilidad total con
  versiones recientes de Python (3.10 a 3.13+), sin dependencias que se queden
  atrás respecto al resto del ecosistema.

| Componente            | Tecnología                                   |
|------------------------|-----------------------------------------------|
| Preprocesamiento y augmentación | Python, OpenCV, `tf.data`, `albumentations` |
| Modelos (entrenamiento e inferencia) | TensorFlow / Keras (Transfer Learning: EfficientNetB0) |
| Servidor de inferencia  | Python, Flask + Flask-CORS                   |
| Interfaz de usuario     | HTML5, CSS3, JavaScript (fetch API, drag&drop, captura de cámara en vivo) |
| Pruebas                 | `tests/smoke_test.py` (valida todo el pipeline sin modelos reales) |
| Contenedorización (opcional) | Docker                                  |

## 3. Estructura del proyecto

```
blood_cell_classifier/
├── README.md
├── GUIDE.md                       # Guía paso a paso (de cero a demo funcionando)
├── INSTALL.md                      # Instalación y solución de errores comunes
├── requirements.txt                 # Todas las dependencias (entrenar + servir)
├── data/
│   └── README_dataset.md          # Cómo descargar y organizar el dataset
├── src/
│   ├── config.py                   # Rutas, hiperparámetros, nombres de clases
│   ├── data_pipeline.py             # Carga, augmentación y tf.data.Dataset
│   ├── model.py                     # Definición de los 3 modelos (etapa1, wbc, rbc)
│   ├── train.py                     # Script de entrenamiento (CLI)
│   ├── download_data.py             # Descarga y organiza el dataset verificado desde Kaggle
│   ├── evaluate.py                  # Métricas, matriz de confusión, reporte
│   ├── inference.py                 # Carga y ejecuta los modelos .keras (backend único)
│   └── predict.py                   # Orquestador de la cascada de 2 etapas
├── tests/
│   └── smoke_test.py                # Valida TODO el pipeline con modelos sintéticos (sin entrenar)
├── models/                          # Pesos entrenados (.keras)
├── api/
│   └── app.py                       # Servidor Flask que expone /predict
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js                    # Modo de carga de archivo y cámara en vivo (con auto-análisis)
```

## 4. Dataset usado (verificado)

Ver `data/README_dataset.md` para el detalle completo, tabla de clases exacta,
limitaciones y equivalencias de carpetas.

**Fuente principal:** [Blood Cells Image Dataset](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset)
(Kaggle) — réplica del dataset **PBC** del Hospital Clínic de Barcelona
(Acevedo et al., *Data in Brief*, 2020; también en Mendeley Data). 17,092
imágenes, 8 clases, anotadas por patólogos clínicos expertos.

```bash
# requiere ~/.kaggle/kaggle.json con tu token de la API de Kaggle
# (kaggle.com -> Account -> "Create New Token")
python -m src.download_data
```

> Se evaluó también `mohamadabouali1/blood-cells-dataset-11-classes-26534-images`
> (más imágenes en bruto), pero no fue posible verificar públicamente su
> composición ni procedencia, por lo que no se usa por defecto — ver el
> detalle en `data/README_dataset.md` si querés evaluarla vos mismo.

## 5. Entrenamiento y evaluación en detalle

```bash
# Etapa 1: clasificador grueso WBC / Plaqueta / Precursor_eritroide
python -m src.train --stage coarse --epochs 20

# Etapa 2: sub-clasificador de glóbulos blancos (única etapa fina disponible
# con este dataset; ver limitación de RBC en data/README_dataset.md)
python -m src.train --stage wbc --epochs 30
```

> La etapa `rbc` no se entrena por defecto: `data/rbc_fine/` no existe con este
> dataset (ver `data/README_dataset.md`). Queda como extensión futura.

Cada corrida guarda el mejor modelo en `models/<stage>_best.keras` y una gráfica de
entrenamiento (loss/accuracy) en `models/<stage>_history.png`.

```bash
python -m src.evaluate --stage wbc
```

Genera matriz de confusión, precisión, recall, F1-score y curva ROC (una vs resto) por clase.

## 6. Interfaz de usuario en detalle

El usuario arrastra o selecciona una imagen de una célula, o cambia al modo
"Cámara del microscopio" para capturar en vivo desde una cámara USB conectada
(con opción de análisis automático continuo). El frontend envía la imagen al
backend, que ejecuta la inferencia en cascada (Etapa 1 → Etapa 2) y devuelve JSON con:

```json
{
  "coarse_class": "WBC",
  "coarse_confidence": 0.98,
  "fine_class": "Neutrofilo",
  "fine_confidence": 0.91,
  "probabilities": {"Neutrofilo": 0.91, "Eosinofilo": 0.04, "...": "..."}
}
```

La interfaz muestra la imagen, ambas etapas, las probabilidades como barras, y una breve
descripción morfológica educativa de la clase detectada.

## 7. Extensiones posibles (para la parte "progresiva" del proyecto)

- Añadir una Etapa 3 de detección de anomalías (leucemia, anemia falciforme) usando autoencoders.
- Incorporar Grad-CAM para explicar visualmente qué regiones de la célula usó el modelo.
- Fine-tuning incremental (aprendizaje continuo) cuando lleguen nuevas imágenes etiquetadas.
- Optimizar la inferencia con `TFLite` o `TensorRT` si se necesita más velocidad, manteniendo
  todo dentro del ecosistema de TensorFlow (sin volver a introducir un framework externo).

## 8. Aviso académico

Este proyecto es con fines educativos/de investigación. **No debe usarse para diagnóstico clínico real.**
Los datasets públicos usados tienen fines de entrenamiento académico y las predicciones del modelo
no sustituyen la evaluación de un profesional de la salud.
