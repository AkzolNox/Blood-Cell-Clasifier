"""
Configuración central del proyecto: rutas, hiperparámetros y nombres de clases.
Mantener todo en un solo lugar facilita reproducibilidad académica.
"""
import os

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

STAGE_DATA_DIRS = {
    "coarse": os.path.join(DATA_DIR, "coarse"),
    "wbc": os.path.join(DATA_DIR, "wbc_fine"),
    "rbc": os.path.join(DATA_DIR, "rbc_fine"),
}

STAGE_MODEL_PATHS = {
    "coarse": os.path.join(MODELS_DIR, "coarse_best.keras"),
    "wbc": os.path.join(MODELS_DIR, "wbc_best.keras"),
    "rbc": os.path.join(MODELS_DIR, "rbc_best.keras"),
}

# ----------------------------------------------------------------------
# Clases — basadas en el dataset REAL y VERIFICADO usado en este proyecto:
# "Blood Cells Image Dataset" (kaggle.com/datasets/unclesamulus/blood-cells-image-dataset),
# que es una réplica del dataset PBC del Hospital Clínic de Barcelona
# (Acevedo et al., "A dataset of microscopic peripheral blood cell images
# for development of automatic recognition systems", Data in Brief, 2020;
# también disponible en Mendeley Data: data.mendeley.com/datasets/snkd93bnjr/1).
#
# IMPORTANTE (limitación real del dataset, no del código):
# el dataset NO incluye morfologías de glóbulos rojos MADUROS (no hay
# célula falciforme, esferocito, eliptocito, etc.). Su única clase
# relacionada a la serie roja es "erythroblast" (precursor eritroide
# inmaduro y nucleado). Por eso la etapa "rbc" fina queda deshabilitada
# por defecto: se documenta como extensión futura si se consigue un
# dataset verificado de morfología de eritrocitos maduros.
# ----------------------------------------------------------------------

# Nombres de carpeta tal como vienen en el dataset original (bloodcells_dataset/<clase>)
KAGGLE_FOLDER_NAMES = {
    "neutrophil": "Neutrofilo",
    "eosinophil": "Eosinofilo",
    "basophil": "Basofilo",
    "lymphocyte": "Linfocito",
    "monocyte": "Monocito",
    "ig": "Granulocito_inmaduro",       # immature granulocytes (promielocito/mielocito/metamielocito)
    "erythroblast": "Precursor_eritroide",
    "platelet": "Plaqueta",
}

CLASS_NAMES = {
    # Etapa 1 (gruesa): a qué linaje celular pertenece la imagen
    "coarse": ["Plaqueta", "Precursor_eritroide", "WBC"],
    # Etapa 2 (fina): subtipo de glóbulo blanco (única etapa fina con datos reales disponibles)
    "wbc": ["Basofilo", "Eosinofilo", "Granulocito_inmaduro", "Linfocito", "Monocito", "Neutrofilo"],
    # Etapa 2 (fina) para RBC: NO disponible con este dataset (ver nota arriba).
    # Se deja declarada para no romper el código de src/predict.py si en el futuro
    # se agrega un dataset verificado de morfología de eritrocitos maduros.
    "rbc": [],
}

# Si "rbc" no tiene clases, la cascada progresiva NO intentará sub-clasificar
# los precursores eritroides; se queda en el resultado de la Etapa 1.
RBC_FINE_STAGE_AVAILABLE = len(CLASS_NAMES["rbc"]) > 0

# Descripciones educativas mostradas en la interfaz
CLASS_DESCRIPTIONS = {
    "WBC": "Glóbulo blanco (leucocito): célula del sistema inmunológico.",
    "Plaqueta": "Fragmento celular (trombocito) implicado en la coagulación sanguínea.",
    "Precursor_eritroide": (
        "Eritroblasto: precursor nucleado e inmaduro de los glóbulos rojos, "
        "presente normalmente en médula ósea. Su presencia en sangre periférica "
        "puede ser normal en neonatos o indicar estrés medular en adultos."
    ),
    "Neutrofilo": "El leucocito más abundante; primera línea de defensa contra bacterias.",
    "Eosinofilo": "Leucocito asociado a respuestas alérgicas e infecciones parasitarias.",
    "Basofilo": "El leucocito menos frecuente; libera histamina en reacciones alérgicas.",
    "Linfocito": "Célula clave de la inmunidad adaptativa (linfocitos T, B y NK).",
    "Monocito": "Precursor de macrófagos; fagocita patógenos y restos celulares.",
    "Granulocito_inmaduro": (
        "Incluye promielocitos, mielocitos y metamielocitos: formas inmaduras "
        "de la serie granulocítica, normalmente restringidas a médula ósea."
    ),
}

# ----------------------------------------------------------------------
# Hiperparámetros de entrenamiento
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
DEFAULT_EPOCHS = 25
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Umbral de confianza mínimo para no marcar la predicción como "incierta"
CONFIDENCE_THRESHOLD = 0.55
