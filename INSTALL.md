# Instalación — Guía rápida

Esta es la única guía que necesitás para instalar el proyecto.

## Paso 1 — Creá un entorno virtual (recomendado)

```bash
cd blood_cell_classifier
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Cualquier Python 3.10 o más nuevo funciona, **incluyendo 3.13+**. El proyecto ya
no depende de `tf2onnx` (la herramienta que antes limitaba la versión de Python
compatible), así que no hay restricciones especiales que revisar.

## Paso 2 — Instalá las dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Es un solo archivo para todo: entrenar, evaluar y correr la API. No hay
entornos separados que mantener sincronizados.

## Paso 3 — Verificá que quedó bien instalado

```bash
python -m tests.smoke_test
```

Si ves `✅ Smoke test completo` al final, la instalación funciona correctamente
(usa modelos de prueba generados al vuelo, no hace falta tener datos reales todavía).

---

## Errores comunes

**`ModuleNotFoundError: No module named 'flask_cors'` (o cualquier otro módulo)**
No activaste el entorno virtual, o el `pip install` del Paso 2 no terminó bien.
Repetí el Paso 1 y el Paso 2, revisando que no haya errores en la instalación.

**`pip install` tarda muchísimo o parece colgado instalando tensorflow**
Es normal — TensorFlow pesa varios cientos de MB. Esperá unos minutos; si pasan más
de 15-20 minutos sin avance, cancelá (`Ctrl+C`) y volvé a correr el mismo comando
(pip retoma la descarga).

**Windows: `pip` o `python3` "no se reconoce como un comando"**
Usá `python` en vez de `python3`, y `py -m venv venv` si `python` tampoco funciona.

**Mac con chip M1/M2/M3: TensorFlow se instala pero es muy lento o falla al importar**
Instalá estas dos variantes específicas de Apple antes que el resto:
```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

**Error de versiones de NumPy incompatibles con TensorFlow**
Poco común, pero si aparece: `pip install "numpy<2"` y volvé a correr
`pip install -r requirements.txt`.

---

## ¿Qué instala este requirements.txt?

| Paquete | Para qué sirve |
|---|---|
| `tensorflow` | Entrenar los modelos y también servir la API (un solo motor para todo) |
| `opencv-python-headless`, `albumentations` | Preprocesamiento y aumentado de imágenes |
| `scikit-learn`, `matplotlib`, `seaborn` | Métricas y gráficas de evaluación |
| `kagglehub` | Descargar el dataset verificado desde Kaggle |
| `Flask`, `flask-cors` | Servidor de la API REST |
| `Pillow`, `numpy`, `tqdm` | Utilidades generales |

Pesa varios cientos de MB (por TensorFlow), pero es el único archivo que vas a
necesitar instalar en cualquier máquina donde quieras entrenar, evaluar o
correr la demo.
