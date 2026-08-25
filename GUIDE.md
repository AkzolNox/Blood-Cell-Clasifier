# Guía paso a paso — De cero a demo funcionando

Esta guía asume que ya descomprimiste el proyecto y estás parado en la carpeta
`blood_cell_classifier/`. Todo corre en un solo entorno de Python (sin separar
entrenamiento y servicio, y sin exportar a ningún formato intermedio).

---

## Paso 1 — Instalar

Seguí **[`INSTALL.md`](./INSTALL.md)** (3 pasos, un solo `requirements.txt`,
compatible con Python 3.10 a 3.13+). Volvé acá cuando termines.

## Paso 2 — Obtener los datos (dataset verificado)

Este proyecto usa como fuente principal el dataset **PBC (Hospital Clínic de
Barcelona)**, replicado en Kaggle como `unclesamulus/blood-cells-image-dataset`
— 17,092 imágenes, 8 clases, anotadas por patólogos clínicos. Ver
`data/README_dataset.md` para el detalle completo y sus limitaciones
(no incluye morfologías de eritrocitos maduros, solo su precursor).

**Opción A — Automática (recomendada):**

```bash
# 1) Crear cuenta gratis en kaggle.com
# 2) kaggle.com -> Account -> "Create New Token" -> descarga kaggle.json
# 3) Colocarlo en ~/.kaggle/kaggle.json (o C:\Users\<usuario>\.kaggle\kaggle.json)

python -m src.download_data
```

Esto descarga el dataset con `kagglehub` y lo reorganiza automáticamente en:

```
data/
├── coarse/{Plaqueta, Precursor_eritroide, WBC}/
└── wbc_fine/{Basofilo, Eosinofilo, Granulocito_inmaduro, Linfocito, Monocito, Neutrofilo}/
```

**Opción B — Manual:** descargá el .zip desde la página del dataset en
Kaggle, descomprimilo, y copiá cada subcarpeta según la tabla de
equivalencias en `data/README_dataset.md`.

## Paso 3 — Entrenar los modelos disponibles

```bash
python -m src.train --stage coarse --epochs 20
python -m src.train --stage wbc --epochs 30 --fine_tune
```

> La etapa `rbc` no se entrena por defecto: el dataset verificado usado no
> contiene morfologías de eritrocitos maduros (ver `data/README_dataset.md`).
> Queda declarada en el código como extensión futura.

Verificá al final de cada corrida:
- `models/<stage>_best.keras` — el modelo entrenado (este mismo archivo se usa
  después directamente para servir la API, sin ningún paso de exportación).
- `models/<stage>_history.png` — curvas de loss/accuracy.
- `models/<stage>_class_indices.json` — mapeo de clases usado (por si el
  orden alfabético de tus carpetas difiere del de `src/config.py`).

## Paso 4 — Evaluar (para tu informe académico)

```bash
python -m src.evaluate --stage coarse
python -m src.evaluate --stage wbc
```

Genera en `models/`:
- `<stage>_classification_report.txt` (precision/recall/F1 por clase)
- `<stage>_confusion_matrix.png`

## Paso 5 — Validar que todo funciona (smoke test)

Antes (o en lugar) de esperar a tener modelos reales entrenados, podés
confirmar que el software entero (cascada + API Flask) funciona usando
modelos Keras sintéticos generados al vuelo:

```bash
python -m tests.smoke_test
```

Deberías ver algo como:

```
1) Generando modelos Keras sintéticos (pesos aleatorios)...
2) Probando cascada de inferencia (src/predict.py)...
3) Probando servidor Flask (api/app.py)...
✅ Smoke test completo...
```

> Nota: este comando sobreescribe temporalmente `models/*.keras` con modelos
> aleatorios solo para probar el software. Si ya entrenaste modelos reales
> (Paso 3), volvé a correr `src/train.py` después del smoke test para
> restaurarlos, o hacé una copia de tus modelos reales antes de probar.

## Paso 6 — Levantar la API

Con tus modelos reales ya en `models/*.keras` (Paso 3):

```bash
python api/app.py
```

Deberías ver:

```
[api] Precargando modelos entrenados (tf.keras)...
[inference] Etapa 'coarse' precargada correctamente.
[inference] Etapa 'wbc' precargada correctamente.
 * Running on http://0.0.0.0:5000
```

Probalo rápido con curl:

```bash
curl http://localhost:5000/health
# {"status": "ok"}
```

## Paso 7 — Levantar la interfaz web

```bash
python -m http.server 8080 -d frontend
```

Abrí `http://localhost:8080` en el navegador. Tenés dos formas de dar una imagen:
- **Subir imagen**: arrastrá o seleccioná un archivo.
- **Cámara del microscopio**: si tenés una cámara USB conectada (por ejemplo,
  acoplada al ocular de un microscopio), elegila de la lista y capturá un cuadro.

En ambos casos, al hacer clic en "Analizar muestra" vas a ver la cascada completa:
Etapa 1 (WBC/Plaqueta/Precursor eritroide) → Etapa 2 (subtipo), con las
probabilidades de cada clase como barras.

---

## Resumen de comandos (referencia rápida)

Instalación: ver `INSTALL.md`. Con eso ya hecho:

```bash
python -m src.download_data        # descarga y organiza el dataset verificado
python -m src.train --stage coarse --epochs 20
python -m src.train --stage wbc --epochs 30 --fine_tune
python -m src.evaluate --stage wbc
python -m tests.smoke_test          # opcional, valida el software
python api/app.py
python -m http.server 8080 -d frontend
```

## Checklist final antes de entregar el proyecto académico

- [ ] Entorno instalado sin errores siguiendo `INSTALL.md` (cualquier Python 3.10+).
- [ ] Dataset verificado descargado y organizado (`python -m src.download_data`).
- [ ] Los 2 modelos `.keras` entrenados (`coarse`, `wbc`) y sus reportes de evaluación guardados.
- [ ] `tests/smoke_test.py` corrido con éxito al menos una vez.
- [ ] API Flask respondiendo en `/health` y `/predict`.
- [ ] Frontend mostrando la cascada completa con imágenes reales de tu dataset
      (y, si aplica, con la cámara del microscopio conectada).
- [ ] Mencionar en el informe la limitación conocida: sin sub-clasificación
      de morfología de eritrocitos maduros (dataset no la incluye).
