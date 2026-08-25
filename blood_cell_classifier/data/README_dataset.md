# Dataset usado en este proyecto (verificado)

## Fuente principal

**Blood Cells Image Dataset** — Kaggle:
https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset

Es una réplica en Kaggle del dataset **PBC (Peripheral Blood Cell)** del
Hospital Clínic de Barcelona, publicado originalmente en:

> Acevedo, A. et al. "A dataset of microscopic peripheral blood cell images
> for development of automatic recognition systems." *Data in Brief*, 2020.
> También disponible en Mendeley Data: https://data.mendeley.com/datasets/snkd93bnjr/1

**Por qué es una fuente verificada:**
- Imágenes adquiridas con el analizador **CellaVision DM96** en un laboratorio clínico real.
- Anotadas por **patólogos clínicos expertos**.
- Pacientes confirmados sin infección, enfermedad hematológica/oncológica ni
  tratamiento farmacológico al momento de la toma.
- Publicación revisada por pares, citada en decenas de papers académicos con
  conteos de clases consistentes entre sí.

| Clase original (inglés) | Clase en este proyecto | Nº imágenes |
|---|---|---|
| neutrophil    | Neutrofilo             | 3,329 |
| eosinophil    | Eosinofilo             | 3,117 |
| ig (immature granulocytes) | Granulocito_inmaduro | 2,895 |
| platelet      | Plaqueta               | 2,348 |
| erythroblast  | Precursor_eritroide    | 1,551 |
| monocyte      | Monocito               | 1,420 |
| basophil      | Basofilo               | 1,218 |
| lymphocyte    | Linfocito              | 1,214 |
| **Total**     |                        | **17,092** |

## ⚠️ Limitación importante (honesta, no un bug del código)

Este dataset **no incluye morfologías de glóbulos rojos maduros** (no hay
célula falciforme, esferocito, eliptocito, célula diana, etc.). Su única
clase relacionada a la serie roja es **erythroblast** (precursor eritroide
inmaduro y nucleado, normalmente de médula ósea).

Por eso, en `src/config.py`, la etapa fina `"rbc"` queda **vacía a propósito**
(`CLASS_NAMES["rbc"] = []`). El sistema seguirá funcionando: si detecta un
"Precursor_eritroide" en la Etapa 1, simplemente no intenta un segundo paso
de sub-clasificación (no existen esas clases en los datos reales).

Si más adelante querés clasificar morfologías de eritrocitos maduros, vas a
necesitar un dataset adicional y verificado específico para eso (por ejemplo,
un banco de imágenes de anemia falciforme/talasemia con anotación clínica),
y completar `CLASS_NAMES["rbc"]` y la carpeta `data/rbc_fine/` en consecuencia.

## Otro dataset evaluado (no usado, verificación pendiente)

`mohamadabouali1/blood-cells-dataset-11-classes-26534-images` — tiene más
imágenes en bruto (26,534), pero no fue posible verificar públicamente su
composición exacta de clases ni su procedencia (Kaggle bloquea el scraping
de esa página y no se encontraron papers académicos que la citen). Si querés
usarla, revisá primero su "Data Card" en Kaggle (fuente original, licencia,
quién anotó las imágenes) antes de entrenar con ella.

## Cómo descargar y organizar (automático)

Con las dependencias ya instaladas (ver `../INSTALL.md` si no lo hiciste):

```bash
# 1) Crear cuenta en kaggle.com (gratis) y generar un token API:
#    kaggle.com -> Account -> "Create New Token" -> descarga kaggle.json
# 2) Colocarlo en ~/.kaggle/kaggle.json (Linux/Mac) o
#    C:\Users\<usuario>\.kaggle\kaggle.json (Windows)

# 3) Descargar y organizar automáticamente:
python -m src.download_data
```

Esto deja las imágenes listas en:

```
data/
├── coarse/
│   ├── Plaqueta/
│   ├── Precursor_eritroide/
│   └── WBC/                  # las 6 subclases juntas
└── wbc_fine/
    ├── Basofilo/
    ├── Eosinofilo/
    ├── Granulocito_inmaduro/
    ├── Linfocito/
    ├── Monocito/
    └── Neutrofilo/
```

## Cómo descargar manualmente (alternativa sin kagglehub)

1. Andá a https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset
2. Botón "Download" (requiere cuenta de Kaggle).
3. Descomprimí el .zip; vas a encontrar una carpeta `bloodcells_dataset/`
   con 8 subcarpetas (una por clase, en inglés).
4. Corré igual `python -m src.download_data` apuntando `raw_root` a esa
   carpeta local si preferís organizarla vos mismo, o simplemente movés
   manualmente cada subcarpeta según la tabla de equivalencias de arriba.
