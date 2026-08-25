"""
Descarga y organiza automáticamente el dataset verificado usado en este
proyecto: "Blood Cells Image Dataset" (kaggle.com/datasets/unclesamulus/
blood-cells-image-dataset), réplica en Kaggle del dataset PBC del Hospital
Clínic de Barcelona (Acevedo et al., Data in Brief, 2020).

Requisitos previos:
    1. Crear una cuenta en Kaggle (gratis).
    2. Generar un token API: kaggle.com -> Account -> "Create New Token".
       Esto descarga un archivo kaggle.json.
    3. Colocarlo en ~/.kaggle/kaggle.json (Linux/Mac) o
       C:\\Users\\<usuario>\\.kaggle\\kaggle.json (Windows).
    4. pip install kagglehub   (ya incluido en requirements.txt)

Uso:
    python -m src.download_data

Esto descarga el dataset y reorganiza las imágenes dentro de
data/coarse/ y data/wbc_fine/ con la nomenclatura en español que usa
el resto del proyecto (ver src/config.py -> KAGGLE_FOLDER_NAMES).
"""
import os
import shutil

from src.config import DATA_DIR, KAGGLE_FOLDER_NAMES

KAGGLE_DATASET_ID = "unclesamulus/blood-cells-image-dataset"

# A qué etapa/carpeta del proyecto pertenece cada clase original del dataset.
# ("Plaqueta" y "Precursor_eritroide" van a la etapa 'coarse' junto con "WBC";
#  las 6 subclases de WBC van además replicadas en 'wbc_fine' para la Etapa 2).
WBC_SUBCLASSES = {"neutrophil", "eosinophil", "basophil", "lymphocyte", "monocyte", "ig"}


def download_dataset() -> str:
    """Descarga el dataset con kagglehub y devuelve la ruta local donde quedó."""
    import kagglehub

    print(f"Descargando '{KAGGLE_DATASET_ID}' desde Kaggle (puede tardar varios minutos)...")
    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print(f"Dataset descargado en: {path}")
    return path


def find_class_folders(root: str) -> dict:
    """
    Busca recursivamente las subcarpetas cuyo nombre (en minúsculas) coincide
    con alguna clave de KAGGLE_FOLDER_NAMES, sin importar en qué nivel de
    anidamiento del zip descargado hayan quedado.
    """
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            key = d.lower()
            if key in KAGGLE_FOLDER_NAMES and key not in found:
                found[key] = os.path.join(dirpath, d)
    return found


def organize_dataset(raw_root: str):
    """Copia las imágenes de cada clase a data/coarse/ y data/wbc_fine/."""
    class_folders = find_class_folders(raw_root)
    missing = set(KAGGLE_FOLDER_NAMES) - set(class_folders)
    if missing:
        print(f"Aviso: no se encontraron carpetas para: {sorted(missing)}. "
              f"Revisá la estructura descargada en {raw_root}.")

    coarse_dir = os.path.join(DATA_DIR, "coarse")
    wbc_dir = os.path.join(DATA_DIR, "wbc_fine")

    for kaggle_name, es_name in KAGGLE_FOLDER_NAMES.items():
        src_folder = class_folders.get(kaggle_name)
        if src_folder is None:
            continue

        files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

        if kaggle_name in WBC_SUBCLASSES:
            # Va a data/coarse/WBC/ (todas las subclases juntas) Y a data/wbc_fine/<clase>/
            coarse_target = os.path.join(coarse_dir, "WBC")
            fine_target = os.path.join(wbc_dir, es_name)
            os.makedirs(coarse_target, exist_ok=True)
            os.makedirs(fine_target, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src_folder, f), os.path.join(coarse_target, f"{kaggle_name}_{f}"))
                shutil.copy2(os.path.join(src_folder, f), os.path.join(fine_target, f))
            print(f"  · {kaggle_name} -> coarse/WBC ({len(files)} img) + wbc_fine/{es_name} ({len(files)} img)")
        else:
            # "platelet" -> coarse/Plaqueta ; "erythroblast" -> coarse/Precursor_eritroide
            coarse_target = os.path.join(coarse_dir, es_name)
            os.makedirs(coarse_target, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src_folder, f), os.path.join(coarse_target, f))
            print(f"  · {kaggle_name} -> coarse/{es_name} ({len(files)} img)")

    print("\nOrganización completa. Estructura final en data/:")
    print(f"  {coarse_dir}/{{Plaqueta, Precursor_eritroide, WBC}}")
    print(f"  {wbc_dir}/{{Basofilo, Eosinofilo, Granulocito_inmaduro, Linfocito, Monocito, Neutrofilo}}")


if __name__ == "__main__":
    raw_path = download_dataset()
    organize_dataset(raw_path)
