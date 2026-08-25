"""
Pipeline de datos: carga desde carpetas, augmentación y utilidades de split.

Usa tf.data.Dataset (más eficiente y "sólido" para producción/academia que
ImageDataGenerator, que está deprecado).
"""
import argparse
import os
import random
import shutil

import tensorflow as tf

from src.config import IMG_SIZE, BATCH_SIZE, SEED, VALIDATION_SPLIT, TEST_SPLIT

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------
# Augmentación (capas nativas de Keras -> se ejecutan en GPU, sin overhead)
# ----------------------------------------------------------------------
def build_augmentation_pipeline():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def load_datasets(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=True):
    """
    Carga un directorio con subcarpetas = clases y devuelve
    (train_ds, val_ds, class_names) listos para .fit().
    Espera estructura: data_dir/<clase>/*.jpg
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    if augment:
        aug = build_augmentation_pipeline()
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Normalización [0,1] (los modelos de transfer learning aplican su propio preprocess_input)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# ----------------------------------------------------------------------
# Utilidad de línea de comandos: separar un dataset plano en train/val/test físicamente
# (útil si prefieres tener carpetas ya divididas en disco en vez de usar validation_split)
# ----------------------------------------------------------------------
def split_dataset(src_folder: str, out_folder: str,
                   val_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=SEED):
    random.seed(seed)
    classes = [d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))]

    for split in ("train", "val", "test"):
        for c in classes:
            os.makedirs(os.path.join(out_folder, split, c), exist_ok=True)

    for c in classes:
        class_dir = os.path.join(src_folder, c)
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(files)

        n = len(files)
        n_test = int(n * test_split)
        n_val = int(n * val_split)

        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]

        for split_name, split_files in (("train", train_files), ("val", val_files), ("test", test_files)):
            for f in split_files:
                shutil.copy2(
                    os.path.join(class_dir, f),
                    os.path.join(out_folder, split_name, c, f),
                )

    print(f"Split completo. Clases: {classes}")
    print(f"Salida en: {out_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide un dataset plano en train/val/test")
    parser.add_argument("--split_folder", required=True, help="Carpeta origen con subcarpetas por clase")
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    args = parser.parse_args()
    split_dataset(args.split_folder, args.out)
