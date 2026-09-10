"""
High-Performance tf.data Dataset Pipeline
Constructs optimized, thread-safe input pipelines for training, validation, and testing.
Uses native TensorFlow I/O operations for zero-GIL multi-threaded data loading.
"""
from pathlib import Path
import pandas as pd
import tensorflow as tf
from src.config import (
    SPLITS_DIR,
    NUM_CLASSES,
    IMAGE_SIZE,
    PROJECT_ROOT,
    CONFIG,
)
from src.data.augmentation import get_augmentation_pipeline

def parse_image_and_label(filepath, label, model_type="custom"):
    """
    Native TensorFlow decoding and preprocessing function.
    Reads JPEG file, decodes to RGB, resizes, and normalizes according to model architecture.
    """
    image_bytes = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")

    if model_type == "custom":
        image = image / 255.0  # Normalize to [0, 1]
    elif model_type == "efficientnet":
        image = tf.keras.applications.efficientnet.preprocess_input(image)

    # One-hot encode label for CategoricalCrossentropy
    one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, one_hot_label

def build_dataset_from_manifest(
    manifest_csv_path,
    model_type="custom",
    batch_size=None,
    shuffle=False,
    augment=False,
    seed=42,
):
    """
    Builds a batched and prefetched tf.data.Dataset from a split CSV manifest.
    """
    if batch_size is None:
        batch_size = CONFIG["training"]["batch_size"]

    df = pd.read_csv(manifest_csv_path)
    # Convert relative paths to absolute paths
    filepaths = [str(PROJECT_ROOT / p) for p in df["filepath"].tolist()]
    labels = df["label"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=seed, reshuffle_each_iteration=True)

    # Parallel mapping using autotuned CPU threads
    ds = ds.map(
        lambda fp, lbl: parse_image_and_label(fp, lbl, model_type=model_type),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Apply data augmentation only if requested (and during training)
    if augment:
        augmentation_layers = get_augmentation_pipeline()
        ds = ds.map(
            lambda img, lbl: (augmentation_layers(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds

def get_train_val_test_datasets(
    model_type="custom",
    batch_size=None,
    augment_training=True,
    seed=42,
):
    """
    Convenience function returning (train_ds, val_ds, test_ds).
    Guarantees validation and test datasets are NEVER augmented and NEVER shuffled.
    """
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"

    for csv_file in [train_csv, val_csv, test_csv]:
        if not csv_file.exists():
            raise FileNotFoundError(f"Manifest not found: {csv_file}. Please run split.py first.")

    train_ds = build_dataset_from_manifest(
        train_csv,
        model_type=model_type,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_training,
        seed=seed,
    )

    val_ds = build_dataset_from_manifest(
        val_csv,
        model_type=model_type,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        seed=seed,
    )

    test_ds = build_dataset_from_manifest(
        test_csv,
        model_type=model_type,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        seed=seed,
    )

    return train_ds, val_ds, test_ds
