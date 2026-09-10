"""
Data Augmentation Pipeline
Provides realistic, moderate augmentations suitable for industrial surface materials:
flips, slight rotations, minor zooms, subtle translations, and contrast shifts.
Does NOT perform extreme distortion that destroys defect morphology.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import CONFIG

def get_augmentation_pipeline():
    """
    Returns a Keras Sequential layer containing data augmentation operations.
    Only active during training.
    """
    aug_cfg = CONFIG.get("augmentation", {})
    rotation = aug_cfg.get("random_rotation", 0.05)
    zoom = aug_cfg.get("random_zoom", 0.08)
    trans_h = aug_cfg.get("random_translation_h", 0.08)
    trans_w = aug_cfg.get("random_translation_w", 0.08)

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="aug_random_flip"),
            layers.RandomRotation(rotation, fill_mode="reflect", name="aug_random_rotation"),
            layers.RandomZoom(height_factor=(-zoom, zoom), width_factor=(-zoom, zoom), fill_mode="reflect", name="aug_random_zoom"),
            layers.RandomTranslation(height_factor=trans_h, width_factor=trans_w, fill_mode="reflect", name="aug_random_translation"),
            layers.RandomContrast(0.12, name="aug_random_contrast"),
        ],
        name="data_augmentation",
    )
