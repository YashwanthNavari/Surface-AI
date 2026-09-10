"""
Image Preprocessing Pipeline
Provides standard resizing, RGB channel alignment, and model-specific normalization routines.
"""
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from src.config import IMAGE_SIZE, PROJECT_ROOT

def load_and_preprocess_image(image_input, target_size=IMAGE_SIZE):
    """
    Loads an image from filepath, raw bytes, or numpy array.
    Converts to 3-channel RGB and resizes to target_size (224, 224).
    Returns numpy array of float32 with pixel range [0, 255].
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if (img.shape[0], img.shape[1]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img.astype(np.float32)

def normalize_for_custom_cnn(image):
    """
    Normalizes pixel intensities to [0.0, 1.0] for the Custom CNN.
    """
    return image / 255.0

def normalize_for_efficientnet(image):
    """
    Prepares input according to EfficientNet requirements (Keras application standard).
    EfficientNetB0 handles its own internal scaling or expects [0, 255].
    Using keras.applications.efficientnet.preprocess_input maintains exact consistency.
    """
    return tf.keras.applications.efficientnet.preprocess_input(image)
