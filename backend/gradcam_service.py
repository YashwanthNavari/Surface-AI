"""
Grad-CAM Service for API Serving
Generates base64-encoded PNG heatmaps and overlays for real-time frontend inspection.
"""
import base64
import io
from typing import Dict
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from backend.inference import get_models
from src.config import CLASSES
from src.data.preprocessing import load_and_preprocess_image
from src.explainability.gradcam import make_gradcam_heatmap, overlay_heatmap

def numpy_to_base64(img_array: np.ndarray) -> str:
    """
    Converts RGB uint8 numpy array to base64 encoded data URI.
    """
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def explain_image(image_bytes: bytes, selected_model: str = "efficientnet_finetuned") -> Dict:
    """
    Generates Grad-CAM visual explanation for the uploaded image using the selected model.
    Returns predicted class, confidence, and base64 strings for:
    - Original RGB
    - Heatmap
    - Superimposed Overlay
    """
    models = get_models()
    if selected_model not in models:
        # Fallback to available model
        selected_model = list(models.keys())[0]

    model = models[selected_model]
    model_type = "custom" if "custom" in selected_model else "efficientnet"

    # Preprocess
    img_rgb = load_and_preprocess_image(image_bytes)
    img_uint8 = np.uint8(img_rgb)

    if model_type == "custom":
        img_tensor = np.expand_dims(img_rgb / 255.0, axis=0)
    else:
        img_tensor = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(img_rgb.copy()), axis=0)

    # Compute heatmap
    heatmap, pred_idx, confidence = make_gradcam_heatmap(img_tensor, model)
    color_hm, overlay = overlay_heatmap(img_uint8, heatmap, alpha=0.45)

    return {
        "model_name": selected_model,
        "predicted_class": CLASSES[pred_idx],
        "confidence": round(float(confidence), 4),
        "original_base64": numpy_to_base64(img_uint8),
        "heatmap_base64": numpy_to_base64(color_hm),
        "overlay_base64": numpy_to_base64(overlay),
    }
