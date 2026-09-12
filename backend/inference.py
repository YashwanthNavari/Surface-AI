"""
Multi-Model Inference Service with Consensus Voting and Confidence Arbitration
Loads Custom CNN, EfficientNetB0 Frozen, and EfficientNetB0 Fine-Tuned.
Evaluates consensus agreement across models to trigger automated verification or manual inspection warnings.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    MODELS_DIR,
    CLASSES,
    CONFIG,
)
from src.data.preprocessing import load_and_preprocess_image

# Global Model Registry
_MODELS: Dict[str, keras.Model] = {}

def get_models():
    """
    Lazy loads and caches all 3 models in memory.
    """
    global _MODELS
    if not _MODELS:
        custom_path = MODELS_DIR / "custom_cnn.keras"
        frozen_path = MODELS_DIR / "efficientnet_frozen.keras"
        finetuned_path = MODELS_DIR / "efficientnet_finetuned.keras"

        if custom_path.exists():
            print("Loading Custom CNN...")
            _MODELS["custom_cnn"] = keras.models.load_model(str(custom_path))
        if frozen_path.exists():
            print("Loading EfficientNetB0 Frozen...")
            _MODELS["efficientnet_frozen"] = keras.models.load_model(str(frozen_path))
        if finetuned_path.exists():
            print("Loading EfficientNetB0 Fine-Tuned...")
            _MODELS["efficientnet_finetuned"] = keras.models.load_model(str(finetuned_path))

    return _MODELS

def predict_single_image(image_bytes: bytes, filename: str = "upload.jpg") -> Dict:
    """
    Runs multi-model inference on uploaded image bytes.
    Computes ensemble consensus and recommendation.
    """
    import time
    start_time = time.perf_counter()

    models = get_models()
    if not models:
        raise RuntimeError("No models loaded. Ensure model weights exist in models/.")

    # Load RGB image (0-255)
    img_rgb = load_and_preprocess_image(image_bytes)

    # Preprocess inputs
    custom_tensor = np.expand_dims(img_rgb / 255.0, axis=0)
    effnet_tensor = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(img_rgb.copy()), axis=0)

    model_results = {}
    predicted_classes = []

    # 1. Custom CNN
    if "custom_cnn" in models:
        c_probs = models["custom_cnn"].predict(custom_tensor, verbose=0)[0]
        c_idx = int(np.argmax(c_probs))
        c_cls = CLASSES[c_idx]
        c_conf = float(c_probs[c_idx])
        predicted_classes.append(c_cls)
        model_results["custom_cnn"] = {
            "predicted_class": c_cls,
            "confidence": round(c_conf, 4),
            "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(CLASSES, c_probs)},
        }

    # 2. EfficientNet Frozen
    if "efficientnet_frozen" in models:
        fz_probs = models["efficientnet_frozen"].predict(effnet_tensor, verbose=0)[0]
        fz_idx = int(np.argmax(fz_probs))
        fz_cls = CLASSES[fz_idx]
        fz_conf = float(fz_probs[fz_idx])
        predicted_classes.append(fz_cls)
        model_results["efficientnet_frozen"] = {
            "predicted_class": fz_cls,
            "confidence": round(fz_conf, 4),
            "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(CLASSES, fz_probs)},
        }

    # 3. EfficientNet Fine-Tuned (Top Authority)
    if "efficientnet_finetuned" in models:
        ft_probs = models["efficientnet_finetuned"].predict(effnet_tensor, verbose=0)[0]
        ft_idx = int(np.argmax(ft_probs))
        ft_cls = CLASSES[ft_idx]
        ft_conf = float(ft_probs[ft_idx])
        predicted_classes.append(ft_cls)
        model_results["efficientnet_finetuned"] = {
            "predicted_class": ft_cls,
            "confidence": round(ft_conf, 4),
            "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(CLASSES, ft_probs)},
        }

    # Consensus and Arbitration Logic
    total_models = len(predicted_classes)
    # The Fine-Tuned model is our highest accuracy authority, fallback to majority
    primary_pred = model_results.get("efficientnet_finetuned", {}).get("predicted_class") or predicted_classes[0]
    primary_conf = model_results.get("efficientnet_finetuned", {}).get("confidence") or model_results[list(model_results.keys())[0]]["confidence"]

    # Count how many models agree with primary_pred
    consensus_count = predicted_classes.count(primary_pred)

    if consensus_count == total_models:
        consensus_status = "CONSENSUS_AGREED"
    else:
        consensus_status = "MODEL_DISAGREEMENT"

    # Confidence Tier Assignment
    high_th = CONFIG.get("serving", {}).get("confidence_threshold_high", 0.85)
    med_th = CONFIG.get("serving", {}).get("confidence_threshold_medium", 0.65)

    if primary_conf >= high_th and consensus_status == "CONSENSUS_AGREED":
        confidence_tier = "HIGH CONFIDENCE"
        recommendation = "Automated Pass"
    elif primary_conf >= med_th and consensus_count >= 2:
        confidence_tier = "MODERATE CONFIDENCE"
        recommendation = "Automated Pass (Standard Inspection)"
    else:
        confidence_tier = "LOW CONFIDENCE" if primary_conf < med_th else "DISPUTED CONSENSUS"
        recommendation = "Manual Inspection Recommended"

    end_time = time.perf_counter()
    inference_time_ms = round((end_time - start_time) * 1000, 2)

    return {
        "primary_prediction": primary_pred,
        "primary_confidence": primary_conf,
        "consensus_count": consensus_count,
        "total_models": total_models,
        "consensus_status": consensus_status,
        "confidence_tier": confidence_tier,
        "recommendation": recommendation,
        "models": model_results,
        "inference_time_ms": inference_time_ms,
    }
