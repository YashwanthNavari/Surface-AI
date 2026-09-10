"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) Module
Generates visual explanations for defect classification decisions across:
- Custom CNN (targets 'conv4')
- EfficientNetB0 Frozen and Fine-Tuned (targets 'top_activation')
Produces normalized heatmaps and colormap overlays for visual verification.
"""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    GRADCAM_DIR,
    CLASSES,
    IMAGE_SIZE,
)
from src.data.preprocessing import load_and_preprocess_image

def find_target_conv_layer(model):
    """
    Automatically detects the appropriate final convolutional or activation layer for Grad-CAM.
    """
    # For custom CNN
    for layer in reversed(model.layers):
        if layer.name == "conv4" or (isinstance(layer, keras.layers.Conv2D)):
            return layer.name

    # For EfficientNetB0
    for layer in reversed(model.layers):
        if layer.name in ["top_activation", "top_conv"]:
            return layer.name

    # Generic fallback: last layer with 4D output
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name

    raise ValueError("Could not find a convolutional layer for Grad-CAM.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a normalized Grad-CAM heatmap for a single preprocessed image (1, H, W, 3).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_target_conv_layer(model)

    # Construct gradient model mapping input image to target conv output and final predictions
    target_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[target_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of winning class with respect to the output feature map of target conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients: importance weights of each feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by 'how important this channel is'
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only features that have positive influence on the target class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), int(pred_index), float(predictions[0][pred_index])

def overlay_heatmap(original_img_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap onto the original RGB image.
    original_img_rgb: numpy uint8 (H, W, 3) with range [0, 255]
    heatmap: numpy float32 (H, W) with range [0, 1]
    """
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply color map
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap on the original image
    superimposed_img = np.uint8(color_heatmap * alpha + original_img_rgb * (1 - alpha))
    return color_heatmap, superimposed_img

def generate_and_save_gradcam_trio(
    model,
    image_input,
    output_filename="gradcam_sample.png",
    model_type="custom",
    title_prefix="Grad-CAM",
):
    """
    Processes an image, computes Grad-CAM, and saves a 3-panel figure:
    [ Original | Heatmap | Superimposed Overlay ]
    """
    # Load original RGB (0-255)
    img_rgb = load_and_preprocess_image(image_input)
    img_uint8 = np.uint8(img_rgb)

    # Normalize for inference
    if model_type == "custom":
        img_tensor = np.expand_dims(img_rgb / 255.0, axis=0)
    else:
        img_tensor = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(img_rgb.copy()), axis=0)

    heatmap, pred_idx, confidence = make_gradcam_heatmap(img_tensor, model)
    color_hm, overlay = overlay_heatmap(img_uint8, heatmap, alpha=0.45)

    pred_class = CLASSES[pred_idx].replace("_", " ").title()

    # Create 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=300)

    axes[0].imshow(img_uint8)
    axes[0].set_title("Input Surface Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(color_hm)
    axes[1].set_title(f"Class Activation Heatmap", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Defect Localization Overlay\n({pred_class}: {confidence*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(f"{title_prefix} — Explainability Visualization", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = GRADCAM_DIR / output_filename
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM visualization to {out_file}")

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "output_image": str(out_file),
    }
