"""
EfficientNetB0 Transfer Learning and Fine-Tuning Architectures
Pretrained on ImageNet.
Provides:
1. Frozen Backbone Feature Extractor (trainable classification head)
2. Fine-Tuning Model (unfreezing upper convolutional blocks while keeping BN layers frozen)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import IMAGE_SIZE, NUM_CLASSES

def build_efficientnet_frozen(
    input_shape=(*IMAGE_SIZE, 3),
    num_classes=NUM_CLASSES,
    dropout_rate=0.4,
    name="efficientnet_frozen",
):
    """
    Constructs an EfficientNetB0 model with frozen ImageNet backbone.
    Only the classification head is trainable.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Load base EfficientNetB0 without top classifier
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    # Freeze entire backbone
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.Dense(128, activation="relu", name="head_dense1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model, base_model

def build_efficientnet_finetuned(
    frozen_model_path,
    unfreeze_layers=25,
    name="efficientnet_finetuned",
):
    """
    Loads a trained frozen model and unfreezes the top N layers for fine-tuning.
    Crucial Deep Learning Best Practice: Keeps all BatchNormalization layers in inference mode
    to prevent destruction of learned mean/variance statistics.
    """
    model = keras.models.load_model(frozen_model_path)

    # Find the base_model (or unfreeze the top layers directly)
    # If the model layers contain the base model or are sequential:
    trainable_count = 0
    total_layers = len(model.layers)

    # If base model is a nested layer:
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or "efficientnet" in layer.name.lower():
            base_model = layer
            break

    target_layers = base_model.layers if base_model else model.layers

    # Set all layers to trainable except BatchNormalization
    for layer in target_layers[-unfreeze_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_count += 1

    if base_model:
        base_model.trainable = True
        # Keep non-target layers frozen
        for layer in target_layers[:-unfreeze_layers]:
            layer.trainable = False

    print(f"Fine-tuning configuration: Unfroze top {unfreeze_layers} layers ({trainable_count} non-BN layers trainable)")
    return model
