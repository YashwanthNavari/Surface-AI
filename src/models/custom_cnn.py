"""
Custom 4-Stage Convolutional Neural Network Architecture
Modular design supporting ablation studies:
- Stage 1: Conv2D(32, 3x3) -> [BatchNorm] -> ReLU -> MaxPool2D(2x2)
- Stage 2: Conv2D(64, 3x3) -> [BatchNorm] -> ReLU -> MaxPool2D(2x2)
- Stage 3: Conv2D(128, 3x3) -> [BatchNorm] -> ReLU -> MaxPool2D(2x2)
- Stage 4: Conv2D(256, 3x3) -> [BatchNorm] -> ReLU -> MaxPool2D(2x2)
- Head: GlobalAveragePooling2D -> Dense(128) -> ReLU -> [Dropout] -> Dense(6, Softmax)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import IMAGE_SIZE, NUM_CLASSES

def build_custom_cnn(
    input_shape=(*IMAGE_SIZE, 3),
    num_classes=NUM_CLASSES,
    use_batch_norm=True,
    dropout_rate=0.5,
    name="custom_cnn",
):
    """
    Constructs a 4-stage convolutional neural network.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stage 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Stage 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Stage 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Stage 4: 256 filters (Final Conv layer targeted for Grad-CAM)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, name="dense1")(x)
    x = layers.Activation("relu", name="relu_dense")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
