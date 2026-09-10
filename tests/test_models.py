"""
Unit Tests for Deep Learning Architectures
"""
import numpy as np
import tensorflow as tf
from src.config import NUM_CLASSES
from src.models.custom_cnn import build_custom_cnn
from src.models.efficientnet import build_efficientnet_frozen

def test_custom_cnn_forward():
    model = build_custom_cnn(num_classes=NUM_CLASSES)
    dummy_batch = np.random.rand(2, 224, 224, 3).astype(np.float32)

    outputs = model(dummy_batch, training=False)
    assert outputs.shape == (2, NUM_CLASSES)
    assert np.allclose(np.sum(outputs.numpy(), axis=1), 1.0, atol=1e-5)

def test_efficientnet_frozen_forward():
    model, _ = build_efficientnet_frozen(num_classes=NUM_CLASSES)
    dummy_batch = np.random.rand(2, 224, 224, 3).astype(np.float32)

    outputs = model(dummy_batch, training=False)
    assert outputs.shape == (2, NUM_CLASSES)
    assert np.allclose(np.sum(outputs.numpy(), axis=1), 1.0, atol=1e-5)

if __name__ == "__main__":
    test_custom_cnn_forward()
    test_efficientnet_frozen_forward()
    print("All model architecture tests passed!")
