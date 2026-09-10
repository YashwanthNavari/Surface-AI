"""
Unit Tests for Image Preprocessing Pipeline
"""
import numpy as np
from src.config import IMAGE_SIZE
from src.data.preprocessing import (
    load_and_preprocess_image,
    normalize_for_custom_cnn,
    normalize_for_efficientnet,
)

def test_load_and_preprocess_image():
    # Synthetic grayscale image
    dummy_gray = np.random.randint(0, 256, (150, 180), dtype=np.uint8)
    processed = load_and_preprocess_image(dummy_gray, target_size=IMAGE_SIZE)

    assert processed.shape == (*IMAGE_SIZE, 3)
    assert processed.dtype == np.float32

def test_normalization_custom_cnn():
    dummy_img = np.ones((*IMAGE_SIZE, 3), dtype=np.float32) * 255.0
    norm = normalize_for_custom_cnn(dummy_img)

    assert np.allclose(norm, 1.0)
    assert norm.min() >= 0.0 and norm.max() <= 1.0

def test_normalization_efficientnet():
    dummy_img = np.ones((*IMAGE_SIZE, 3), dtype=np.float32) * 128.0
    norm = normalize_for_efficientnet(dummy_img)

    assert norm.shape == (*IMAGE_SIZE, 3)

if __name__ == "__main__":
    test_load_and_preprocess_image()
    test_normalization_custom_cnn()
    test_normalization_efficientnet()
    print("All preprocessing tests passed!")
