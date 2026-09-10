"""
Integration Tests for FastAPI Backend Endpoints
"""
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "crazing" in data["supported_classes"]

def test_history_endpoint():
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_samples_endpoint():
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_endpoint_synthetic():
    # Create dummy test image in memory
    dummy_img = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    img_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/api/v1/predict",
        files={"file": ("test_surface.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "primary_prediction" in data
    assert "consensus_status" in data
    assert "models" in data

if __name__ == "__main__":
    test_root_endpoint()
    test_history_endpoint()
    test_samples_endpoint()
    print("API tests passed!")
