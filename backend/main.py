"""
FastAPI Application Entry Point
Exposes REST endpoints for:
- /api/v1/predict (Multi-model prediction, consensus agreement, and arbitration)
- /api/v1/explain (Grad-CAM heatmaps & overlays in base64)
- /api/v1/metrics (Model comparison tables and per-class metrics)
- /api/v1/history (Inspection audit logs from SQLite)
- /api/v1/samples (Pre-loaded test defect samples for quick demonstration)
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.database import get_recent_inspections, log_inspection
from backend.gradcam_service import explain_image
from backend.inference import get_models, predict_single_image
from backend.schemas import (
    GradCAMResponse,
    InspectionHistoryItem,
    InspectionResponse,
    MetricItem,
)
from src.config import (
    CLASSES,
    METRICS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)

app = FastAPI(
    title="Manufacturing Surface Defect Detection API",
    description="Industrial AI Vision Inspection Backend using Custom CNN and Transfer Learning (EfficientNetB0).",
    version="1.0.0",
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Pre-loading models into GPU/CPU memory...")
    try:
        models = get_models()
        print(f"Loaded models: {list(models.keys())}")
    except Exception as e:
        print(f"Warning during model loading on startup: {e}")

@app.get("/")
def read_root():
    return {
        "system": "Industrial Surface Defect Inspection Console",
        "status": "ONLINE",
        "supported_classes": CLASSES,
        "models_available": list(get_models().keys()),
    }

@app.post("/api/v1/predict", response_model=InspectionResponse)
async def predict_defect(file: UploadFile = File(...)):
    """
    Analyzes an uploaded manufacturing component image across Custom CNN,
    EfficientNetB0 Frozen, and EfficientNetB0 Fine-Tuned.
    Computes ensemble consensus agreement and issues inspection recommendation.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    image_bytes = await file.read()
    try:
        result = predict_single_image(image_bytes, filename=file.filename)
        # Log to SQLite database
        inspection_id, timestamp_str = log_inspection(
            filename=file.filename,
            primary_prediction=result["primary_prediction"],
            primary_confidence=result["primary_confidence"],
            consensus_count=result["consensus_count"],
            total_models=result["total_models"],
            consensus_status=result["consensus_status"],
            confidence_tier=result["confidence_tier"],
            recommendation=result["recommendation"],
            models_dict=result["models"],
        )
        result["inspection_id"] = inspection_id
        result["timestamp"] = timestamp_str
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/v1/explain", response_model=GradCAMResponse)
async def explain_defect(
    file: UploadFile = File(...),
    model_name: str = Form("efficientnet_finetuned"),
):
    """
    Generates Grad-CAM visual activation heatmaps and overlay for any uploaded image
    using the specified model ('custom_cnn', 'efficientnet_frozen', or 'efficientnet_finetuned').
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid image."
        )

    image_bytes = await file.read()
    try:
        explanation = explain_image(image_bytes, selected_model=model_name)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation error: {str(e)}")

@app.get("/api/v1/metrics")
def get_benchmarks():
    """
    Returns the master model comparison table and per-class evaluation data.
    """
    master_csv = RESULTS_DIR / "model_comparison.csv"
    if not master_csv.exists():
        return {"models": [], "per_class": {}}

    df = pd.read_csv(master_csv).fillna("N/A")
    models_data = df.to_dict(orient="records")

    per_class_data = {}
    for prefix in ["custom_cnn", "efficientnet_frozen", "efficientnet_finetuned"]:
        p_csv = METRICS_DIR / f"{prefix}_per_class.csv"
        if p_csv.exists():
            per_class_data[prefix] = pd.read_csv(p_csv).fillna("N/A").to_dict(orient="records")

    return {
        "models": models_data,
        "per_class": per_class_data,
    }

@app.get("/api/v1/history", response_model=List[InspectionHistoryItem])
def get_history(limit: int = 25):
    """
    Retrieves recent inspection audit records from SQLite.
    """
    records = get_recent_inspections(limit=limit)
    return [
        InspectionHistoryItem(
            id=r["id"],
            timestamp=r["timestamp"],
            filename=r["filename"],
            primary_prediction=r["primary_prediction"],
            primary_confidence=r["primary_confidence"],
            consensus_count=r["consensus_count"],
            total_models=r["total_models"],
            consensus_status=r["consensus_status"],
            recommendation=r["recommendation"],
        )
        for r in records
    ]

@app.get("/api/v1/samples")
def get_sample_test_images():
    """
    Provides a curated list of representative test set images across all 6 classes
    for quick demonstration and testing in the frontend.
    """
    test_csv = PROJECT_ROOT / "data/splits/test.csv"
    if not test_csv.exists():
        return []

    df = pd.read_csv(test_csv)
    samples = []
    # Pick 2 samples per class
    for cls_name in CLASSES:
        cls_sub = df[df["class"] == cls_name].head(2)
        for _, row in cls_sub.iterrows():
            samples.append({
                "class": cls_name,
                "filename": row["filename"],
                "filepath": row["filepath"],
                "url": f"/api/v1/sample-image/{cls_name}/{row['filename']}",
            })
    return samples

@app.get("/api/v1/sample-image/{cls_name}/{filename}")
def serve_sample_image(cls_name: str, filename: str):
    """
    Serves a sample test image directly from data/raw/NEU-CLS.
    """
    img_path = PROJECT_ROOT / "data" / "raw" / "NEU-CLS" / cls_name / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    
    from mimetypes import guess_type
    media_type, _ = guess_type(str(img_path))
    
    return FileResponse(
        str(img_path),
        media_type=media_type or "application/octet-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
