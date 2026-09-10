"""
Pydantic Data Schemas for API Request and Response Serialization
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ModelPrediction(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]

class InspectionResponse(BaseModel):
    primary_prediction: str
    primary_confidence: float
    consensus_count: int
    total_models: int
    consensus_status: str  # "CONSENSUS_AGREED" or "MODEL_DISAGREEMENT"
    confidence_tier: str   # "HIGH CONFIDENCE", "MODERATE CONFIDENCE", "LOW CONFIDENCE"
    recommendation: str    # "Automated Pass" or "Manual Inspection Recommended"
    models: Dict[str, ModelPrediction]
    timestamp: str
    inspection_id: Optional[int] = None

class GradCAMResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    heatmap_base64: str
    overlay_base64: str
    original_base64: str

class InspectionHistoryItem(BaseModel):
    id: int
    timestamp: str
    filename: str
    primary_prediction: str
    primary_confidence: float
    consensus_status: str
    recommendation: str

class MetricItem(BaseModel):
    model: str
    type: str
    accuracy: float
    macro_f1: float
    latency_ms: float
    throughput_fps: float
    parameters: str
