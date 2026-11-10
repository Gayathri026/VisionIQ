import os
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "VisionIQ: Adaptive Multi-Agent Vision Intelligence"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///./visioniq.db"
    
    # Storage directories
    STORAGE_DIR: str = "storage"
    UPLOADS_DIR: str = "storage/uploads"
    REPORTS_DIR: str = "storage/reports"
    ANNOTATIONS_DIR: str = "storage/annotations"
    RAG_DIR: str = "storage/rag"
    
    # Model configuration
    MODEL_YOLO_PATH: str = "backend/models/yolo11n.pt"
    MODEL_PPE_PATH: str = "backend/models/yolo_ppe.pt"
    MODEL_DEFECT_PATH: str = "backend/models/defect_model.pt"
    
    # LLM Settings (optional - deterministic fallback provided)
    OPENAI_API_KEY: str | None = None
    LLM_MODEL: str = "gpt-4o"
    
    # Research Routing Parameters (Phase-I calibrated weights)
    # S(a, q) = alpha * R(a,q) + delta * E(a) - beta * C(a) - gamma * L(a)
    ROUTING_ALPHA: float = 0.50  # Query relevance weight
    ROUTING_BETA: float = 0.20   # Computational cost penalty weight
    ROUTING_GAMMA: float = 0.15  # Latency penalty weight
    ROUTING_DELTA: float = 0.15  # Expected evidence capability weight
    ROUTING_THRESHOLD: float = 0.45  # Calibrated operating threshold (Phase-I initial hypothesis was 0.50)
    
    # Per-Agent baseline normalized costs C(a) in [0, 1]
    AGENT_COSTS: Dict[str, float] = {
        "intent": 0.05,
        "object": 0.15,
        "ppe": 0.20,
        "ocr": 0.30,
        "defect": 0.35,
        "scene": 0.10,
        "evidence_validation": 0.10,
        "rag": 0.25,
        "rca": 0.20,
        "recommendation": 0.15,
        "report": 0.10
    }
    
    # Per-Agent baseline normalized expected latencies L(a) in [0, 1]
    AGENT_LATENCIES: Dict[str, float] = {
        "intent": 0.05,
        "object": 0.18,
        "ppe": 0.22,
        "ocr": 0.40,
        "defect": 0.35,
        "scene": 0.12,
        "evidence_validation": 0.08,
        "rag": 0.25,
        "rca": 0.20,
        "recommendation": 0.15,
        "report": 0.15
    }
    
    # Per-Agent expected evidence capability E(a) in [0, 1]
    AGENT_CAPABILITIES: Dict[str, float] = {
        "object": 0.85,
        "ppe": 0.90,
        "ocr": 0.80,
        "defect": 0.85,
        "scene": 0.75,
        "evidence_validation": 0.90,
        "rag": 0.85,
        "rca": 0.80,
        "recommendation": 0.85,
        "report": 0.95
    }

    # Limits
    MAX_IMAGE_SIZE_MB: int = 15
    MAX_VIDEO_SIZE_MB: int = 100
    DEFAULT_VIDEO_SAMPLE_FPS: float = 1.0

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure directories exist
for path in [settings.STORAGE_DIR, settings.UPLOADS_DIR, settings.REPORTS_DIR, 
             settings.ANNOTATIONS_DIR, settings.RAG_DIR]:
    os.makedirs(path, exist_ok=True)
