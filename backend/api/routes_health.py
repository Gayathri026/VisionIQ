import os
from fastapi import APIRouter
from backend.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "models": {
            "yolo_detector": os.path.exists(settings.MODEL_YOLO_PATH),
            "ppe_model": os.path.exists(settings.MODEL_PPE_PATH),
            "defect_model": os.path.exists(settings.MODEL_DEFECT_PATH)
        },
        "routing_parameters": {
            "alpha_relevance": settings.ROUTING_ALPHA,
            "beta_cost": settings.ROUTING_BETA,
            "gamma_latency": settings.ROUTING_GAMMA,
            "delta_capability": settings.ROUTING_DELTA,
            "threshold_tau": settings.ROUTING_THRESHOLD
        }
    }
