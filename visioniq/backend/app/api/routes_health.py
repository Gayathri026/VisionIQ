import os
from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.contracts import HealthResponse
from backend.app.database.session import engine

router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="System Health & Diagnostic Check")
async def health_check():
    """
    Diagnostic endpoint reporting:
    - Operational framework status and version
    - Database engine connectivity
    - Model registry presence (YOLOv11, PPE, Defect)
    - Active routing formula parameters (Equation 1)
    """
    db_ok = True
    try:
        with engine.connect() as conn:
            pass
    except Exception:
        db_ok = False

    return HealthResponse(
        status="online",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        database_connected=db_ok,
        models={
            "yolo_detector": os.path.exists(settings.MODEL_YOLO_PATH),
            "ppe_model": os.path.exists(settings.MODEL_PPE_PATH),
            "defect_model": os.path.exists(settings.MODEL_DEFECT_PATH)
        },
        routing_parameters={
            "alpha_relevance": settings.ROUTING_ALPHA,
            "beta_cost": settings.ROUTING_BETA,
            "gamma_latency": settings.ROUTING_GAMMA,
            "delta_capability": settings.ROUTING_DELTA,
            "threshold_tau": settings.ROUTING_THRESHOLD
        }
    )
