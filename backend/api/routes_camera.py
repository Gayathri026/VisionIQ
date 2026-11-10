from fastapi import APIRouter, HTTPException
from backend.schemas.analysis import CameraFrameRequest
from backend.services.camera_service import camera_service

router = APIRouter()

@router.post("/camera/frame")
async def process_live_frame(payload: CameraFrameRequest):
    if not payload.frame_base64:
        raise HTTPException(status_code=400, detail="Missing frame_base64 data")

    try:
        result = camera_service.process_camera_frame(
            frame_base64=payload.frame_base64,
            query=payload.query,
            timestamp=payload.timestamp
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera frame processing failed: {str(e)}")
