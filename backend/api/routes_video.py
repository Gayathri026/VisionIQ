import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.config import settings
from backend.services.video_service import video_service

router = APIRouter()

@router.post("/video/process")
async def process_video_endpoint(
    video: UploadFile = File(...),
    query: str = Form(...),
    sample_fps: float = Form(1.0),
    max_frames: int = Form(25)
):
    if not (video.content_type.startswith("video/") or video.filename.endswith((".mp4", ".avi", ".mov", ".mkv"))):
        raise HTTPException(status_code=400, detail="Provided file must be a video format.")

    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    file_ext = os.path.splitext(video.filename)[1] or ".mp4"
    unique_name = f"video_{uuid.uuid4().hex[:8]}{file_ext}"
    video_path = os.path.join(settings.UPLOADS_DIR, unique_name)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    try:
        result = video_service.process_video(
            video_path=video_path,
            query=query,
            sample_fps=sample_fps,
            max_frames=max_frames
        )
        return {
            "status": "success",
            "message": "Video processing completed.",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(e)}")
