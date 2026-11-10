import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.session import get_db
from backend.database.repository import AnalysisRepository
from backend.orchestrator.graph import vision_iq_app

router = APIRouter()

@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Provided file must be an image format (JPEG, PNG, etc.).")

    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    file_ext = os.path.splitext(image.filename)[1] or ".jpg"
    unique_name = f"upload_{uuid.uuid4().hex[:8]}{file_ext}"
    image_path = os.path.join(settings.UPLOADS_DIR, unique_name)

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded image: {str(e)}")

    initial_state = {
        "image_path": image_path,
        "user_query": query,
        "input_type": "image"
    }

    try:
        final_state = vision_iq_app.invoke(initial_state)
        response_data = final_state.get("final_response", {})
        
        # Attach public URLs for images
        if response_data.get("input_path"):
            rel_in = os.path.relpath(response_data["input_path"], settings.STORAGE_DIR).replace("\\", "/")
            response_data["input_url"] = f"/storage/{rel_in}"
            
        if response_data.get("annotated_path"):
            rel_ann = os.path.relpath(response_data["annotated_path"], settings.STORAGE_DIR).replace("\\", "/")
            response_data["annotated_url"] = f"/storage/{rel_ann}"
            
        if response_data.get("pdf_report_path"):
            rel_pdf = os.path.relpath(response_data["pdf_report_path"], settings.STORAGE_DIR).replace("\\", "/")
            response_data["pdf_url"] = f"/storage/{rel_pdf}"

        return {
            "status": "success",
            "message": "Analysis completed successfully.",
            "data": response_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration execution failed: {str(e)}")

@router.get("/history")
async def get_history(limit: int = 50, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    sessions = repo.list_sessions(limit=limit)
    return {
        "status": "success",
        "total": len(sessions),
        "sessions": [s.to_dict() for s in sessions]
    }

@router.get("/session/{session_id}")
async def get_session_details(session_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    session = repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session record not found")
    return {
        "status": "success",
        "session": session.to_dict()
    }
