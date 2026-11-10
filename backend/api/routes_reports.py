import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.session import get_db
from backend.database.repository import AnalysisRepository

router = APIRouter()

@router.get("/reports/{session_id}/pdf")
async def download_pdf_report(session_id: str, db: Session = Depends(get_db)):
    repo = AnalysisRepository(db)
    session = repo.get_session(session_id)
    if not session or not session.pdf_report_path:
        # Check if file exists directly matching session id
        potential_path = os.path.join(settings.REPORTS_DIR, f"VisionIQ_Inspection_Report_{session_id[:8]}.pdf")
        if os.path.exists(potential_path):
            return FileResponse(
                path=potential_path,
                filename=os.path.basename(potential_path),
                media_type="application/pdf"
            )
        raise HTTPException(status_code=404, detail="PDF report not found for this session.")

    if not os.path.exists(session.pdf_report_path):
        raise HTTPException(status_code=404, detail="Report file missing from disk.")

    return FileResponse(
        path=session.pdf_report_path,
        filename=os.path.basename(session.pdf_report_path),
        media_type="application/pdf"
    )
