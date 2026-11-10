import os
import re
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.session import get_db
from backend.app.database.repository import AnalysisRepository
from backend.app.reporting.pdf_generator import generate_inspection_pdf

router = APIRouter()

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")

def validate_safe_session_id(session_id: str):
    """
    Guarantees session ID adheres strictly to alphanumeric, hyphen and underscore tokens.
    Rejects path traversal markers such as '..', slashes, and control characters.
    """
    if not session_id or not SESSION_ID_PATTERN.match(session_id) or ".." in session_id:
        raise HTTPException(
            status_code=400,
            detail=f"Security rejection: Invalid session ID format '{session_id}'. Path traversal prohibited."
        )

@router.get("/reports/{session_id}/pdf", summary="Download Production Engineering PDF Report")
async def get_report_pdf(session_id: str, db: Session = Depends(get_db)):
    """
    Serves verifiable ReportLab PDF report containing full visual provenance,
    findings, Equation (1) agent execution traces, RAG citations, and reproducibility metadata.
    Regenerates the PDF on-demand if missing from disk.
    """
    validate_safe_session_id(session_id)
    repo = AnalysisRepository(db)
    session = repo.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis session '{session_id}' not found in registry."
        )

    expected_pdf_name = f"VisionIQ_Inspection_Report_{session_id[:8]}.pdf"
    expected_path = os.path.join(settings.REPORTS_DIR, expected_pdf_name)

    # Check if existing PDF path is valid and within storage directory
    current_pdf_path = session.pdf_report_path or expected_path
    normalized_path = os.path.abspath(current_pdf_path)
    normalized_storage = os.path.abspath(settings.STORAGE_DIR)

    if not normalized_path.startswith(normalized_storage):
        raise HTTPException(
            status_code=400,
            detail="Security rejection: Report path resides outside authorized storage directory."
        )

    if not os.path.exists(normalized_path) or os.path.getsize(normalized_path) == 0:
        # Generate PDF on demand from session data
        logger.info(f"Generating PDF report for session {session_id} on-demand...")
        findings_data = [f.to_dict() for f in session.findings]
        traces_data = [t.to_dict() for t in session.traces]
        rag_data = [r.to_dict() for r in session.rag_sources]

        try:
            normalized_path = generate_inspection_pdf(
                session_id=session.id,
                query=session.user_query,
                input_path=session.input_path,
                annotated_path=session.annotated_path,
                input_type=session.input_type,
                findings=findings_data,
                traces=traces_data,
                rag_references=rag_data,
                evidence_support_score=session.evidence_support_score,
                total_latency_ms=session.total_latency_ms
            )
            file_size = os.path.getsize(normalized_path)
            repo.update_pdf_path(session_id, normalized_path)
            repo.create_report_record(session_id, normalized_path, file_size)
        except Exception as gen_err:
            logger.error(f"Failed to generate inspection report: {gen_err}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to compile PDF report: {str(gen_err)}"
            )

    return FileResponse(
        path=normalized_path,
        media_type="application/pdf",
        filename=os.path.basename(normalized_path)
    )

@router.post("/reports/{session_id}/generate", summary="Synthesize / Regenerate PDF Report")
async def generate_report_record(session_id: str, db: Session = Depends(get_db)):
    """
    Explicitly forces re-compilation of the ReportLab PDF inspection report,
    persisting a new ReportRecord audit entry in SQLite.
    """
    validate_safe_session_id(session_id)
    repo = AnalysisRepository(db)
    session = repo.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis session '{session_id}' not found."
        )

    findings_data = [f.to_dict() for f in session.findings]
    traces_data = [t.to_dict() for t in session.traces]
    rag_data = [r.to_dict() for r in session.rag_sources]

    try:
        pdf_path = generate_inspection_pdf(
            session_id=session.id,
            query=session.user_query,
            input_path=session.input_path,
            annotated_path=session.annotated_path,
            input_type=session.input_type,
            findings=findings_data,
            traces=traces_data,
            rag_references=rag_data,
            evidence_support_score=session.evidence_support_score,
            total_latency_ms=session.total_latency_ms
        )
        file_size = os.path.getsize(pdf_path)
        repo.update_pdf_path(session_id, pdf_path)
        record = repo.create_report_record(
            session_id=session_id,
            pdf_path=pdf_path,
            file_size_bytes=file_size,
            report_format="PDF/A-1b"
        )
    except Exception as e:
        logger.error(f"Manual PDF generation error for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF report generation failed: {str(e)}"
        )

    return {
        "status": "success",
        "message": "Inspection PDF report synthesized successfully.",
        "report_uid": record.report_uid,
        "session_id": session_id,
        "file_size_bytes": record.file_size_bytes,
        "format": record.format,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "pdf_url": f"{settings.API_PREFIX}/reports/{session_id}/pdf"
    }

@router.get("/reports/{session_id}/metadata", summary="Get Persisted Report Records")
async def get_report_metadata(session_id: str, db: Session = Depends(get_db)):
    """Retrieves all historical PDF generation metadata records for a given session."""
    validate_safe_session_id(session_id)
    repo = AnalysisRepository(db)
    records = repo.get_report_records(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "total_records": len(records),
        "reports": [r.to_dict() for r in records]
    }
