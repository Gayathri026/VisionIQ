from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.app.services.assistant_service import assistant_service

router = APIRouter()

class AssistantQueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Why did VisionIQ select these agents?"})
    session_id: Optional[str] = Field(None, json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"})
    current_trace: Optional[List[Dict[str, Any]]] = Field(None)
    active_finding_uid: Optional[str] = Field(None, json_schema_extra={"example": "F-PPE-01"})

@router.post("/assistant/query", summary="Query VisionIQ In-App Technical Assistant")
async def query_assistant(req: AssistantQueryRequest):
    """
    Provides context-aware technical guidance grounded in repository specifications
    and active analysis session states.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    result = assistant_service.query(
        user_query=req.query,
        session_id=req.session_id,
        current_trace=req.current_trace,
        active_finding_uid=req.active_finding_uid
    )
    return result

@router.get("/assistant/faq", summary="List All Canonical Assistant Topics")
async def list_assistant_faq():
    """Returns canonical questions and topics supported by the deterministic engineering assistant."""
    questions = [
        {"id": "mode_selection", "question": "Which analysis mode should I use?", "category": "Operations"},
        {"id": "agent_selection", "question": "Why did VisionIQ select these agents?", "category": "Orchestration"},
        {"id": "confidence_scores", "question": "What does this confidence score mean?", "category": "Explainability"},
        {"id": "train_ppe", "question": "How do I train the PPE model?", "category": "Machine Learning"},
        {"id": "add_rag_document", "question": "How can I add a new technical document to RAG?", "category": "Knowledge Base"},
        {"id": "undetermined_cause", "question": "Why could the damage cause not be determined?", "category": "Explainability"},
        {"id": "evaluation_metrics", "question": "How do I interpret the evaluation metrics?", "category": "Scientific Research"},
        {"id": "pdf_report", "question": "How do I export and understand the PDF report?", "category": "Reporting"}
    ]
    return {
        "status": "success",
        "total_faqs": len(questions),
        "questions": questions
    }
