import os
import shutil
import uuid
import time
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.session import get_db
from backend.app.database.models import AnalysisSession, Finding, AgentExecutionTrace, RAGSource, ReportRecord
from backend.app.database.repository import AnalysisRepository
from backend.app.reporting.pdf_generator import generate_inspection_pdf
from backend.app.schemas.contracts import (
    HistoryResponse, HistorySessionContract,
    OrchestrationPlanRequest, OrchestrationPlanResponse,
    AnalysisResponse, FindingContract, AgentTraceContract, RAGSourceContract,
    SessionTraceResponse,
    VideoProcessingRequest, VideoProcessingResponse,
    CameraFrameRequest, CameraFrameResponse
)
from backend.app.orchestration.router import route_query
from backend.app.orchestration.graph import vision_iq_app
from backend.app.services.video_tracking_service import video_tracking_service
from backend.app.services.camera_service import camera_inference_service

router = APIRouter()

@router.post("/orchestration/plan", response_model=OrchestrationPlanResponse, summary="Compute Cost-Aware Routing Plan")
async def get_orchestration_plan(req: OrchestrationPlanRequest):
    """
    Evaluates query intent, calculates Equation (1) scores for all candidate agents,
    resolves mandatory dependencies, and returns the justified execution plan without invoking vision models.
    """
    t0 = time.time()
    plan = route_query(
        query=req.query,
        alpha=req.alpha,
        beta=req.beta,
        gamma=req.gamma,
        delta=req.delta,
        threshold=req.threshold
    )
    latency_ms = round((time.time() - t0) * 1000.0, 2)

    traces_contract = {
        name: AgentTraceContract(
            agent_name=t["agent_name"],
            relevance_score=t["relevance_score"],
            evidence_capability=t["evidence_capability"],
            normalized_cost=t["normalized_cost"],
            expected_latency=t["expected_latency"],
            final_routing_score=t["final_routing_score"],
            decision=t["decision"],
            selection_reason=t["selection_reason"],
            execution_status=t["execution_status"],
            actual_latency_ms=t["actual_latency_ms"],
            error_message=t.get("error_message")
        )
        for name, t in plan["traces"].items()
    }

    return OrchestrationPlanResponse(
        query=req.query,
        detected_intent=plan["intent"],
        intent_confidence=plan["intent_confidence"],
        selected_agents=plan["selected_agents"],
        skipped_agents=plan["skipped_agents"],
        considered_agents=plan["considered_agents"],
        threshold=plan["threshold"],
        traces=traces_contract,
        latency_ms=latency_ms
    )

@router.post("/analyze/image", response_model=AnalysisResponse, summary="Run Multi-Agent Vision Inspection")
@router.post("/analyze", response_model=AnalysisResponse, summary="Run Multi-Agent Vision Inspection (Alias)")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    query: str = Form("Analyze visible objects, compliance and anomalies"),
    alpha: Optional[float] = Form(None),
    beta: Optional[float] = Form(None),
    gamma: Optional[float] = Form(None),
    delta: Optional[float] = Form(None),
    threshold: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Synchronously uploads an image, dynamically routes to required specialists via LangGraph,
    fuses evidence, compiles a verifiable PDF report, and persists records in SQLite.
    """
    upload_file = file or image
    if not upload_file:
        raise HTTPException(status_code=400, detail="No image file provided in request.")

    # 1. Read and validate content
    content = await upload_file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes). Please upload a valid image.")

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds maximum limit of 25MB.")

    # 2. Validate visual raster decoding
    import cv2
    import numpy as np
    nparr = np.frombuffer(content, np.uint8)
    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded_img is None or decoded_img.shape[0] < 5 or decoded_img.shape[1] < 5:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file: unable to decode visual raster.")

    session_id = str(uuid.uuid4())
    ext = os.path.splitext(upload_file.filename or "")[1] or ".jpg"
    safe_filename = f"upload_{session_id[:8]}{ext}"
    input_path = os.path.join(settings.UPLOADS_DIR, safe_filename)

    with open(input_path, "wb") as f_out:
        f_out.write(content)

    initial_state = {
        "session_id": session_id,
        "image_path": input_path,
        "user_query": query,
        "input_type": "image",
        "timeout_seconds": 15.0
    }

    try:
        final_state = vision_iq_app.invoke(initial_state)
    except Exception as e:
        logger.error(f"Orchestration invocation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failure: {str(e)}")

    final_resp = final_state.get("final_response", {})

    finding_contracts = [
        FindingContract(
            finding_uid=f.get("finding_uid", "F-00"),
            agent_source=f.get("agent_source", "unknown"),
            label=f.get("label", "Unknown"),
            bbox=f.get("bbox"),
            detection_confidence=f.get("detection_confidence"),
            anomaly_score=f.get("anomaly_score"),
            routing_score=f.get("routing_score"),
            observed_evidence=f.get("observed_evidence", ""),
            root_cause_hypothesis=f.get("root_cause_hypothesis"),
            alternative_explanations=f.get("alternative_explanations"),
            recommendation=f.get("recommendation"),
            verification_procedure=f.get("verification_procedure"),
            human_review_required=f.get("human_review_required", False),
            frame_timestamp=f.get("frame_timestamp"),
            track_id=f.get("track_id"),
            model_name=f.get("model_name"),
            model_version=f.get("model_version"),
            score_interpretation=f.get("score_interpretation"),
            person_id=f.get("person_id"),
            is_confirmed_defect=f.get("is_confirmed_defect")
        )
        for f in final_resp.get("findings", [])
    ]

    trace_contracts = [
        AgentTraceContract(
            agent_name=t["agent_name"],
            relevance_score=t["relevance_score"],
            evidence_capability=t["evidence_capability"],
            normalized_cost=t["normalized_cost"],
            expected_latency=t["expected_latency"],
            final_routing_score=t["final_routing_score"],
            decision=t["decision"],
            selection_reason=t.get("selection_reason"),
            execution_status=t.get("execution_status", "SUCCESS"),
            actual_latency_ms=t.get("actual_latency_ms", 0.0),
            error_message=t.get("error_message")
        )
        for t in final_resp.get("traces", [])
    ]

    rag_contracts = [
        RAGSourceContract(
            document_title=r["document_title"],
            section_reference=r["section_reference"],
            excerpt=r["excerpt"],
            similarity_score=r["similarity_score"],
            source_uri=r.get("source_uri")
        )
        for r in final_resp.get("rag_references", [])
    ]

    return AnalysisResponse(
        session_id=session_id,
        created_at=final_resp.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        user_query=query,
        input_type="image",
        input_path=input_path,
        input_url=f"/storage/uploads/{safe_filename}",
        annotated_path=final_resp.get("annotated_path"),
        annotated_url=f"/storage/annotations/{os.path.basename(final_resp.get('annotated_path', ''))}" if final_resp.get("annotated_path") else None,
        detected_intent=final_resp.get("detected_intent"),
        intent_confidence=final_resp.get("intent_confidence", 1.0),
        selected_agents=final_resp.get("selected_agents", []),
        executed_agents=final_resp.get("executed_agents", []),
        skipped_agents=final_resp.get("skipped_agents", []),
        failed_agents=final_resp.get("failed_agents", []),
        status=final_resp.get("status", "completed"),
        total_latency_ms=final_resp.get("total_latency_ms", 0.0),
        synthesis_text=final_resp.get("synthesis_text"),
        evidence_support_score=final_resp.get("evidence_support_score", 1.0),
        pdf_report_path=final_resp.get("pdf_report_path"),
        pdf_url=f"{settings.API_PREFIX}/reports/{session_id}/pdf" if final_resp.get("pdf_report_path") else None,
        findings=finding_contracts,
        traces=trace_contracts,
        rag_sources=rag_contracts
    )

@router.get("/analysis/{session_id}/trace", response_model=SessionTraceResponse, summary="Get Persisted Execution Trace")
async def get_session_trace(session_id: str, db: Session = Depends(get_db)):
    """Retrieves the complete persisted mathematical trace and agent execution times for a session."""
    session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    traces = db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == session_id).all()
    trace_contracts = [
        AgentTraceContract(
            agent_name=t.agent_name,
            relevance_score=t.relevance_score,
            evidence_capability=t.evidence_capability,
            normalized_cost=t.normalized_cost,
            expected_latency=t.expected_latency,
            final_routing_score=t.final_routing_score,
            decision=t.decision,
            selection_reason=t.selection_reason,
            execution_status=t.execution_status,
            actual_latency_ms=t.actual_latency_ms,
            error_message=t.error_message
        )
        for t in traces
    ]

    return SessionTraceResponse(
        session_id=session.id,
        user_query=session.user_query,
        detected_intent=session.detected_intent,
        selected_agents=session.selected_agents or [],
        executed_agents=session.executed_agents or [],
        skipped_agents=session.skipped_agents or [],
        failed_agents=session.failed_agents or [],
        total_latency_ms=session.total_latency_ms,
        traces=trace_contracts
    )

@router.get("/session/{session_id}", summary="Get Complete Session Details")
async def get_session_details(session_id: str, db: Session = Depends(get_db)):
    """Retrieves full session record with findings, traces, and RAG sources."""
    session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    return {"status": "success", "session": session.to_dict()}

@router.get("/history", response_model=HistoryResponse, summary="Query Analysis History")
async def get_history(
    query: Optional[str] = None,
    input_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    order: str = "desc",
    db: Session = Depends(get_db)
):
    """Retrieves previous inspection sessions with optional query search, type/status filters, and latency records."""
    repo = AnalysisRepository(db)
    order_desc = order.lower() != "asc"
    sessions = repo.search_sessions(
        query=query,
        input_type=input_type,
        status=status,
        limit=limit,
        order_desc=order_desc
    )
    session_contracts = [
        HistorySessionContract(
            id=s.id,
            created_at=s.created_at.isoformat() if s.created_at else "",
            user_query=s.user_query,
            input_type=s.input_type,
            detected_intent=s.detected_intent,
            status=s.status,
            total_latency_ms=s.total_latency_ms,
            evidence_support_score=s.evidence_support_score,
            pdf_report_path=s.pdf_report_path
        )
        for s in sessions
    ]
    return HistoryResponse(status="success", total=len(sessions), sessions=session_contracts)

@router.post("/video/process", response_model=VideoProcessingResponse, summary="Process Video with Incremental Tracking")
@router.post("/analyze/video", response_model=VideoProcessingResponse, summary="Process Video with Incremental Tracking (Alias)")
async def process_video(
    video: UploadFile = File(...),
    query: str = Form("Track personnel and verify safety compliance"),
    sample_fps: float = Form(1.0),
    max_frames: int = Form(25),
    db: Session = Depends(get_db)
):
    """
    Incrementally processes uploaded video stream, tracking entities across frames
    to distinguish instantaneous per-frame counts from unique tracked individuals.
    Verifies temporal violation persistence across sequential frames.
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing video file in request.")

    ext = os.path.splitext(video.filename)[1].lower()
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed formats: {', '.join(sorted(allowed_extensions))}"
        )

    content = await video.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")

    max_video_size = 100 * 1024 * 1024  # 100 MB limit
    if len(content) > max_video_size:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded video ({len(content)/(1024*1024):.1f}MB) exceeds maximum limit of 100MB."
        )

    session_id = str(uuid.uuid4())
    safe_video_name = f"video_{session_id[:8]}{ext}"
    video_save_path = os.path.join(settings.UPLOADS_DIR, safe_video_name)

    with open(video_save_path, "wb") as f_out:
        f_out.write(content)

    try:
        result = video_tracking_service.process_video(
            video_path=video_save_path,
            query=query,
            sample_fps=sample_fps,
            max_frames=max_frames
        )
    except Exception as e:
        logger.error(f"Video tracking execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video tracking failure: {str(e)}")

    # Persist session & temporal findings to SQLite
    try:
        db_session = AnalysisSession(
            id=result["session_id"],
            user_query=query,
            input_type="video",
            input_path=video_save_path,
            detected_intent=result.get("detected_intent"),
            selected_agents=result.get("selected_agents", []),
            executed_agents=result.get("selected_agents", []),
            status="completed",
            total_latency_ms=result.get("total_latency_ms", 0.0),
            evidence_support_score=1.0,
            synthesis_text=result.get("synthesis_text")
        )
        db.add(db_session)
        db.commit()

        # Persist findings with frame timestamps and track IDs
        for fr in result.get("frame_results", []):
            for f in fr.get("findings", []):
                db_finding = Finding(
                    session_id=result["session_id"],
                    finding_uid=f.get("finding_uid", "F-TRK"),
                    agent_source=f.get("agent_source", "object"),
                    label=f.get("label", "Detection"),
                    bbox=f.get("bbox"),
                    detection_confidence=f.get("detection_confidence"),
                    anomaly_score=f.get("anomaly_score"),
                    observed_evidence=f.get("observed_evidence", ""),
                    root_cause_hypothesis=f.get("root_cause_hypothesis"),
                    recommendation=f.get("recommendation"),
                    verification_procedure=f.get("verification_procedure"),
                    human_review_required=f.get("human_review_required", False),
                    frame_timestamp=f.get("frame_timestamp"),
                    track_id=f.get("track_id"),
                    model_name=f.get("model_name"),
                    model_version=f.get("model_version"),
                    score_interpretation=f.get("score_interpretation"),
                    person_id=f.get("person_id"),
                    is_confirmed_defect=f.get("is_confirmed_defect")
                )
                db.add(db_finding)
        db.commit()
    except Exception as db_err:
        logger.warning(f"Video session SQLite persistence warning: {db_err}")
        db.rollback()

    return VideoProcessingResponse(
        session_id=result["session_id"],
        video_path=video_save_path,
        duration_sec=result["duration_sec"],
        total_video_frames=result["total_video_frames"],
        sampled_frames_processed=result["sampled_frames_processed"],
        sample_fps=result["sample_fps"],
        unique_tracked_entities=result["unique_tracked_entities"],
        total_unique_entities=result["total_unique_entities"],
        frame_results=result["frame_results"],
        temporal_violations=result.get("temporal_violations"),
        detected_intent=result.get("detected_intent"),
        selected_agents=result.get("selected_agents"),
        synthesis_text=result.get("synthesis_text"),
        total_latency_ms=result["total_latency_ms"]
    )

@router.post("/camera/frame", response_model=CameraFrameResponse, summary="Process Live Browser Camera Frame")
@router.post("/analyze/camera/frame", response_model=CameraFrameResponse, summary="Process Live Browser Camera Frame (Alias)")
async def process_camera_frame(req: CameraFrameRequest):
    """
    Decodes real-time camera payload, executes query-routed specialists,
    and returns annotated overlays with measured latency.
    """
    try:
        res = camera_inference_service.process_frame(
            frame_base64=req.frame_base64,
            query=req.query,
            timestamp=req.timestamp
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Live camera inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Camera processing failure: {str(e)}")

    finding_contracts = [
        FindingContract(
            finding_uid=f.get("finding_uid", "F-CAM"),
            agent_source=f.get("agent_source", "unknown"),
            label=f.get("label", "Unknown"),
            bbox=f.get("bbox"),
            detection_confidence=f.get("detection_confidence"),
            anomaly_score=f.get("anomaly_score"),
            observed_evidence=f.get("observed_evidence", ""),
            root_cause_hypothesis=f.get("root_cause_hypothesis"),
            recommendation=f.get("recommendation"),
            verification_procedure=f.get("verification_procedure"),
            human_review_required=f.get("human_review_required", False),
            frame_timestamp=f.get("frame_timestamp"),
            track_id=f.get("track_id"),
            model_name=f.get("model_name"),
            model_version=f.get("model_version"),
            score_interpretation=f.get("score_interpretation"),
            person_id=f.get("person_id"),
            is_confirmed_defect=f.get("is_confirmed_defect")
        )
        for f in res.get("findings", [])
    ]

    return CameraFrameResponse(
        status=res.get("status", "success"),
        session_id=res.get("session_id"),
        timestamp=res.get("timestamp", req.timestamp),
        latency_ms=res.get("latency_ms", 0.0),
        detected_intent=res.get("detected_intent"),
        selected_agents=res.get("selected_agents", []),
        executed_agents=res.get("executed_agents", []),
        findings=finding_contracts,
        annotated_frame_base64=res.get("annotated_frame_base64"),
        synthesis=res.get("synthesis")
    )
