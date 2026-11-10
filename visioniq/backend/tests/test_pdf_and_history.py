import os
import uuid
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.database.models import AnalysisSession, Finding, AgentExecutionTrace, RAGSource, ReportRecord
from backend.app.reporting.pdf_generator import generate_inspection_pdf, get_scaled_flowable_image

client = TestClient(app)

@pytest.fixture
def sample_image():
    """Generates a temporary synthetic inspection image."""
    img_dir = os.path.join(settings.UPLOADS_DIR, "test_pdf_fixtures")
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f"test_img_{uuid.uuid4().hex[:6]}.jpg")
    
    # Create 300x200 canvas
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.putText(img, "TEST VISIONIQ", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(img_path, img)
    
    yield img_path
    
    if os.path.exists(img_path):
        os.remove(img_path)

def test_pdf_generator_creates_valid_pdf(sample_image):
    """Verifies generate_inspection_pdf creates a binary PDF with valid %PDF-1. header and NumberedCanvas."""
    session_id = str(uuid.uuid4())
    findings = [
        {
            "finding_uid": "F-PPE-01",
            "agent_source": "ppe",
            "label": "NO-Hardhat",
            "detection_confidence": 0.945,
            "observed_evidence": "Worker #1 head region (0-25% height) lacking certified ANSI Z89.1 hard hat.",
            "score_interpretation": "High confidence neural classification (94.5%).",
            "root_cause_hypothesis": "Lack of enforcement or storage near entrance.",
            "recommendation": "Mandate ANSI hard hat before site clearance.",
            "verification_procedure": "Supervisor physical checkpoint verification.",
            "human_review_required": True,
            "bbox": [100.0, 50.0, 200.0, 180.0]
        },
        {
            "finding_uid": "F-DEF-02",
            "agent_source": "defect",
            "label": "crazing",
            "detection_confidence": 0.887,
            "observed_evidence": "Micro-network network fissures on cold-rolled steel surface.",
            "score_interpretation": "High certainty NEU-DefectNet CNN feature activation.",
            "root_cause_hypothesis": "Thermal fatigue during rapid annealing cooling cycle.",
            "recommendation": "Adjust cooling rate to < 15 C/min.",
            "verification_procedure": "ASTM E165 Liquid Penetrant non-destructive testing.",
            "human_review_required": False,
            "is_confirmed_defect": True,
            "bbox": [40.0, 40.0, 120.0, 140.0]
        }
    ]

    traces = [
        {
            "agent_name": "ppe",
            "relevance_score": 0.95,
            "evidence_capability": 0.90,
            "normalized_cost": 0.20,
            "expected_latency": 0.25,
            "final_routing_score": 0.582,
            "decision": "SELECTED",
            "actual_latency_ms": 48.2
        },
        {
            "agent_name": "defect",
            "relevance_score": 0.85,
            "evidence_capability": 0.88,
            "normalized_cost": 0.20,
            "expected_latency": 0.22,
            "final_routing_score": 0.541,
            "decision": "SELECTED",
            "actual_latency_ms": 32.5
        },
        {
            "agent_name": "ocr",
            "relevance_score": 0.10,
            "evidence_capability": 0.50,
            "normalized_cost": 0.40,
            "expected_latency": 0.40,
            "final_routing_score": 0.070,
            "decision": "SKIPPED",
            "actual_latency_ms": 0.0
        }
    ]

    rag_refs = [
        {
            "document_title": "OSHA Standard 1926.100 - Head Protection",
            "section_reference": "29 CFR 1926.100(a)",
            "similarity_score": 0.895,
            "excerpt": "Employees working in areas where there is a possible danger of head injury from impact shall be protected by protective helmets.",
            "source_uri": "https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.100"
        }
    ]

    pdf_path = generate_inspection_pdf(
        session_id=session_id,
        query="Verify PPE compliance and detect surface steel defects",
        input_path=sample_image,
        annotated_path=sample_image,
        input_type="image",
        findings=findings,
        traces=traces,
        rag_references=rag_refs,
        evidence_support_score=0.92,
        total_latency_ms=80.7
    )

    assert os.path.exists(pdf_path)
    file_size = os.path.getsize(pdf_path)
    assert file_size > 1000  # Substantial PDF document

    # Read binary header
    with open(pdf_path, "rb") as f:
        header = f.read(8)
        assert header.startswith(b"%PDF-1.")

    # Clean up test output
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

def test_image_scaling_helper(sample_image):
    """Verifies get_scaled_flowable_image respects bounding limits and handles missing files gracefully."""
    # Valid image
    rl_img = get_scaled_flowable_image(sample_image, 200, 100)
    assert rl_img is not None
    assert rl_img.drawWidth <= 200.0
    assert rl_img.drawHeight <= 100.0

    # Non-existent image returns None
    missing = get_scaled_flowable_image("non_existent_file.png", 200, 100)
    assert missing is None

def test_get_report_pdf_endpoint(sample_image):
    """Verifies GET /api/v1/reports/{session_id}/pdf returns application/pdf."""
    db = SessionLocal()
    session_id = str(uuid.uuid4())
    try:
        session = AnalysisSession(
            id=session_id,
            user_query="Inspect steel plate crazing",
            input_type="image",
            input_path=sample_image,
            annotated_path=sample_image,
            detected_intent="defect",
            status="completed",
            total_latency_ms=45.0,
            evidence_support_score=0.95
        )
        db.add(session)
        finding = Finding(
            session_id=session_id,
            finding_uid="F-DEF-01",
            agent_source="defect",
            label="crazing",
            detection_confidence=0.91,
            observed_evidence="Network cracks visible on metal surface",
            model_name="NEU-DefectNet",
            model_version="v1.0.0"
        )
        db.add(finding)
        db.commit()

        # Request PDF download
        res = client.get(f"/api/v1/reports/{session_id}/pdf")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert res.content.startswith(b"%PDF-1.")
        assert len(res.content) > 1000

        # Verify ReportRecord was saved
        records = db.query(ReportRecord).filter(ReportRecord.session_id == session_id).all()
        assert len(records) >= 1
        assert records[0].file_size_bytes > 0
    finally:
        db.close()

def test_post_generate_report_endpoint(sample_image):
    """Verifies POST /api/v1/reports/{session_id}/generate forces re-compilation."""
    db = SessionLocal()
    session_id = str(uuid.uuid4())
    try:
        session = AnalysisSession(
            id=session_id,
            user_query="PPE check for helmet and vest",
            input_type="image",
            input_path=sample_image,
            annotated_path=sample_image,
            detected_intent="ppe",
            status="completed",
            total_latency_ms=62.0
        )
        db.add(session)
        db.commit()

        res = client.post(f"/api/v1/reports/{session_id}/generate")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "report_uid" in data
        assert data["file_size_bytes"] > 0
        assert data["pdf_url"] == f"/api/v1/reports/{session_id}/pdf"
    finally:
        db.close()

def test_path_traversal_rejection():
    """Verifies path traversal attempts in session_id are rejected with 400 Bad Request."""
    res1 = client.get("/api/v1/reports/..%2F..%2Fetc%2Fpasswd/pdf")
    assert res1.status_code in (400, 404)

    res2 = client.get("/api/v1/reports/invalid;id/pdf")
    assert res2.status_code == 400

    res3 = client.post("/api/v1/reports/invalid..session/generate")
    assert res3.status_code == 400

def test_history_search_and_filters(sample_image):
    """Verifies GET /api/v1/history filtering by query, input_type, and status."""
    db = SessionLocal()
    u1 = str(uuid.uuid4())
    u2 = str(uuid.uuid4())
    try:
        s1 = AnalysisSession(
            id=u1,
            user_query="Crane operator safety check",
            input_type="video",
            input_path=sample_image,
            detected_intent="ppe",
            status="completed",
            total_latency_ms=120.0
        )
        s2 = AnalysisSession(
            id=u2,
            user_query="Steel surface scratch inspection",
            input_type="image",
            detected_intent="defect",
            status="failed",
            input_path=sample_image,
            total_latency_ms=15.0
        )
        db.add_all([s1, s2])
        db.commit()

        # Filter by input_type=video
        res_video = client.get("/api/v1/history?input_type=video")
        assert res_video.status_code == 200
        video_ids = [s["id"] for s in res_video.json()["sessions"]]
        assert u1 in video_ids
        assert u2 not in video_ids

        # Filter by query search
        res_query = client.get("/api/v1/history?query=crane")
        assert res_query.status_code == 200
        query_ids = [s["id"] for s in res_query.json()["sessions"]]
        assert u1 in query_ids
        assert u2 not in query_ids

        # Filter by status=failed
        res_status = client.get("/api/v1/history?status=failed")
        assert res_status.status_code == 200
        failed_ids = [s["id"] for s in res_status.json()["sessions"]]
        assert u2 in failed_ids
    finally:
        db.close()
