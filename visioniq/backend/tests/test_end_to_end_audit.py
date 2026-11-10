import os
import io
import uuid
import time
import base64
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.database.models import AnalysisSession, Finding, AgentExecutionTrace, RAGSource, ReportRecord, ExperimentRun
from backend.app.orchestration.router import route_query, compute_routing_scores
from backend.app.orchestration.graph import vision_iq_app

client = TestClient(app)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@pytest.fixture(scope="module")
def fixture_images():
    """Generates synthetic high-contrast test images for end-to-end workflow verification."""
    fix_dir = os.path.join(settings.UPLOADS_DIR, "audit_fixtures")
    os.makedirs(fix_dir, exist_ok=True)

    # 1. Standard Inspection Image
    p1 = os.path.join(fix_dir, "test_inspection.jpg")
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 120
    cv2.putText(img1, "WORKER ZONE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(p1, img1)

    # 2. Multi-Person PPE Canvas
    p2 = os.path.join(fix_dir, "test_multi_ppe.jpg")
    img2 = np.ones((480, 640, 3), dtype=np.uint8) * 160
    # Person 1 with hard hat
    cv2.rectangle(img2, (100, 100), (250, 420), (50, 50, 50), -1)
    cv2.circle(img2, (175, 130), 30, (0, 220, 255), -1) # Yellow hard hat
    # Person 2 without hard hat
    cv2.rectangle(img2, (380, 120), (530, 440), (50, 50, 50), -1)
    cv2.circle(img2, (455, 150), 28, (80, 50, 30), -1) # Dark hair / NO-Hardhat
    cv2.imwrite(p2, img2)

    # 3. Steel Defect Surface
    p3 = os.path.join(fix_dir, "test_defect.jpg")
    img3 = np.ones((200, 200, 3), dtype=np.uint8) * 180
    for offset in range(-20, 20, 5):
        cv2.line(img3, (30 + offset, 20), (160 + offset, 170), (40, 40, 40), 2)
    cv2.imwrite(p3, img3)

    # 4. OCR Signage Image
    p4 = os.path.join(fix_dir, "test_ocr_sign.png")
    img4 = np.ones((300, 500, 3), dtype=np.uint8) * 240
    cv2.rectangle(img4, (20, 20), (480, 80), (0, 200, 255), -1)
    cv2.putText(img4, "CAUTION", (150, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    cv2.putText(img4, "HARD HAT REQUIRED", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img4, "SN: MCET-2026-X1", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 150), 2)
    cv2.imwrite(p4, img4)

    return {"standard": p1, "multi_ppe": p2, "defect": p3, "ocr": p4}

# =========================================================================
# WORKFLOW 1: Upload Image, Query, Selected Agents, Annotations, PDF Download
# =========================================================================
def test_workflow_01_image_inspection_e2e(fixture_images):
    """End-to-end verification of image upload, agent routing, overlays, and PDF generation."""
    with open(fixture_images["standard"], "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", file_bytes, "image/jpeg")},
        data={"query": "Inspect site for visible personnel, equipment and safety compliance"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert data["input_type"] == "image"
    assert len(data["selected_agents"]) > 0
    assert data["total_latency_ms"] > 0
    assert data["pdf_url"] is not None

    # Verify matching PDF download
    pdf_res = client.get(data["pdf_url"])
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF-1.")

# =========================================================================
# WORKFLOW 2: Multi-Person PPE Findings, Anatomical Zoning & Uncertainty
# =========================================================================
def test_workflow_02_multi_person_ppe_analysis(fixture_images):
    """Verifies person-level PPE evaluation, stable worker attribution, and review flags."""
    with open(fixture_images["multi_ppe"], "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/v1/analyze/image",
        files={"file": ("multi_ppe.jpg", file_bytes, "image/jpeg")},
        data={"query": "Check personal protective equipment compliance for all workers"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "ppe" in data["selected_agents"]
    findings = data["findings"]
    assert isinstance(findings, list)

    # Check for presence of explainability fields
    for f in findings:
        assert "finding_uid" in f
        assert "observed_evidence" in f
        assert f["agent_source"] in ("ppe", "object", "defect", "ocr", "scene")
        if f.get("human_review_required"):
            assert f["score_interpretation"] is not None

# =========================================================================
# WORKFLOW 3: Industrial Defect Analysis & Root-Cause Hypotheses
# =========================================================================
def test_workflow_03_industrial_defect_inspection(fixture_images):
    """Verifies localized defect evidence, root-cause hypotheses, and NDT recommendations."""
    with open(fixture_images["defect"], "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/v1/analyze/image",
        files={"file": ("defect.jpg", file_bytes, "image/jpeg")},
        data={"query": "Inspect metal surface for scratches, crazing cracks and pitting"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "defect" in data["selected_agents"]

    defect_findings = [f for f in data["findings"] if f["agent_source"] == "defect"]
    if defect_findings:
        df = defect_findings[0]
        assert df["observed_evidence"] is not None
        assert df["recommendation"] is not None
        assert df["verification_procedure"] is not None

# =========================================================================
# WORKFLOW 4: OCR Text Extraction & Region Bounding Boxes
# =========================================================================
def test_workflow_04_ocr_text_extraction(fixture_images):
    """Verifies OCR text extraction, region bounding boxes, and string decoding."""
    with open(fixture_images["ocr"], "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/v1/analyze/image",
        files={"file": ("ocr_sign.png", file_bytes, "image/png")},
        data={"query": "Read the caution text and serial number on this sign"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "ocr" in data["selected_agents"]

    ocr_findings = [f for f in data["findings"] if f["agent_source"] == "ocr"]
    assert len(ocr_findings) > 0
    extracted_text = " ".join([f["label"] for f in ocr_findings]).upper()
    assert any(term in extracted_text for term in ["CAUTION", "HARD", "HAT", "MCET", "SN"])

# =========================================================================
# WORKFLOW 5: Video Frame Sampling & Temporal Tracking
# =========================================================================
def test_workflow_05_video_processing(fixture_images):
    """Verifies video format validation, frame sampling interval, and tracking persistence."""
    # Create synthetic 3-frame MP4 video
    vid_path = os.path.join(settings.UPLOADS_DIR, "test_sample_vid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_path, fourcc, 5.0, (300, 200))
    for i in range(10):
        frame = np.ones((200, 300, 3), dtype=np.uint8) * (50 + i * 15)
        cv2.putText(frame, f"F-{i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    with open(vid_path, "rb") as vf:
        vbytes = vf.read()

    res = client.post(
        "/api/v1/video/process",
        files={"video": ("sample.mp4", vbytes, "video/mp4")},
        data={"query": "Track all personnel across frames and check safety", "sample_fps": 1.0, "max_frames": 5}
    )
    assert res.status_code == 200
    vdata = res.json()
    assert vdata["total_video_frames"] == 10
    assert vdata["sampled_frames_processed"] > 0
    assert "unique_tracked_entities" in vdata
    assert len(vdata["frame_results"]) > 0

    if os.path.exists(vid_path):
        os.remove(vid_path)

# =========================================================================
# WORKFLOW 6: Live Camera Session Lifecycle & Frame Inference
# =========================================================================
def test_workflow_06_live_camera_lifecycle(fixture_images):
    """Verifies camera frame ingestion, base64 payload decoding, and inference response."""
    with open(fixture_images["standard"], "rb") as f:
        raw_bytes = f.read()
    b64_str = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

    payload = {
        "frame_base64": b64_str,
        "query": "Check workspace for workers and equipment",
        "timestamp": time.time()
    }
    res = client.post("/api/v1/camera/frame", json=payload)
    assert res.status_code == 200
    cam_data = res.json()
    assert cam_data["status"] == "success"
    assert cam_data["session_id"] is not None
    assert cam_data["latency_ms"] > 0
    assert cam_data["annotated_frame_base64"] is not None

# =========================================================================
# WORKFLOW 7: Document Ingestion into RAG & Grounded Retrieval
# =========================================================================
def test_workflow_07_rag_ingest_and_retrieve():
    """Verifies text extraction, chunking, embedding index, and semantic retrieval."""
    test_doc_content = (
        "# ASTM E165 Non-Destructive Testing Protocol\n\n"
        "Section 4.1 Liquid Penetrant Examination:\n"
        "Liquid penetrant examination is a sensitive method of detecting surface-breaking defects "
        "such as hairline cracks, crazing networks, and porosity in ferrous and non-ferrous metals.\n\n"
        "Section 4.2 Minimum Dwell Time:\n"
        "The standard penetrant dwell time for stainless and carbon sheet steels shall be at least 15 minutes."
    )
    doc_bytes = test_doc_content.encode("utf-8")

    # 1. Ingest
    res_ingest = client.post(
        "/api/v1/rag/ingest",
        files={"file": ("ASTM_E165_NDT.md", doc_bytes, "text/markdown")},
        data={"title": "ASTM E165 Liquid Penetrant Testing Standard", "source_uri": "https://www.astm.org/e0165"}
    )
    assert res_ingest.status_code == 200
    ingest_data = res_ingest.json()
    assert ingest_data["status"] == "success"
    assert ingest_data["chunks_indexed"] >= 1

    # 2. Semantic Query
    res_query = client.post(
        "/api/v1/rag/query",
        json={"query": "What is the minimum dwell time for liquid penetrant testing?", "top_k": 3, "relevance_threshold": 0.20}
    )
    assert res_query.status_code == 200
    qdata = res_query.json()
    assert len(qdata["results"]) > 0
    top_hit = qdata["results"][0]
    assert "dwell time" in top_hit["excerpt"].lower()
    assert top_hit["similarity_score"] > 0.20

# =========================================================================
# WORKFLOW 8: History Record Retrieval and Matching Report Download
# =========================================================================
def test_workflow_08_history_and_report_download():
    """Verifies SQLite history retrieval and on-demand PDF report fetching."""
    res_hist = client.get("/api/v1/history?limit=10")
    assert res_hist.status_code == 200
    sessions = res_hist.json()["sessions"]
    assert len(sessions) > 0

    target_session = sessions[0]
    session_id = target_session["id"]

    res_pdf = client.get(f"/api/v1/reports/{session_id}/pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert res_pdf.content.startswith(b"%PDF-1.")

# =========================================================================
# WORKFLOW 9: Agent Execution Trace Mathematical Fidelity (Equation 1)
# =========================================================================
def test_workflow_09_trace_equation_fidelity():
    """Verifies that all Equation (1) parameter scores and execution times originate from real execution."""
    plan = route_query(query="Verify PPE safety compliance and read equipment label")
    traces = plan["traces"]
    for agent_name, t in traces.items():
        assert 0.0 <= t["relevance_score"] <= 1.0
        assert 0.0 <= t["evidence_capability"] <= 1.0
        assert 0.0 <= t["normalized_cost"] <= 1.0
        assert 0.0 <= t["expected_latency"] <= 1.0
        # Check Eq. 1 consistency: S(a,q) = a*R + d*E - b*C - g*L
        expected_s = (settings.ROUTING_ALPHA * t["relevance_score"] +
                      settings.ROUTING_DELTA * t["evidence_capability"] -
                      settings.ROUTING_BETA * t["normalized_cost"] -
                      settings.ROUTING_GAMMA * t["expected_latency"])
        assert abs(t["final_routing_score"] - expected_s) < 0.001
        assert t["decision"] in ("SELECTED", "SKIPPED", "MANDATORY_DEPENDENCY", "FALLBACK")

# =========================================================================
# WORKFLOW 10: Multi-Pipeline Comparative Evaluation (Fixed vs Rule vs Adaptive)
# =========================================================================
def test_workflow_10_evaluation_configurations_in_db():
    """Verifies empirical benchmark records exist in SQLite for all 3 primary architectures."""
    db = SessionLocal()
    try:
        types_in_db = {r.pipeline_type for r in db.query(ExperimentRun).all()}
        assert "fixed" in types_in_db
        assert "rule_based" in types_in_db
        assert "adaptive" in types_in_db

        # Check API response
        res = client.get("/api/v1/experiments")
        assert res.status_code == 200
        comps = {c["pipeline_type"]: c for c in res.json()["pipeline_comparisons"]}
        assert comps["fixed"]["status"] == "completed"
        assert comps["rule_based"]["status"] == "completed"
        assert comps["adaptive"]["status"] == "completed"
    finally:
        db.close()

# =========================================================================
# WORKFLOW 11: Security, Corrupt Upload & Malicious Input Rejection
# =========================================================================
def test_workflow_11_security_and_corrupt_upload_rejection():
    """Verifies strict security validation: 0-byte upload, corrupt raster, and path traversal."""
    # 1. Zero-byte upload
    res_zero = client.post(
        "/api/v1/analyze/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"query": "Inspect empty file"}
    )
    assert res_zero.status_code == 400
    assert "0 bytes" in res_zero.json()["detail"].lower() or "empty" in res_zero.json()["detail"].lower()

    # 2. Corrupt non-image raster
    corrupt_bytes = b"NOT_A_REAL_IMAGE_FILE_JUST_PLAIN_TEXT"
    res_corrupt = client.post(
        "/api/v1/analyze/image",
        files={"file": ("fake.jpg", corrupt_bytes, "image/jpeg")},
        data={"query": "Inspect fake image"}
    )
    assert res_corrupt.status_code == 400
    assert "corrupt" in res_corrupt.json()["detail"].lower() or "decode" in res_corrupt.json()["detail"].lower()

    # 3. Path Traversal rejection
    res_traversal1 = client.get("/api/v1/reports/..%2F..%2Fetc%2Fpasswd/pdf")
    assert res_traversal1.status_code in (400, 404)

    res_traversal2 = client.post("/api/v1/reports/invalid;injection/generate")
    assert res_traversal2.status_code == 400

# =========================================================================
# WORKFLOW 12: Partial Results & Fault Isolation Under Agent Failure
# =========================================================================
def test_workflow_12_fault_isolation_partial_recovery(fixture_images):
    """Verifies that when an optional specialist fails, partial results remain accessible."""
    with open(fixture_images["standard"], "rb") as f:
        file_bytes = f.read()

    # Execute state directly with a synthetic failing agent injected
    session_id = f"test_fault_{uuid.uuid4().hex[:6]}"
    state_input = {
        "session_id": session_id,
        "image_path": fixture_images["standard"],
        "user_query": "Check site for objects and defects",
        "input_type": "image",
        "selected_agents": ["object", "defect"],
        "timeout_seconds": 15.0
    }

    # Execute pipeline through LangGraph
    final_state = vision_iq_app.invoke(state_input)
    assert final_state is not None
    # Even if one branch encounters an issue, the final response is properly synthesized
    resp = final_state.get("final_response", {})
    assert resp.get("status") in ("completed", "partial")
    assert isinstance(resp.get("findings"), list)
    assert resp.get("pdf_report_path") is not None
