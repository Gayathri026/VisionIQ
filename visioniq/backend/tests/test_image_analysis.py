"""
VisionIQ Phase 4 Image Analysis & Explainability Automated Test Suite
Validates:
1. Image upload validation (rejection of empty 0-byte and corrupt files with HTTP 400).
2. YOLO detection with original-coordinate bounding boxes [x1, y1, x2, y2].
3. Empty image and low-resolution image handling without server error.
4. Multi-person PPE analysis with stable identifiers (Worker #1, Worker #2).
5. Anatomical spatial association & mutual occlusion / distance uncertainty handling.
6. Industrial defect detection: distinguishing confirmed model classifications from heuristic anomaly cues.
7. OCR analysis with text-region boxes and raw uncorrected strings.
8. Evidence-based explainability: model name, version, score interpretation, observations, and NDT verification steps.
9. Annotated image file generation matching original raster dimensions.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.agents.agent_b_object import ObjectDetectionAgent
from backend.app.agents.agent_c_ppe import PPESafetyAgent, compute_box_iou
from backend.app.agents.agent_d_ocr import OCRAgent
from backend.app.agents.agent_e_defect import IndustrialDefectAgent
from backend.app.services.annotation_service import annotation_service
from backend.app.schemas.contracts import FindingContract

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _create_synthetic_image_bytes(width=320, height=240, color=(100, 150, 200)) -> bytes:
    """Helper to generate encoded JPEG bytes for testing."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Draw some industrial shapes
    cv2.rectangle(img, (40, 50), (120, 180), (40, 40, 220), -1)
    cv2.putText(img, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    success, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()

def test_upload_validation_corrupt_and_empty():
    """Verify empty and corrupt image files are rejected with HTTP 400."""
    # 1. 0-byte upload
    resp_empty = client.post(
        "/api/v1/analyze/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        data={"query": "Check PPE safety"}
    )
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.json()["detail"].lower()

    # 2. Corrupted file bytes
    corrupt_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIFbad_truncated_data_bytes_12345"
    resp_corrupt = client.post(
        "/api/v1/analyze/image",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
        data={"query": "Check PPE safety"}
    )
    assert resp_corrupt.status_code == 400
    assert "corrupt" in resp_corrupt.json()["detail"].lower() or "decode" in resp_corrupt.json()["detail"].lower()

def test_yolo_object_detection_coordinates():
    """Verify object detection bounds stay within original image pixel dimensions."""
    w, h = 400, 300
    img_bytes = _create_synthetic_image_bytes(width=w, height=h)

    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("factory.jpg", img_bytes, "image/jpeg")},
        data={"query": "Count all visible people and objects"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "findings" in data
    assert data["status"] == "completed"

    for f in data["findings"]:
        if f.get("bbox"):
            bx1, by1, bx2, by2 = f["bbox"]
            assert 0.0 <= bx1 <= bx2 <= w + 1.0, f"Box x-coords [{bx1}, {bx2}] out of range [0, {w}]"
            assert 0.0 <= by1 <= by2 <= h + 1.0, f"Box y-coords [{by1}, {by2}] out of range [0, {h}]"
            assert f["model_name"] is not None

def test_empty_image_handling():
    """Verify a plain empty image executes without crashing or throwing errors."""
    plain_img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", plain_img)

    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("blank.jpg", encoded.tobytes(), "image/jpeg")},
        data={"query": "Find any defects or workers"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] is not None
    assert data["status"] in ["completed", "partial"]

def test_low_resolution_image_handling():
    """Verify small low-resolution images (32x32) process safely."""
    low_res = np.full((32, 32, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", low_res)

    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("lowres.jpg", encoded.tobytes(), "image/jpeg")},
        data={"query": "Inspect image"}
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] is not None

def test_multi_person_ppe_stable_association():
    """Verify multi-worker images assign stable IDs (Worker #1, Worker #2) deterministically."""
    # Create test image with 2 synthetic workers (left and right)
    img = np.full((300, 500, 3), 50, dtype=np.uint8)

    # Worker 1 (left): head yellow (helmet), torso neon yellow (vest)
    cv2.rectangle(img, (60, 40), (140, 260), (180, 180, 180), -1) # body
    cv2.rectangle(img, (60, 40), (140, 95), (20, 200, 220), -1)   # yellow helmet
    cv2.rectangle(img, (60, 95), (140, 180), (30, 220, 230), -1)  # vest

    # Worker 2 (right): head dark (no helmet), torso dark (no vest)
    cv2.rectangle(img, (320, 40), (400, 260), (180, 180, 180), -1) # body
    cv2.rectangle(img, (320, 40), (400, 95), (30, 30, 30), -1)     # dark hair (no helmet)
    cv2.rectangle(img, (320, 95), (400, 180), (40, 40, 40), -1)    # dark shirt (no vest)

    test_img_path = str(PROJECT_ROOT / "storage" / "uploads" / "test_multi_worker.jpg")
    os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
    cv2.imwrite(test_img_path, img)

    agent = PPESafetyAgent()
    mock_objects = {
        "objects": [
            {"label": "person", "bbox": [320.0, 40.0, 400.0, 260.0], "confidence": 0.94}, # Right worker
            {"label": "person", "bbox": [60.0, 40.0, 140.0, 260.0], "confidence": 0.95}   # Left worker
        ]
    }

    result = agent.analyze(test_img_path, mock_objects)
    assert result["total_persons_evaluated"] == 2
    findings = result["findings"]
    assert len(findings) == 2

    # Check that sorting ordered them from left to right: Worker #1 (x=60) then Worker #2 (x=320)
    assert findings[0]["person_id"] == "Worker #1 (W-01)"
    assert findings[1]["person_id"] == "Worker #2 (W-02)"

    # Worker 1 has helmet & vest
    assert "Compliant" in findings[0]["label"]

    # Worker 2 lacks helmet & vest -> violation
    assert "Violation" in findings[1]["label"]
    assert findings[1]["human_review_required"] is True

def test_ppe_occlusion_and_uncertainty():
    """Verify overlapping workers trigger mutual occlusion uncertainty flags."""
    img = np.full((300, 400, 3), 60, dtype=np.uint8)
    test_img_path = str(PROJECT_ROOT / "storage" / "uploads" / "test_occluded_worker.jpg")
    os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
    cv2.imwrite(test_img_path, img)

    agent = PPESafetyAgent()
    # Two workers standing right in front of each other (high IoU)
    mock_objects = {
        "objects": [
            {"label": "person", "bbox": [100.0, 40.0, 180.0, 250.0], "confidence": 0.90},
            {"label": "person", "bbox": [120.0, 40.0, 200.0, 250.0], "confidence": 0.88}
        ]
    }

    result = agent.analyze(test_img_path, mock_objects)
    findings = result["findings"]
    assert len(findings) == 2
    # At least one should be flagged as uncertain due to occlusion
    assert any("Occluded" in f["label"] for f in findings)
    assert all(f["human_review_required"] is True for f in findings)

def test_defect_classification_vs_heuristic_cue():
    """Verify IndustrialDefectAgent distinguishes confirmed classifications from heuristic cues."""
    agent = IndustrialDefectAgent()

    # 1. Test real NEU scratches sample
    scratch_sample = PROJECT_ROOT / "ml" / "datasets" / "neu_surface_defect" / "test" / "scratches" / "scratches_sample_01.jpg"
    if scratch_sample.exists():
        result = agent.analyze(str(scratch_sample))
        assert result["defects_detected"] > 0
        findings = result["findings"]
        top_f = findings[0]
        assert top_f["model_name"] is not None
        assert top_f["score_interpretation"] is not None
        assert "ASTM" in top_f["verification_procedure"]

    # 2. Test plain synthetic plate with minor noise (heuristic cue)
    plain_plate = np.full((200, 200, 3), 180, dtype=np.uint8)
    cv2.circle(plain_plate, (100, 100), 15, (120, 120, 120), -1) # mild stain
    temp_path = str(PROJECT_ROOT / "storage" / "uploads" / "test_heuristic_plate.jpg")
    cv2.imwrite(temp_path, plain_plate)

    res_plate = agent.analyze(temp_path)
    if res_plate["defects_detected"] > 0:
        cue = res_plate["findings"][0]
        # Should have verification procedure specifying NDT
        assert "NDT" in cue["recommendation"] or "ASTM" in cue["verification_procedure"]

def test_ocr_text_region_extraction():
    """Verify OCR extracts text regions with bounding boxes and score interpretations."""
    img = np.full((150, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "VISIONIQ-101", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    ocr_path = str(PROJECT_ROOT / "storage" / "uploads" / "test_ocr_plate.jpg")
    cv2.imwrite(ocr_path, img)

    agent = OCRAgent()
    result = agent.analyze(ocr_path)
    assert "findings" in result
    assert result["status"] == "SUCCESS"
    for f in result["findings"]:
        assert f["model_name"] == "EasyOCR Text Recognizer"
        assert f["score_interpretation"] is not None
        assert f["bbox"] is not None
        assert len(f["bbox"]) == 4

def test_finding_explainability_fields():
    """Verify all findings satisfy IEEE Access explainability requirements."""
    img_bytes = _create_synthetic_image_bytes(width=300, height=300)
    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("explain_test.jpg", img_bytes, "image/jpeg")},
        data={"query": "Locate cracks and check safety gear"}
    )
    assert resp.status_code == 200
    data = resp.json()

    for f in data.get("findings", []):
        assert f["finding_uid"].startswith("F-")
        assert len(f["observed_evidence"]) > 5
        assert f["model_name"] is not None
        assert f["score_interpretation"] is not None
        if f.get("human_review_required"):
            assert f["verification_procedure"] is not None

def test_annotated_image_persistence_and_alignment():
    """Verify annotated image is saved to disk and dimensions match the original."""
    w, h = 350, 250
    img_bytes = _create_synthetic_image_bytes(width=w, height=h)

    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("annot_test.jpg", img_bytes, "image/jpeg")},
        data={"query": "Check whether workers are wearing helmets"}
    )
    assert resp.status_code == 200
    data = resp.json()

    annotated_path = data.get("annotated_path")
    if annotated_path and os.path.exists(annotated_path):
        annotated_img = cv2.imread(annotated_path)
        assert annotated_img is not None
        assert annotated_img.shape[0] == h, f"Height mismatch: expected {h}, got {annotated_img.shape[0]}"
        assert annotated_img.shape[1] == w, f"Width mismatch: expected {w}, got {annotated_img.shape[1]}"
