import os
import cv2
import uuid
import base64
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from backend.app.database.models import AnalysisSession, Finding
from backend.app.services.video_tracking_service import video_tracking_service, ActiveTrack

client = TestClient(app)

SAMPLE_VIDEO_PATH = "visioniq/sample_data/vidssave.com People Entering And Exiting Mall Stock Footage 720P.mp4"

def create_synthetic_frame_base64() -> str:
    """Creates a synthetic BGR frame with distinct visual objects encoded as base64."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 180
    # Draw simulated person/object shapes
    cv2.rectangle(img, (150, 100), (300, 420), (50, 120, 200), -1)
    cv2.circle(img, (225, 140), 35, (40, 180, 240), -1)
    cv2.rectangle(img, (170, 180), (280, 340), (20, 180, 20), -1)
    
    ret, buf = cv2.imencode(".jpg", img)
    assert ret is True
    b64_str = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def test_video_upload_validation_format_and_empty():
    """Verify safe rejection of unsupported video formats and 0-byte video files with HTTP 400."""
    # 1. Unsupported extension (.txt)
    response = client.post(
        "/api/v1/video/process",
        files={"video": ("test_video.txt", b"not a real video file", "text/plain")},
        data={"query": "Track all visible people"}
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]

    # 2. Empty video (0 bytes)
    response_empty = client.post(
        "/api/v1/video/process",
        files={"video": ("empty_video.mp4", b"", "video/mp4")},
        data={"query": "Track all visible people"}
    )
    assert response_empty.status_code == 400
    assert "empty (0 bytes)" in response_empty.json()["detail"]

def test_camera_frame_base64_validation():
    """Verify validation of live camera base64 payloads, rejecting corrupted or empty inputs."""
    # 1. Empty payload
    res_empty = client.post(
        "/api/v1/camera/frame",
        json={"frame_base64": "", "query": "Monitor safety", "timestamp": 1.0}
    )
    assert res_empty.status_code == 400
    assert "Empty camera frame payload" in res_empty.json()["detail"]

    # 2. Corrupt non-base64
    res_corrupt = client.post(
        "/api/v1/camera/frame",
        json={"frame_base64": "??!!this_is_not_base64!!??", "query": "Monitor safety", "timestamp": 1.0}
    )
    assert res_corrupt.status_code == 400

    # 3. Micro payload (< 100 bytes)
    res_tiny = client.post(
        "/api/v1/camera/frame",
        json={"frame_base64": base64.b64encode(b"too small").decode("utf-8"), "query": "Monitor safety", "timestamp": 1.0}
    )
    assert res_tiny.status_code == 400
    assert "too small" in res_tiny.json()["detail"]

def test_camera_frame_real_inference_execution():
    """Verify live camera frame decoding, query-aware agent routing, and annotated base64 response."""
    frame_b64 = create_synthetic_frame_base64()
    req_timestamp = 1729876543.5

    response = client.post(
        "/api/v1/camera/frame",
        json={
            "frame_base64": frame_b64,
            "query": "Continuous PPE safety compliance monitoring",
            "timestamp": req_timestamp
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["timestamp"] == req_timestamp
    assert data["latency_ms"] > 0.0
    assert data["detected_intent"] == "ppe_safety_compliance"
    assert "ppe" in data["selected_agents"]
    assert "ppe" in data["executed_agents"]
    assert data["annotated_frame_base64"] is not None
    assert data["annotated_frame_base64"].startswith("data:image/jpeg;base64,")
    assert data["synthesis"] is not None

def test_camera_query_intent_switching():
    """Verify that different inspection queries correctly trigger different specialist agents."""
    frame_b64 = create_synthetic_frame_base64()

    # Query for metal surface defects
    response = client.post(
        "/api/v1/camera/frame",
        json={
            "frame_base64": frame_b64,
            "query": "Inspect metal surface for scratches and cracks",
            "timestamp": 2.0
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "defect_inspection"
    assert "defect" in data["selected_agents"]
    assert "defect" in data["executed_agents"]
    assert "ppe" not in data["selected_agents"]

def test_video_temporal_violation_persistence_logic():
    """
    Verify IEEE research grounding:
    Do not claim a violation persists across time unless tracking evidence supports that conclusion!
    """
    # Track A: Observed in 4 frames, violation in 3 frames -> Persistent Violation
    track_persistent = ActiveTrack(
        track_id=1,
        class_label="person",
        bbox=[100, 100, 200, 300],
        confidence=0.92,
        frame_idx=0,
        timestamp=0.0
    )
    track_persistent.update([102, 100, 202, 300], 0.91, 1, 1.0, ppe_violation=True)
    track_persistent.update([105, 101, 205, 301], 0.90, 2, 2.0, ppe_violation=True)
    track_persistent.update([108, 102, 208, 302], 0.93, 3, 3.0, ppe_violation=True)

    is_persistent_A = (track_persistent.ppe_violation_count >= 2) or (
        track_persistent.frames_seen >= 2 and (track_persistent.ppe_violation_count / track_persistent.frames_seen) >= 0.5
    )
    assert is_persistent_A is True

    # Track B: Observed in 5 frames, violation in only 1 frame -> Intermittent / Single-Frame Anomaly
    track_transient = ActiveTrack(
        track_id=2,
        class_label="person",
        bbox=[300, 100, 400, 300],
        confidence=0.88,
        frame_idx=0,
        timestamp=0.0
    )
    track_transient.update([302, 100, 402, 300], 0.89, 1, 1.0, ppe_violation=True) # 1 violation
    track_transient.update([305, 101, 405, 301], 0.87, 2, 2.0, ppe_violation=False)
    track_transient.update([308, 102, 408, 302], 0.86, 3, 3.0, ppe_violation=False)
    track_transient.update([310, 103, 410, 303], 0.85, 4, 4.0, ppe_violation=False)

    is_persistent_B = (track_transient.ppe_violation_count >= 2) or (
        track_transient.frames_seen >= 2 and (track_transient.ppe_violation_count / track_transient.frames_seen) >= 0.5
    )
    assert is_persistent_B is False

def test_real_sample_video_incremental_tracking():
    """
    Verify incremental video processing, frame extraction, multi-object tracking,
    unique non-duplicate counts, and annotated frame persistence on real sample video.
    """
    if not os.path.exists(SAMPLE_VIDEO_PATH):
        pytest.skip(f"Real sample video not found at {SAMPLE_VIDEO_PATH}")

    with open(SAMPLE_VIDEO_PATH, "rb") as vid_file:
        response = client.post(
            "/api/v1/video/process",
            files={"video": ("sample_mall.mp4", vid_file, "video/mp4")},
            data={
                "query": "Count all visible people entering the zone",
                "sample_fps": "1.0",
                "max_frames": "4"
            }
        )

    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] is not None
    assert data["duration_sec"] > 0.0
    assert data["total_video_frames"] > 0
    assert data["sampled_frames_processed"] == 4
    assert data["sample_fps"] == 1.0
    assert data["total_latency_ms"] > 0.0
    assert data["total_unique_entities"] >= 0
    assert len(data["frame_results"]) == 4

    # Verify per-frame structure & annotation artifacts
    for fr in data["frame_results"]:
        assert "frame_index" in fr
        assert "timestamp_sec" in fr
        assert "per_frame_counts" in fr
        assert "findings" in fr
        assert fr["annotated_frame_url"].startswith("/storage/annotations/")
        
        # Verify file exists on disk
        local_ann_path = fr["annotated_frame_url"].replace("/storage/annotations/", f"{settings.ANNOTATIONS_DIR}/")
        assert os.path.exists(local_ann_path)

def test_video_sqlite_session_persistence():
    """Verify that video inspection results are correctly recorded in SQLite database."""
    if not os.path.exists(SAMPLE_VIDEO_PATH):
        pytest.skip(f"Real sample video not found at {SAMPLE_VIDEO_PATH}")

    with open(SAMPLE_VIDEO_PATH, "rb") as vid_file:
        response = client.post(
            "/api/v1/video/process",
            files={"video": ("sample_mall_db.mp4", vid_file, "video/mp4")},
            data={
                "query": "Track people and count total unique entities",
                "sample_fps": "1.0",
                "max_frames": "2"
            }
        )

    assert response.status_code == 200
    session_id = response.json()["session_id"]

    db = SessionLocal()
    try:
        session_row = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        assert session_row is not None
        assert session_row.input_type == "video"
        assert session_row.status == "completed"

        # Verify findings are recorded
        findings = db.query(Finding).filter(Finding.session_id == session_id).all()
        # Each finding should have track_id or frame_timestamp
        for f in findings:
            assert f.frame_timestamp is not None
    finally:
        db.close()
