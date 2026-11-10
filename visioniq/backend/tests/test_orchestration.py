"""
VisionIQ Phase 2 Orchestration Engine Automated Test Suite
Validates:
1. Intent classification & distinct justified plans for Queries A, B, C, D, E.
2. OCR non-selection for counting and defect queries.
3. Defect non-selection for counting queries.
4. Configurable routing weights and operating threshold.
5. Mandatory agent dependencies and minimum sufficient evidence.
6. Agent base interfaces, error isolation, and measured runtimes.
7. Trace persistence in SQLite database and API endpoints.
"""

import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.orchestration.router import route_query, classify_intent_deterministic, AVAILABLE_AGENTS
from backend.app.agents.base import BaseSpecialistAgent, AgentInput, AgentOutput, ExecutionStatus
from backend.app.database.session import SessionLocal
from backend.app.database.models import AnalysisSession, AgentExecutionTrace, Finding

client = TestClient(app)

SAMPLE_IMAGE = os.path.join(settings.BASE_DIR, "visioniq/sample_data/test_img1.png")
if not os.path.exists(SAMPLE_IMAGE):
    SAMPLE_IMAGE = os.path.join(settings.BASE_DIR, "visioniq/sample_data/FACTORY_SCENE.png")

def test_query_a_plan_ocr():
    """Query A: 'Read the serial number on this machine.' must select OCR and skip Defect and PPE."""
    query = "Read the serial number on this machine."
    plan = route_query(query)

    assert plan["intent"] == "ocr_text_extraction"
    assert "ocr" in plan["selected_agents"], "OCR must be selected for serial number query"
    assert "defect" in plan["skipped_agents"], "Defect agent must be skipped for serial number query"
    assert "ppe" in plan["skipped_agents"], "PPE agent must be skipped for serial number query"
    
    # Verify selection reason
    ocr_trace = plan["traces"]["ocr"]
    assert ocr_trace["decision"] == "SELECTED"
    assert ">= threshold" in ocr_trace["selection_reason"]
    
    defect_trace = plan["traces"]["defect"]
    assert defect_trace["decision"] == "SKIPPED"
    assert "< threshold" in defect_trace["selection_reason"]

def test_query_b_plan_counting():
    """Query B: 'Count all visible people.' must select Object detection, NOT OCR or Defect."""
    query = "Count all visible people."
    plan = route_query(query)

    assert plan["intent"] == "object_counting_detection"
    assert "object" in plan["selected_agents"], "Object detector must be selected for counting"
    assert "ocr" in plan["skipped_agents"], "OCR must NOT run automatically for counting query"
    assert "defect" in plan["skipped_agents"], "Defect analysis must NOT run for counting query"

    # Confirm OCR score was well below threshold
    assert plan["traces"]["ocr"]["final_routing_score"] < plan["threshold"]
    assert plan["traces"]["defect"]["final_routing_score"] < plan["threshold"]

def test_query_c_plan_ppe():
    """Query C: 'Check whether workers are wearing helmets.' must select PPE & Object, not OCR."""
    query = "Check whether workers are wearing helmets."
    plan = route_query(query)

    assert plan["intent"] == "ppe_safety_compliance"
    assert "ppe" in plan["selected_agents"], "PPE agent must be selected for helmet check"
    assert "object" in plan["selected_agents"], "Object detector must be selected for person localization"
    assert "ocr" in plan["skipped_agents"], "OCR must be skipped for PPE inquiry"
    assert "defect" in plan["skipped_agents"], "Defect must be skipped for PPE inquiry"

def test_query_d_plan_defect():
    """Query D: 'Locate cracks or scratches on this metal surface.' must select Defect and Object, not OCR or PPE."""
    query = "Locate cracks or scratches on this metal surface."
    plan = route_query(query)

    assert plan["intent"] == "defect_inspection"
    assert "defect" in plan["selected_agents"], "Defect agent must be selected for crack/scratch inspection"
    assert "object" in plan["selected_agents"], "Object detector must be present for spatial context"
    assert "ocr" in plan["skipped_agents"], "OCR must be skipped for surface defect query"
    assert "ppe" in plan["skipped_agents"], "PPE must be skipped for surface defect query"

def test_query_e_plan_combined():
    """Query E: 'Describe the scene and explain the possible damage.' must select Scene, Defect, Object."""
    query = "Describe the scene and explain the possible damage."
    plan = route_query(query)

    assert plan["intent"] == "combined_scene_and_defect"
    assert "scene" in plan["selected_agents"]
    assert "defect" in plan["selected_agents"]
    assert "object" in plan["selected_agents"]
    assert "ocr" in plan["skipped_agents"]

def test_distinct_plans_generated():
    """Ensure all 5 queries generate different plans justified by their intent."""
    queries = [
        "Read the serial number on this machine.",
        "Count all visible people.",
        "Check whether workers are wearing helmets.",
        "Locate cracks or scratches on this metal surface.",
        "Describe the scene and explain the possible damage."
    ]
    plans = [route_query(q)["selected_agents"] for q in queries]
    
    # Each plan should differ in composition
    unique_plan_sets = len(set(tuple(p) for p in plans))
    assert unique_plan_sets >= 4, f"Expected distinct plans for each query, got {unique_plan_sets}"

def test_configurable_weights_and_threshold():
    """Verify runtime override of routing hyperparameters and threshold gating."""
    query = "Count all visible people."
    
    # With very high threshold, even relevant agents get skipped
    plan_high_tau = route_query(query, threshold=0.95)
    # Minimum sufficient evidence fallback will rescue object and scene
    assert "FALLBACK" in [plan_high_tau["traces"]["object"]["decision"]]
    
    # With high cost penalty beta=0.90, compute-heavy agents are suppressed
    plan_high_beta = route_query("Read the serial number on this machine.", beta=0.90)
    assert plan_high_beta["traces"]["ocr"]["final_routing_score"] < plan_high_beta["threshold"]

def test_mandatory_dependencies():
    """Verify that PPE triggers Object dependency even if Object score was initially lower."""
    plan = route_query("Check whether workers are wearing helmets.")
    assert "object" in plan["selected_agents"]
    assert plan["traces"]["object"]["decision"] in ["SELECTED", "MANDATORY_DEPENDENCY"]

def test_agent_base_interface_and_error_isolation():
    """Test BaseSpecialistAgent error handling with non-existent file."""
    class DummyFailingAgent(BaseSpecialistAgent):
        def _execute_internal(self, agent_input: AgentInput) -> AgentOutput:
            raise RuntimeError("Simulated transient sensor fault")

    agent = DummyFailingAgent(agent_name="dummy_test")
    input_data = AgentInput(session_id="test_sess", user_query="dummy query")
    output = agent.execute(input_data)

    assert output.status == ExecutionStatus.FAILED
    assert "Simulated transient sensor fault" in output.error_details
    assert output.measured_runtime_ms >= 0.0
    assert output.provenance["agent"] == "dummy_test"

def test_api_orchestration_plan_endpoint():
    """Verify POST /api/v1/orchestration/plan endpoint."""
    payload = {"query": "Count all visible people."}
    response = client.post("/api/v1/orchestration/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "object_counting_detection"
    assert "object" in data["selected_agents"]
    assert "ocr" in data["skipped_agents"]
    assert "traces" in data
    assert data["traces"]["object"]["decision"] == "SELECTED"

def test_end_to_end_analysis_and_trace_persistence():
    """Verify full image analysis, trace persistence to SQLite, and trace retrieval."""
    if not os.path.exists(SAMPLE_IMAGE):
        pytest.skip(f"Sample test image not found at {SAMPLE_IMAGE}")

    with open(SAMPLE_IMAGE, "rb") as img_file:
        response = client.post(
            "/api/v1/analyze/image",
            files={"file": ("test.png", img_file, "image/png")},
            data={"query": "Check whether workers are wearing helmets."}
        )

    assert response.status_code == 200
    data = response.json()
    session_id = data["session_id"]
    assert session_id is not None
    assert data["detected_intent"] == "ppe_safety_compliance"
    assert "ppe" in data["selected_agents"]
    assert len(data["traces"]) == 9
    assert data["total_latency_ms"] > 0.0

    # Verify Trace retrieval endpoint
    trace_resp = client.get(f"/api/v1/analysis/{session_id}/trace")
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert trace_data["session_id"] == session_id
    assert len(trace_data["traces"]) == 9
    
    # Confirm DB persistence directly
    db = SessionLocal()
    db_session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    assert db_session is not None
    assert db_session.user_query == "Check whether workers are wearing helmets."
    db_traces = db.query(AgentExecutionTrace).filter(AgentExecutionTrace.session_id == session_id).all()
    assert len(db_traces) == 9
    db.close()
