import os
import pytest
from backend.orchestrator.router import route_query
from backend.orchestrator.graph import vision_iq_app
from backend.database.session import SessionLocal, init_db
from backend.database.models import AnalysisSession

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()

def test_cost_aware_routing():
    query = "Check if workers are wearing safety helmets and reflective vests"
    result = route_query(query)
    
    assert result["intent"] == "ppe_safety_compliance"
    assert "ppe" in result["selected_agents"]
    assert "object" in result["selected_agents"]  # Mandatory dependency
    assert "ocr" in result["skipped_agents"]      # Cost-aware exclusion
    
    # Verify traces exist for all available agents
    assert len(result["traces"]) == 9
    for agent_name, trace in result["traces"].items():
        assert "final_routing_score" in trace
        assert "decision" in trace
        assert trace["decision"] in ["SELECTED", "SKIPPED", "MANDATORY_DEPENDENCY", "FALLBACK"]

def test_ocr_routing():
    query = "Read the serial number and text on the metal machine nameplate"
    result = route_query(query)
    
    assert result["intent"] == "ocr_text_extraction"
    assert "ocr" in result["selected_agents"]
    # Heavy defect agent should be skipped to conserve compute and latency
    assert "defect" in result["skipped_agents"]

def test_end_to_end_pipeline():
    sample_img = "sample_data/FACTORY_SCENE.png"
    assert os.path.exists(sample_img), f"Sample image {sample_img} missing"

    state = {
        "image_path": sample_img,
        "user_query": "Check if workers are wearing safety helmets and reflective vests",
        "input_type": "image"
    }

    final_state = vision_iq_app.invoke(state)
    assert "final_response" in final_state
    
    resp = final_state["final_response"]
    assert resp["session_id"] is not None
    assert resp["detected_intent"] == "ppe_safety_compliance"
    assert len(resp["selected_agents"]) > 0
    assert len(resp["traces"]) > 0
    
    # Check PDF generation
    pdf_path = resp.get("pdf_report_path")
    assert pdf_path and os.path.exists(pdf_path), f"PDF report {pdf_path} was not generated"
    assert os.path.getsize(pdf_path) > 1000, "PDF report file is empty or corrupted"

    # Check database persistence
    db = SessionLocal()
    session_record = db.query(AnalysisSession).filter(AnalysisSession.id == resp["session_id"]).first()
    assert session_record is not None
    assert session_record.user_query == state["user_query"]
    assert len(session_record.findings) == len(resp["findings"])
    db.close()
