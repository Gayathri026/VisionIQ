import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_assistant_canonical_questions():
    """Verify that the VisionIQ Assistant answers all 8 required canonical questions accurately."""
    canonical_queries = [
        "Which analysis mode should I use?",
        "Why did VisionIQ select these agents?",
        "What does this confidence score mean?",
        "How do I train the PPE model?",
        "How can I add a new technical document to RAG?",
        "Why could the damage cause not be determined?",
        "How do I interpret the evaluation metrics?",
        "How do I export and understand the PDF report?"
    ]

    for q in canonical_queries:
        response = client.post("/api/v1/assistant/query", json={"query": q})
        assert response.status_code == 200, f"Failed for query: {q}"
        data = response.json()
        assert data["status"] == "success"
        assert len(data["answer"]) > 100
        assert data["mode"] == "deterministic_verified"
        assert len(data["source_references"]) > 0
        assert data["recommended_navigation"] is not None
        assert len(data["suggested_followups"]) > 0

def test_assistant_context_aware_active_session():
    """Verify that the assistant explains the actual active analysis session and agent trace."""
    mock_trace = [
        {
            "agent_name": "agent_c_ppe",
            "decision": "SELECTED",
            "final_routing_score": 0.485,
            "selection_reason": "High relevance for safety inspection intent",
            "actual_latency_ms": 112.4,
            "execution_status": "COMPLETED"
        },
        {
            "agent_name": "agent_e_defect",
            "decision": "SKIPPED",
            "final_routing_score": 0.120,
            "selection_reason": "Below threshold 0.35"
        }
    ]

    req_payload = {
        "query": "Why did VisionIQ select these agents for this analysis?",
        "session_id": "test-session-550e8400",
        "current_trace": mock_trace,
        "active_finding_uid": "F-PPE-01"
    }

    response = client.post("/api/v1/assistant/query", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Should explain agent_c_ppe and agent_e_defect scores and decision
    assert "agent_c_ppe" in data["answer"]
    assert "0.485" in data["answer"]
    assert "SKIPPED" in data["answer"]

def test_assistant_faq_list_endpoint():
    """Verify the assistant FAQ endpoint returns all canonical engineering questions."""
    response = client.get("/api/v1/assistant/faq")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_faqs"] == 8
    assert any("PPE" in q["question"] for q in data["questions"])
    assert any("mode" in q["question"] for q in data["questions"])

def test_experiments_endpoint_and_pipeline_comparisons():
    """Verify stored experiments and the comparison table among Fixed, Rule-Based, and Adaptive pipelines."""
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "pipeline_comparisons" in data
    assert len(data["pipeline_comparisons"]) == 3

    # Check Fixed, Rule-based, and Adaptive present
    types = [p["pipeline_type"] for p in data["pipeline_comparisons"]]
    assert "fixed" in types
    assert "rule_based" in types
    assert "adaptive" in types

    # At least adaptive is completed or missing ones are clearly labeled not_yet_run
    for p in data["pipeline_comparisons"]:
        assert p["status"] in ("completed", "not_yet_run")
        if p["status"] == "not_yet_run":
            assert "not" in p.get("note", "").lower() or "pending" in p.get("note", "").lower()

    # Active models and datasets
    assert len(data["active_models"]) >= 2
    assert len(data["datasets"]) == 3
