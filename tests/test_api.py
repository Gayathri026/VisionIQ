import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "routing_parameters" in data
    assert data["routing_parameters"]["alpha_relevance"] == 0.50

def test_rag_documents_endpoint():
    response = client.get("/api/v1/rag/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] > 0
    assert len(data["documents"]) > 0
    assert "OSHA" in data["documents"][0]["title"]

def test_evaluation_runs_endpoint():
    response = client.get("/api/v1/evaluation/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "runs" in data

def test_image_analysis_api():
    sample_img = "sample_data/FACTORY_SCENE.png"
    assert os.path.exists(sample_img)

    with open(sample_img, "rb") as f:
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("test.png", f, "image/png")},
            data={"query": "Check if workers are wearing safety helmets and reflective vests"}
        )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    data = res["data"]
    assert "session_id" in data
    assert len(data["selected_agents"]) > 0
    assert len(data["traces"]) > 0
    assert data["annotated_url"] is not None
