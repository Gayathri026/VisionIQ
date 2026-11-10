import pytest
from fastapi.testclient import TestClient
from backend.app.core.config import settings
from backend.app.database.session import init_db, engine
from backend.app.database.models import (
    AnalysisSession, Finding, AgentExecutionTrace, ModelVersion, RAGSource, ReportRecord, ExperimentRun
)
from backend.app.main import app

client = TestClient(app)

def test_config_validation():
    """Verify configuration parameters and routing hyperparameter bounds."""
    assert settings.PROJECT_NAME is not None
    assert settings.VERSION == "1.0.0"
    assert 0.0 <= settings.ROUTING_ALPHA <= 1.0
    assert 0.0 <= settings.ROUTING_BETA <= 1.0
    assert 0.0 <= settings.ROUTING_GAMMA <= 1.0
    assert 0.0 <= settings.ROUTING_DELTA <= 1.0
    assert settings.ROUTING_THRESHOLD == 0.45
    assert len(settings.AGENT_COSTS) == 11
    assert len(settings.AGENT_LATENCIES) == 11

def test_database_initialization():
    """Verify table creation for all 7 foundation entities."""
    init_db()
    table_names = engine.table_names() if hasattr(engine, "table_names") else []
    # Using inspect for SQLAlchemy 2.0
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    expected_tables = [
        "analysis_sessions",
        "findings",
        "agent_execution_traces",
        "model_versions",
        "rag_sources",
        "report_records",
        "experiment_runs"
    ]
    for table in expected_tables:
        assert table in tables, f"Expected table '{table}' not found in SQLite schema"

def test_health_endpoint():
    """Verify that health endpoint returns complete diagnostics matching contract."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["database_connected"] is True
    assert "models" in data
    assert "routing_parameters" in data
    assert data["routing_parameters"]["alpha_relevance"] == 0.50
    assert data["routing_parameters"]["threshold_tau"] == 0.45

def test_history_endpoint():
    """Verify history endpoint returns structured list."""
    response = client.get("/api/v1/history")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "sessions" in data
    assert isinstance(data["sessions"], list)

def test_root_endpoint():
    """Verify API root metadata endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Phase 1: Foundation" in data["phase"]
    assert data["docs"] == "/docs"
