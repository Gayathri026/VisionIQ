import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session import SessionLocal
from backend.app.database.models import ExperimentRun

client = TestClient(app)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_eval_testset_integrity():
    """Verifies held-out benchmark test set schema and session isolation."""
    testset_path = os.path.join(PROJECT_ROOT, "experiments", "benchmark", "eval_testset.json")
    assert os.path.exists(testset_path), "eval_testset.json missing."

    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 20, f"Expected at least 20 test cases, found {len(data)}"

    session_ids = set()
    for item in data:
        assert "id" in item
        assert "session_id" in item
        assert "query" in item
        assert "ground_truth_intent" in item
        assert "required_agents" in item
        assert "rubric" in item
        assert len(item["required_agents"]) > 0
        session_ids.add(item["session_id"])

    # Ensure diverse capture sessions to prevent duplicate leakage
    assert len(session_ids) >= 5, f"Expected at least 5 distinct capture sessions, found {len(session_ids)}"

def test_benchmark_summary_results():
    """Verifies benchmark_summary.json contains empirical evaluation metrics."""
    summary_path = os.path.join(PROJECT_ROOT, "experiments", "results", "benchmark_summary.json")
    assert os.path.exists(summary_path), "benchmark_summary.json missing."

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for p in ["fixed", "rule_based", "adaptive"]:
        assert p in summary, f"Pipeline {p} missing from benchmark summary."
        metrics = summary[p]
        assert metrics["total_queries"] >= 20
        assert 0.0 <= metrics["routing_f1"] <= 1.0
        assert metrics["median_latency_ms"] > 0
        assert metrics["mean_agents_invoked"] > 0

    # Verify Equation (1) Adaptive pipeline exhibits higher Routing F1 than Fixed
    assert summary["adaptive"]["routing_f1"] > summary["fixed"]["routing_f1"]
    # Verify Adaptive invokes significantly fewer agents than Fixed
    assert summary["adaptive"]["mean_agents_invoked"] < summary["fixed"]["mean_agents_invoked"]

def test_get_experiments_api_reflects_db():
    """Verifies GET /api/v1/experiments serves real completed empirical runs from SQLite."""
    res = client.get("/api/v1/experiments")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_experiments_recorded"] >= 3

    comparisons = {c["pipeline_type"]: c for c in data["pipeline_comparisons"]}
    assert "fixed" in comparisons
    assert "rule_based" in comparisons
    assert "adaptive" in comparisons

    # Verify status is now 'completed' instead of 'not_yet_run'
    assert comparisons["fixed"]["status"] == "completed"
    assert comparisons["rule_based"]["status"] == "completed"
    assert comparisons["adaptive"]["status"] == "completed"

def test_publication_plots_exist():
    """Verifies high-resolution publication figures generated on disk."""
    plots_dir = os.path.join(PROJECT_ROOT, "experiments", "plots")
    expected_plots = [
        "fig1_latency_vs_f1.png",
        "fig2_agent_invocations.png",
        "fig3_latency_distribution.png",
        "fig4_ablation_breakdown.png"
    ]
    for plot_name in expected_plots:
        p = os.path.join(plots_dir, plot_name)
        assert os.path.exists(p), f"Plot {plot_name} not found in {plots_dir}"
        assert os.path.getsize(p) > 5000, f"Plot {plot_name} appears empty or corrupted."
