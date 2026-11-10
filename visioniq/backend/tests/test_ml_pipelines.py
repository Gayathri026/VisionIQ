"""
VisionIQ Machine Learning Pipeline & Reproducibility Test Suite
Verifies:
1. Manifest integrity, versioning, classes, and research licenses.
2. Corrupt-file detection & zero-byte image handling.
3. Bounding-box label validation & normalized range constraints [0, 1].
4. Model registry catalog & checkpoint file persistence.
5. IEEE Access Model Cards structure and clarity.
6. Multi-modal evaluation metrics in metrics.json.
7. Experiment run tracking and persistence in SQLite.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from PIL import Image

from backend.app.database.session import SessionLocal, init_db
from backend.app.database.models import ModelVersion, ExperimentRun
from ml.datasets.prepare_datasets import (
    check_file_integrity,
    validate_yolo_label,
    PPE_CLASSES,
    NEU_CLASSES
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def test_dataset_manifest_integrity():
    """Verify dataset manifest exists and specifies required research fields."""
    manifest_path = PROJECT_ROOT / "ml" / "datasets" / "manifest.json"
    assert manifest_path.exists(), "Dataset manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "datasets" in manifest
    datasets = manifest["datasets"]

    # Verify all 3 specified datasets are defined
    assert "construction_site_safety" in datasets
    assert "neu_surface_defect" in datasets
    assert "mvtec_ad_benchmark" in datasets

    # Verify licenses
    assert "CC BY 4.0" in datasets["construction_site_safety"]["license"]
    assert "Academic" in datasets["neu_surface_defect"]["license"]
    assert "CC BY-NC-SA 4.0" in datasets["mvtec_ad_benchmark"]["license"]

    # Verify task types / formats
    assert datasets["construction_site_safety"]["annotation_format"] == "yolo_v11_normalized"
    assert datasets["neu_surface_defect"]["annotation_format"] == "folder_per_class"
    assert datasets["mvtec_ad_benchmark"]["annotation_format"] == "pixel_groundtruth_masks"

    # Verify classes
    assert len(datasets["construction_site_safety"]["classes"]) == 10
    assert len(datasets["neu_surface_defect"]["classes"]) == 6

def test_corrupt_file_detector():
    """Verify check_file_integrity correctly flags corrupt/zero-byte files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Valid image
        valid_img_path = tmp_path / "valid.jpg"
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(str(valid_img_path))
        assert check_file_integrity(str(valid_img_path)) is True

        # 2. Zero-byte file
        empty_img_path = tmp_path / "empty.jpg"
        empty_img_path.write_bytes(b"")
        assert check_file_integrity(str(empty_img_path)) is False

        # 3. Truncated corrupt image bytes
        corrupt_img_path = tmp_path / "corrupt.jpg"
        corrupt_img_path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xFF\xDBbadbytes")
        assert check_file_integrity(str(corrupt_img_path)) is False

        # 4. Non-existent file
        assert check_file_integrity(str(tmp_path / "nonexistent.jpg")) is False

def test_yolo_label_boundary_validation():
    """Verify validate_yolo_label enforces normalized [0, 1] bounds and class indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        num_classes = len(PPE_CLASSES)

        # Valid label
        valid_label = tmp_path / "valid.txt"
        valid_label.write_text("0 0.5000 0.5000 0.2000 0.3000\n5 0.1000 0.2000 0.0500 0.1500\n", encoding="utf-8")
        is_valid, errors = validate_yolo_label(str(valid_label), num_classes)
        assert is_valid is True
        assert len(errors) == 0

        # Out-of-bounds coordinate (> 1.0)
        invalid_coord = tmp_path / "invalid_coord.txt"
        invalid_coord.write_text("0 1.2500 0.5000 0.2000 0.3000\n", encoding="utf-8")
        is_valid, errors = validate_yolo_label(str(invalid_coord), num_classes)
        assert is_valid is False
        assert any("outside normalized [0, 1]" in err for err in errors)

        # Negative coordinate (< 0.0)
        negative_coord = tmp_path / "negative_coord.txt"
        negative_coord.write_text("0 -0.0500 0.5000 0.2000 0.3000\n", encoding="utf-8")
        is_valid, errors = validate_yolo_label(str(negative_coord), num_classes)
        assert is_valid is False
        assert any("outside normalized [0, 1]" in err for err in errors)

        # Invalid class ID (>= num_classes)
        invalid_cls = tmp_path / "invalid_cls.txt"
        invalid_cls.write_text(f"{num_classes} 0.5000 0.5000 0.2000 0.3000\n", encoding="utf-8")
        is_valid, errors = validate_yolo_label(str(invalid_cls), num_classes)
        assert is_valid is False
        assert any("Class ID" in err and "out of bounds" in err for err in errors)

def _resolve_checkpoint(raw_path_str: str) -> Path:
    """Helper to resolve checkpoint path from monorepo or workspace root."""
    p = Path(raw_path_str)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to PROJECT_ROOT
    if (PROJECT_ROOT / p).exists():
        return PROJECT_ROOT / p
    # If path starts with visioniq, strip it relative to PROJECT_ROOT
    parts = p.parts
    if len(parts) > 0 and parts[0] == "visioniq":
        stripped = Path(*parts[1:])
        if (PROJECT_ROOT / stripped).exists():
            return PROJECT_ROOT / stripped
    # Try relative to workspace root (PROJECT_ROOT.parent)
    if (PROJECT_ROOT.parent / p).exists():
        return PROJECT_ROOT.parent / p
    return PROJECT_ROOT / p

def test_model_registry_catalog():
    """Verify registry.json records correct model metadata and weights exist."""
    registry_path = PROJECT_ROOT / "ml" / "model_registry" / "registry.json"
    assert registry_path.exists(), "registry.json must exist"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert "models" in registry
    models = registry["models"]

    # Verify both trained models are registered
    assert "ppe_detector" in models
    assert "defect_classifier" in models
    assert "anomaly_localizer" in models

    # Check that weights exist on disk for ppe_detector
    ppe_ckpt = _resolve_checkpoint(models["ppe_detector"]["checkpoint_path"])
    assert ppe_ckpt.exists(), f"PPE weights missing at {ppe_ckpt}"
    assert ppe_ckpt.stat().st_size > 100_000, "PPE checkpoint must not be empty"

    # Check that weights exist on disk for defect_classifier
    defect_ckpt = _resolve_checkpoint(models["defect_classifier"]["checkpoint_path"])
    assert defect_ckpt.exists(), f"Defect weights missing at {defect_ckpt}"
    assert defect_ckpt.stat().st_size > 10_000, "Defect checkpoint must not be empty"

def test_model_cards_standards():
    """Verify IEEE Access Model Cards document architectures, parameters, and limitations."""
    card_ppe = PROJECT_ROOT / "ml" / "model_registry" / "MODEL_CARD_PPE.md"
    card_defect = PROJECT_ROOT / "ml" / "model_registry" / "MODEL_CARD_DEFECT.md"

    assert card_ppe.exists(), "MODEL_CARD_PPE.md must exist"
    assert card_defect.exists(), "MODEL_CARD_DEFECT.md must exist"

    ppe_content = card_ppe.read_text(encoding="utf-8")
    defect_content = card_defect.read_text(encoding="utf-8")

    for section in ["Model Details", "Intended Use", "Training Data & Methodology", "Evaluation Metrics", "Limitations"]:
        assert section in ppe_content, f"MODEL_CARD_PPE.md missing section '{section}'"
        assert section in defect_content, f"MODEL_CARD_DEFECT.md missing section '{section}'"

def test_evaluation_metrics_persisted():
    """Verify evaluation outputs in metrics.json have valid non-null numerical values."""
    metrics_path = PROJECT_ROOT / "ml" / "evaluation" / "metrics.json"
    assert metrics_path.exists(), "metrics.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "ppe_detector" in metrics
    assert "defect_classifier" in metrics
    assert "anomaly_localizer" in metrics

    # Check defect classifier metrics
    defect_m = metrics["defect_classifier"]
    assert isinstance(defect_m["accuracy"], float)
    assert 0.0 <= defect_m["accuracy"] <= 1.0
    assert isinstance(defect_m["macro_f1"], float)
    assert 0.0 <= defect_m["macro_f1"] <= 1.0
    assert "confusion_matrix" in defect_m
    assert len(defect_m["confusion_matrix"]["classes"]) == 6

    # Check anomaly localizer metrics
    anomaly_m = metrics["anomaly_localizer"]
    assert anomaly_m["anomaly_separation_ratio"] > 1.0
    assert anomaly_m["pixel_auroc"] >= 0.90

def test_experiment_runs_logged_in_db():
    """Verify evaluation run is recorded in SQLite database experiment_runs table."""
    init_db()
    db = SessionLocal()
    try:
        runs = db.query(ExperimentRun).all()
        assert len(runs) > 0, "At least one experiment run must be persisted in database"

        adaptive_runs = [r for r in runs if r.pipeline_type == "adaptive"]
        assert len(adaptive_runs) > 0, "At least one adaptive experiment run must exist in database"
        latest_run = adaptive_runs[-1]
        assert latest_run.pipeline_type == "adaptive"
        assert latest_run.dataset_name is not None
        assert latest_run.raw_metrics_json is not None
        assert isinstance(latest_run.raw_metrics_json, dict)

        # Verify ModelVersion entries exist
        model_versions = db.query(ModelVersion).all()
        assert len(model_versions) >= 2, "ModelVersion records for PPE and Defect must exist"
        model_names = [mv.model_name for mv in model_versions]
        assert "VisionIQ-PPE" in model_names
        assert "NEU-Defect" in model_names
    finally:
        db.close()
