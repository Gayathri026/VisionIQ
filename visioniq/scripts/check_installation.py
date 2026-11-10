"""
VisionIQ Foundation Verification Script
Performs pre-flight integrity verification across Python packages, vision model weights,
SQLAlchemy database tables, and Node/NPM frontend toolchain.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

# Fix Windows console UTF-8 output encoding if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root and backend to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)

def check_python_environment():
    print_header("1. Python Runtime & Core Libraries")
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    print(" [OK] Python version compatible (>= 3.10)")

    packages = [
        ("torch", "PyTorch Deep Learning Engine"),
        ("torchvision", "PyTorch Computer Vision"),
        ("ultralytics", "Ultralytics YOLO Framework"),
        ("cv2", "OpenCV Image Processing"),
        ("fastapi", "FastAPI Asynchronous Framework"),
        ("uvicorn", "ASGI Server"),
        ("pydantic", "Pydantic Schema Validation"),
        ("sqlalchemy", "SQLAlchemy ORM"),
        ("reportlab", "ReportLab PDF Engine"),
        ("pytest", "Pytest Automated Testing"),
    ]

    missing = []
    for pkg, name in packages:
        try:
            __import__(pkg)
            print(f" [OK] {name:<35} ({pkg})")
        except ImportError as e:
            print(f" [FAIL] {name:<35} ({pkg}) - MISSING")
            missing.append(pkg)

    # Optional specialists
    for pkg, name in [("easyocr", "EasyOCR Text Recognition"), ("langgraph", "LangGraph StateGraph Engine"), ("faiss", "FAISS Vector Index")]:
        try:
            __import__(pkg)
            print(f" [OK] {name:<35} ({pkg})")
        except ImportError:
            print(f" [WARN] {name:<35} ({pkg}) - Will be loaded in Phase 2")

    return len(missing) == 0

def check_models():
    print_header("2. Model Registry & Weights")
    registry_path = PROJECT_ROOT / "ml" / "model_registry"
    yolo_weight = registry_path / "yolo11n.pt"

    if yolo_weight.exists():
        size_mb = yolo_weight.stat().st_size / (1024 * 1024)
        print(f" [OK] Baseline YOLO weights: {yolo_weight.name} ({size_mb:.2f} MB)")
        try:
            from ultralytics import YOLO
            model = YOLO(str(yolo_weight))
            print(f" [OK] Successfully instantiated YOLO model ({len(model.names)} COCO classes)")
        except Exception as e:
            print(f" [FAIL] Failed to load YOLO model: {e}")
            return False
    else:
        print(f" [WARN] Model weights not found at {yolo_weight}")
        return False
    return True

def check_database():
    print_header("3. Database Schema & Tables")
    try:
        from backend.app.database.session import init_db, engine
        from sqlalchemy import inspect
        init_db()
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
        all_present = True
        for t in expected_tables:
            if t in tables:
                print(f" [OK] Table present: {t}")
            else:
                print(f" [FAIL] Missing table: {t}")
                all_present = False
        return all_present
    except Exception as e:
        print(f" [FAIL] Database verification failed: {e}")
        return False

def check_frontend():
    print_header("4. Frontend Toolchain")
    node_path = shutil.which("node")
    npm_path = shutil.which("npm") or shutil.which("npm.cmd")

    if node_path:
        node_ver = subprocess.check_output(["node", "-v"], text=True, shell=True).strip()
        print(f" [OK] Node.js installed: {node_ver} ({node_path})")
    else:
        print(" [FAIL] Node.js not found in PATH")

    if npm_path:
        npm_ver = subprocess.check_output(["npm", "-v"], text=True, shell=True).strip()
        print(f" [OK] NPM installed: {npm_ver} ({npm_path})")
    else:
        print(" [FAIL] NPM not found in PATH")

    frontend_dir = PROJECT_ROOT / "frontend"
    pkg_json = frontend_dir / "package.json"
    dist_dir = frontend_dir / "dist"

    if pkg_json.exists():
        print(f" [OK] Frontend configuration present: {pkg_json.name}")
    else:
        print(" [FAIL] Frontend package.json missing")

    if dist_dir.exists():
        print(f" [OK] Production bundle compiled: {dist_dir.name}")
    else:
        print(" [WARN] Production bundle not yet built (run `npm run build` in frontend)")

    return bool(node_path and npm_path and pkg_json.exists())

def main():
    print("=" * 60)
    print(" VISIONIQ: SYSTEM INSTALLATION & PRE-FLIGHT VERIFIER")
    print(" IEEE Access Research Implementation & Academic Evaluation")
    print("=" * 60)

    p1 = check_python_environment()
    p2 = check_models()
    p3 = check_database()
    p4 = check_frontend()

    print_header("Final Verification Summary")
    if p1 and p2 and p3 and p4:
        print(" [OK] ALL CHECKS PASSED: VisionIQ Foundation is 100% operational.")
        sys.exit(0)
    else:
        print(" [WARN] Some checks failed or require attention. Review logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
