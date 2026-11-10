# VisionIQ: Environment Setup & Deployment Guide

This document provides step-by-step instructions for configuring, installing, verifying, and deploying the **VisionIQ** multi-agent vision intelligence platform from scratch.

---

## 1. System Requirements & Hardware Guidelines

### Supported Operating Systems
- **Windows:** Windows 10 / 11 (64-bit) (PowerShell 7+ or Command Prompt)
- **Linux:** Ubuntu 22.04 LTS / 24.04 LTS or Debian-based distributions
- **macOS:** macOS Sonoma (Apple Silicon M-series or Intel x86_64)

### Hardware Requirements
| Component | Minimum Requirement | Recommended Specification |
|---|---|---|
| **CPU** | 4-Core x86_64 / ARM64 (Intel Core i5 / Ryzen 5 / Apple M1) | 8-Core (Intel Core i7 / Ryzen 7 / Apple M2 Pro) |
| **RAM** | 8 GB DDR4 | 16 GB DDR4/DDR5 |
| **GPU (Optional)** | Integrated Graphics (CPU inference supported) | NVIDIA RTX 3060+ (CUDA 12.1+ / TensorRT) |
| **Disk Space** | 4 GB free disk space | 10 GB free SSD storage |
| **Network** | Offline supported after initial setup | Internet connection for pip/npm packages |

---

## 2. Software Prerequisites

Ensure the following runtimes are installed on your machine:

1. **Python 3.12 (64-bit)**
   - Verify version: `python --version` (Output must be `Python 3.12.x`)
2. **Node.js (v18.0.0 or higher, LTS recommended)**
   - Verify version: `node --version` (e.g., `v20.x` or `v22.x`)
3. **npm (v9.0.0 or higher)**
   - Verify version: `npm --version`
4. **Git**
   - Verify version: `git --version`

---

## 3. Repository Cloning and Layout

Clone the repository to your local workspace:

```bash
git clone https://github.com/your-org/visioniq.git
cd gmd_final_year_proj
```

The monorepo structure is organized into clean functional tiers:

```
visioniq/
├── backend/            # FastAPI Python 3.12 backend & LangGraph agents
├── frontend/           # React 18 / Vite / TypeScript web application
├── ml/                 # PyTorch models, weights, datasets, training scripts
├── knowledge_base/     # Authenticated engineering standards (OSHA, ASTM, ISO)
├── experiments/        # Empirical benchmark test sets and generated plots
├── storage/            # Local SQLite database, uploads, and PDF reports
└── docs/               # Architecture diagrams, API specs, IEEE Access paper
```

---

## 4. Backend Virtual Environment Setup

### 4.1 Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
cd visioniq\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS (Bash / Zsh):**
```bash
cd visioniq/backend
python3 -m venv venv
source venv/bin/activate
```

### 4.2 Install Core Python Dependencies

Upgrade pip and install pinned dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key core libraries installed:
- `fastapi==0.115.0`, `uvicorn==0.31.0` (REST API & ASGI server)
- `langgraph==0.2.20`, `langchain-core` (StateGraph agent orchestration)
- `torch>=2.4.0`, `torchvision>=0.19.0` (PyTorch inference engine)
- `ultralytics>=8.3.0` (YOLO11 object detection & PPE classification)
- `easyocr>=1.7.0` (Optical Character Recognition)
- `reportlab>=4.2.0` (Production PDF generation)
- `sqlalchemy>=2.0.0` (Relational ORM for SQLite/PostgreSQL)
- `pytest>=8.3.0` (Automated testing framework)

### 4.3 Configure Environment Variables

Copy the provided `.env.example` into `.env`:
```bash
# From visioniq/ root:
cp .env.example .env
```

Review and adjust settings in `.env` if necessary:
```ini
VISIONIQ_ENV=development
SECRET_KEY=visioniq-secret-key-change-in-production
DATABASE_URL=sqlite:///./storage/visioniq.db
ROUTER_ALPHA=0.50
ROUTER_DELTA=0.15
ROUTER_BETA=0.20
ROUTER_GAMMA=0.15
ROUTER_THRESHOLD=0.45
STORAGE_DIR=./storage
```

---

## 5. Machine Learning Models & Datasets Setup

VisionIQ comes with pre-trained checkpoints in `visioniq/ml/model_registry/`:
- `yolo11n.pt` — General object and person detector
- `yolo_ppe.pt` — Specialized multi-person PPE detector (Hardhat, Safety Vest, Mask)
- `neu_defect_classifier.pt` — NEU Surface Defect CNN (Crazing, Inclusions, Pitted Surface, etc.)

To re-verify or regenerate datasets:
```bash
python ml/datasets/prepare_datasets.py
```

To run a fast pre-flight check across all vision weights:
```bash
python -c "from backend.app.models.model_loader import get_model_registry; reg = get_model_registry(); print(reg.list_models())"
```

---

## 6. Frontend Setup & Build

### 6.1 Install Frontend Dependencies

```bash
cd visioniq/frontend
npm install
```

### 6.2 Launch Vite Development Server

```bash
npm run dev
```
The application will be live at `http://localhost:5173`.

### 6.3 Compile Production Assets

To compile the production bundle:
```bash
npm run build
```
Compiled assets will be output to `visioniq/frontend/dist/`.

---

## 7. Running the Full System

### Start Backend Server
```bash
# From workspace root:
cd visioniq/backend
# Activate your virtual environment first
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API Base: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`
- System Health Diagnostics: `http://localhost:8000/api/v1/health`

### Start Frontend Application
In a separate terminal:
```bash
cd visioniq/frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 8. Automated Verification & Testing

VisionIQ features an automated test suite with **74 tests across 10 test suites**:

```bash
# Run all unit, integration, and end-to-end tests
pytest visioniq/backend/tests/ -v

# Run only the 12-workflow end-to-end audit
pytest visioniq/backend/tests/test_end_to_end_audit.py -v
```

Expected result:
```
====================== 74 passed in 30.99s =======================
```

---

## 9. Troubleshooting & FAQ

1. **Permission Denied on PowerShell:**
   Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` to allow activating the virtual environment.
2. **Missing Weights:**
   If `yolo_ppe.pt` is not found, verify that `visioniq/ml/model_registry/` contains the `.pt` checkpoints. The system automatically initializes standard fallbacks if optional weights are missing.
3. **Port 8000 or 5173 Already in Use:**
   Change the port in `uvicorn` using `--port 8080`, and update `VITE_API_URL` in `visioniq/frontend/.env` accordingly.
