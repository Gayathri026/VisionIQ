# VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration and Cost-Aware Query Routing

[![Evaluation Status](https://img.shields.io/badge/All_10_Phases-Complete-emerald.svg)](#)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](#)
[![React](https://img.shields.io/badge/React-18.3.1-61dafb.svg)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6.3-3178c6.svg)](#)
[![Tests Passing](https://img.shields.io/badge/Tests-74%2F74%20Passing-brightgreen.svg)](#)
[![Target Publication](https://img.shields.io/badge/Target-IEEE_Access-00629B.svg)](#)

> **Final-Year Engineering Project (Course Code: 23CSP701)**  
> **Department of Computer Science and Engineering**  
> **Dr. Mahalingam College of Engineering and Technology (Autonomous), Pollachi**  
> **Affiliated with Anna University, Chennai**  
> **Target Journal:** IEEE Access  

---

## 🚀 Key Innovations & Architectural Highlights

1. **Cost-Aware Dynamic Query Routing (Equation 1)**:
   $$S(a, q) = \alpha R(a,q) + \delta E(a) - \beta C(a) - \gamma L(a)$$
   Balances semantic relevance $\alpha = 0.50$ and verifiable evidence capability $\delta = 0.15$ against computational FLOPs $\beta = 0.20$ and latency penalty $\gamma = 0.15$ (with gating threshold $\tau = 0.45$).
2. **LangGraph StateGraph Dynamic Orchestration**:
   Orchestrates 12 specialized vision and cognitive agents with conditional routing, fault isolation, and graceful fallback.
3. **Specialist Neural Agents**:
   - **YOLO11n PPE Specialist**: Fine-tuned on construction workplace safety gear with anatomical spatial zoning (`W-01`, `W-02`) separating cranial and torso zones.
   - **NEU DefectNet CNN**: 6-class hot-rolled steel surface defect classifier.
   - **EasyOCR Specialist**: Multi-line industrial text extraction on nameplates and gauges.
4. **Temporal Video Tracking & Deduplication**:
   Configurable frame sampling interval and IoU multi-object tracking to distinguish instantaneous detections from unique tracked personnel across frames.
5. **Real-Time WebRTC Live Camera Stream**:
   Browser camera integration with frame-by-frame inference, lifecycle controls (start, pause, resume, stop), and resource teardown.
6. **Regulatory Knowledge RAG & 5-Layer Epistemology**:
   Hybrid vector store (TF-IDF + BM25) with authentic OSHA 1926, ASTM E165, and ISO standards, two-way citation verification, and strict epistemological separation (Observations vs. Predictions vs. Standards vs. Hypotheses vs. Actions).
7. **Production ReportLab PDF Synthesis**:
   Two-page research-grade inspection reports with `NumberedCanvas` ("Page X of Y"), proportional visual scaling, and path traversal protection.
8. **Empirically Proven on IEEE Benchmark**:
   Evaluated on held-out test suite ($N=24$ across 6 sessions):
   - **60.38% reduction in model invocations** (3.17 vs. 8.00 agents).
   - **50.68% reduction in p95 tail latency** ($3206.41\text{ ms}$ vs. $6501.74\text{ ms}$).
   - **+30.65% relative gain in routing F1-score** (0.5806 vs. 0.4444).

---

## 🛠️ Quick Start Guide

### 1. Prerequisites
- Python 3.12 (64-bit)
- Node.js v18+ and npm

### 2. Environment Setup
Please refer to the full setup guide in [`visioniq/SETUP.md`](visioniq/SETUP.md).

```bash
# Clone the repository
git clone https://github.com/your-org/visioniq.git
cd gmd_final_year_proj

# Backend Setup
cd visioniq/backend
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
cp ../.env.example ../.env

# Launch Backend Server (http://localhost:8000)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
In a separate terminal:
```bash
cd visioniq/frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

### 4. Automated Verification Test Suite
```bash
# Run all 74 unit, integration, and end-to-end audit tests
pytest visioniq/backend/tests -v
```

---

## 📁 Repository Structure

```
gmd_final_year_proj/
├── visioniq/
│   ├── backend/               # FastAPI backend & LangGraph agents
│   │   ├── app/               # Core, database, schemas, agents, RAG, reporting
│   │   └── tests/             # 10 Pytest suites (74 tests passing)
│   ├── frontend/              # React 18 + Vite + TypeScript graphite workspace
│   ├── ml/                    # Models, checkpoints (yolo_ppe.pt, neu_defect_classifier.pt), datasets
│   ├── knowledge_base/        # Authentic engineering standards (OSHA, ASTM, ISO)
│   ├── experiments/           # Benchmark dataset (eval_testset.json), runner, publication plots
│   ├── docs/                  # Architecture diagrams, API reference, IEEE Access manuscript
│   ├── storage/               # SQLite database, uploads, and ReportLab PDF reports
│   ├── SETUP.md               # Complete environment setup guide
│   └── README.md              # Project monorepo guide
├── README.md                  # Root project overview
└── .env.example               # Environment variables template
```

---

## 👥 Authors and Project Team

- **Gayathri R** (Roll No: 727623BCS065) — BE CSE, Dr. MCET
- **Monica T S** (Roll No: 727623BCS083) — BE CSE, Dr. MCET
- **Dhamodharan R** (Roll No: 727624BCS308) — BE CSE, Dr. MCET
- **Project Supervisor:** **Ms. Gayathri S**, Assistant Professor, Department of Computer Science and Engineering
- **Head of Department:** **Dr. D. Sivaganesan**, Professor & Head, Department of Computer Science and Engineering
