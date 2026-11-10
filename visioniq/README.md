# VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration and Cost-Aware Query Routing

[![Evaluation Status](https://img.shields.io/badge/All_10_Phases-Complete-emerald.svg)](#)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](#)
[![React](https://img.shields.io/badge/React-18.3.1-61dafb.svg)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6.3-3178c6.svg)](#)
[![Tests Passing](https://img.shields.io/badge/Tests-74%2F74%20Passing-brightgreen.svg)](#)
[![Target Publication](https://img.shields.io/badge/Target-IEEE_Access-00629B.svg)](#)

---

## 1. Academic & Institutional Provenance

- **Academic Context:** Project Work Phase-I & II (Course Code: **23CSP701**)
- **Degree Program:** Bachelor of Engineering in Computer Science and Engineering
- **Institution:** **Dr. Mahalingam College of Engineering and Technology (Autonomous)**, Pollachi – 642 003, Tamil Nadu, India
- **Affiliation:** **Anna University**, Chennai
- **Project Team:**
  - **Gayathri R** (Roll No: 727623BCS065)
  - **Monica T S** (Roll No: 727623BCS083)
  - **Dhamodharan R** (Roll No: 727624BCS308)
- **Faculty Supervisor:** **Ms. Gayathri S**, Assistant Professor, Department of Computer Science and Engineering
- **Head of Department:** **Dr. D. Sivaganesan**, Professor & Head, Department of Computer Science and Engineering
- **Target Journal:** **IEEE Access**
- **Manuscript Draft:** [`docs/paper/IEEE_Access_VisionIQ_Manuscript.md`](docs/paper/IEEE_Access_VisionIQ_Manuscript.md)
- **Limitations & Ethical Report:** [`docs/paper/LIMITATIONS_REPORT.md`](docs/paper/LIMITATIONS_REPORT.md)

---

## 2. Research Problem & Mathematical Formulation

Traditional computer vision architectures deploy rigid monolithic pipelines or static ensembles. In real-world industrial environments (such as manufacturing quality assurance and occupational safety inspection), executing every model on every input is computationally wasteful, incurs excessive tail latency, and produces irrelevant predictions. Conversely, naive heuristic routing lacks mathematical rigor and fails to balance semantic relevance against computational overhead.

**VisionIQ** addresses this challenge through an adaptive multi-agent framework powered by **LangGraph dynamic orchestration** and **cost-aware query routing**. The router evaluates each specialist agent $a$ for a given user query $q$ using a multi-criteria utility objective:

$$S(a, q) = \alpha R(a, q) + \delta E(a) - \beta C(a) - \gamma L(a)$$

### Calibrated Routing Hyperparameters:
- **$\alpha = 0.50$ (Relevance Weight):** Measures semantic query alignment $R(a, q) \in [0, 1]$ via cosine similarity and intent matching.
- **$\delta = 0.15$ (Evidence Capability):** Measures historical precision and verifiable grounding capability $E(a) \in [0, 1]$.
- **$\beta = 0.20$ (Compute Cost Penalty):** Normalized FLOPs and compute overhead $C(a) \in [0, 1]$.
- **$\gamma = 0.15$ (Latency Penalty):** Normalized execution latency $L(a) \in [0, 1]$.
- **Constraint:** $\alpha + \delta + \beta + \gamma = 1.0$.
- **Gating Decision Rule:** Agent $a$ is scheduled if and only if $S(a, q) \ge \tau$, where the decision threshold $\tau = 0.45$.

---

## 3. System Architecture & Multi-Agent Orchestration

```mermaid
flowchart TD
    User([User Request / Media Upload]) --> Ingest[Input Validation & Safe Storage]
    Ingest --> StateGraph[LangGraph Adaptive Orchestrator]

    subgraph Cost_Aware_Routing [Equation 1 Dynamic Router]
        StateGraph --> Intent[Agent A: Intent & Uncertainty Analyzer]
        Intent --> UtilityCalc["Compute S(a,q) = αR + δE - βC - γL"]
        UtilityCalc --> Gate{S(a,q) >= 0.45?}
    end

    subgraph Specialist_Agents [Specialist Vision & Cognitive Agents]
        Gate -- Yes --> AgentB[Agent B: Object Detector - YOLO11n]
        Gate -- Yes --> AgentC[Agent C: PPE Compliance - Spatial Zoning]
        Gate -- Yes --> AgentD[Agent D: OCR Extraction - EasyOCR]
        Gate -- Yes --> AgentE[Agent E: Industrial Defect - NEU ResNet]
        Gate -- Yes --> AgentF[Agent F: Spatial Scene Understanding]
        Gate -- Yes --> AgentG[Agent G: Evidence Validator]
        Gate -- Yes --> AgentH[Agent H: RAG Knowledge - OSHA/ASTM/ISO]
        Gate -- Yes --> AgentI[Agent I: Root-Cause Causal Reasoner]
        Gate -- Yes --> AgentJ[Agent J: CAPA Recommendation Agent]
    end

    AgentB & AgentC & AgentD & AgentE & AgentF --> Synth[Epistemological Evidence Synthesizer]
    AgentG & AgentH & AgentI & AgentJ --> Synth

    subgraph Outputs [Workspace & Deliverables]
        Synth --> Inspector[Technical Workspace & Finding Inspector]
        Synth --> PDFGen[ReportLab Formal PDF Generator]
        Synth --> DB[(SQLite / PostgreSQL Persistence)]
    end
```

---

## 4. Key Experimental Benchmark Results

Evaluated across an independently curated held-out test set of 24 real-world test cases across 6 distinct capture sessions (zero data leakage):

| Metric | Fixed Pipeline (Baseline A) | Rule-Based Pipeline (Baseline B) | Adaptive VisionIQ (Proposed) | Improvement / Delta |
|---|---|---|---|---|
| **Routing F1-Score** | 0.4444 | 0.5366 | **0.5806** | **+30.65%** relative gain vs. Fixed |
| **Routing Precision** | 0.2857 | **0.7857** | 0.6207 | Balances precision and recall |
| **Routing Recall** | **1.0000** (Naive all-on) | 0.4074 | 0.5455 | +33.9% recall vs. Rule-based |
| **Median Latency** | $2792.95\text{ ms}$ | **$1782.94\text{ ms}$** | $2455.90\text{ ms}$ | **12.07% reduction** vs. Fixed |
| **p95 Tail Latency** | $6501.74\text{ ms}$ | $3108.82\text{ ms}$ | **$3206.41\text{ ms}$** | **50.68% reduction** vs. Fixed |
| **Mean Invocations** | 8.00 agents | **2.21 agents** | 3.17 agents | **60.38% reduction** in model calls |
| **Failure Recovery** | 100% | 100% | **100%** | Zero crashes under agent failure |

### Key Takeaways:
1. **Adaptive Efficiency:** VisionIQ eliminates **60.38%** of unneeded agent executions compared to the fixed pipeline.
2. **Tail Latency Halved:** The 95th-percentile tail latency drops from **$6.50\text{ s}$** down to **$3.21\text{ s}$** (**50.68% reduction**).
3. **Resilience:** Partial results remain fully intact when individual non-critical models fail or encounter corrupt inputs.

---

## 5. Summary of Completed Phases

- **Phase 1: Foundation & Infrastructure** — Monorepo layout, FastAPI backend, SQLAlchemy ORM (all 7 entities), React 18 graphite UI, health diagnostics.
- **Phase 2: LangGraph Orchestration & Dynamic Routing** — LangGraph `StateGraph`, Equation (1) cost-aware router, uncertainty estimation, graceful fallback.
- **Phase 3: Specialist Vision Models & Registry** — Fine-tuned YOLO11n PPE detector (`yolo_ppe.pt`), NEU DefectNet CNN (`neu_defect_classifier.pt`), MVTec-AD benchmark, model registry.
- **Phase 4: Multi-Person Spatial Reasoning & Defect Inspector** — Clamped normalized bounding boxes, multi-person anatomical spatial zoning (`W-01`, `W-02`), dual-layer defect classification, Evidence Inspector.
- **Phase 5: Video Processing & Live Camera Stream** — Incremental OpenCV frame sampling, IoU multi-object tracking, non-duplicate unique entity counts, temporal persistence reasoning, WebRTC camera feed.
- **Phase 6: Retrieval-Augmented Generation (RAG)** — Multi-format document loader (PDF, MD, TXT, HTML), hybrid vector store (TF-IDF + BM25), prompt injection defense, 5-layer epistemological separation, 2-way citation validator.
- **Phase 7: Professional Technical Workspace & Assistant** — 10-item navigation, responsive 4-panel analysis layout, full agent lifecycle trace with Equation (1) breakdown, multi-criteria Finding Inspector filters, Research Experiments dashboard, deterministic VisionIQ Assistant.
- **Phase 8: Production PDF Reports & History** — ReportLab 2-page research-grade PDF generator (`NumberedCanvas`, proportional scaling), secure download and regeneration endpoints, path traversal defense, upgraded interactive History Page.
- **Phase 9: Experimental Benchmark & IEEE Access Paper** — Reproducible empirical benchmark pipeline (`eval_testset.json`, `benchmark_runner.py`), generated publication figures (`fig1`–`fig4`), complete IEEE Access manuscript draft, and scientific limitations report.
- **Phase 10: End-to-End Verification & Release** — 12-workflow automated audit test suite (`test_end_to_end_audit.py`), full regression verification (**74/74 passing tests**), clean frontend production asset build, comprehensive release checklist.

---

## 6. Repository Layout

```
visioniq/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routers (health, analysis, history, reports, rag)
│   │   ├── core/            # Config, structured JSON logging, tracing
│   │   ├── database/        # SQLAlchemy ORM models (7 entities) & scoped sessions
│   │   ├── schemas/         # Pydantic v2 validation contracts
│   │   ├── agents/          # 12 Specialist vision & cognitive agents (A through L)
│   │   ├── orchestration/   # LangGraph StateGraph & dynamic routing engine
│   │   ├── models/          # Vision model runners (YOLO11, DefectNet, EasyOCR)
│   │   ├── rag/             # Hybrid vector store, document chunker, citation validator
│   │   ├── reporting/       # ReportLab PDF synthesis engine with NumberedCanvas
│   │   ├── services/        # Video sampling, IoU tracking, assistant engine
│   │   └── main.py          # FastAPI application entrypoint with CORS & static mount
│   └── tests/               # 10 Pytest test suites (74 tests total)
├── frontend/
│   ├── src/
│   │   ├── components/      # Technical UI components (Workspace, Traces, RAG, Navbar)
│   │   ├── pages/           # 10 Application pages (Overview, Image, Video, Camera, RAG, etc.)
│   │   ├── services/        # Axios API client contracts & telemetry
│   │   ├── types/           # TypeScript contracts matching backend schemas
│   │   ├── App.tsx          # Router and application root
│   │   └── index.css        # Graphite technical design system
│   ├── package.json
│   └── vite.config.ts
├── ml/
│   ├── datasets/            # Construction PPE, NEU Defect, and MVTec manifests
│   ├── evaluation/          # Model-level evaluation routines
│   ├── model_registry/      # Serialized weights (yolo11n.pt, yolo_ppe.pt, neu_defect_classifier.pt)
│   ├── training/            # Fine-tuning and transfer learning recipes
│   └── README.md            # Model cards & dataset specifications
├── knowledge_base/          # Authenticated engineering standards (OSHA 1926, ASTM E165, ISO 9001)
├── experiments/             # Benchmark test set (eval_testset.json), runner, publication plots
├── docs/                    # Architecture diagrams, API manual, IEEE Access manuscript
├── storage/                 # Persisted SQLite database, media uploads, and PDF reports
├── SETUP.md                 # Complete environment setup guide
└── README.md
```

---

## 7. Quick Start & Execution

### 1. Prerequisites
- Python 3.12 (64-bit)
- Node.js v18+ and npm

### 2. Backend Launch
```bash
cd visioniq/backend
# Activate virtual environment
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Documentation: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/v1/health`

### 3. Frontend Launch
```bash
cd visioniq/frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

### 4. Running the Complete Verification Test Suite
```bash
# Run all 74 unit, integration, and end-to-end tests
pytest visioniq/backend/tests -v

# Run the 12-workflow end-to-end audit
pytest visioniq/backend/tests/test_end_to_end_audit.py -v
```

---

## 8. Authors & Institutional Credits

- **Gayathri R** (Roll No: 727623BCS065) — B.E. Computer Science and Engineering, Dr. MCET
- **Monica T S** (Roll No: 727623BCS083) — B.E. Computer Science and Engineering, Dr. MCET
- **Dhamodharan R** (Roll No: 727624BCS308) — B.E. Computer Science and Engineering, Dr. MCET
- **Supervisor:** **Ms. Gayathri S**, Assistant Professor, Department of Computer Science and Engineering, Dr. MCET
- **Head of Department:** **Dr. D. Sivaganesan**, Professor & Head, Department of Computer Science and Engineering, Dr. MCET
