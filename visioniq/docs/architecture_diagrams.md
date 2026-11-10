# VisionIQ: System Architecture, Data-Flow & Sequence Specifications

**Project Title:** VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration and Cost-Aware Query Routing  
**Academic Evaluation:** BE CSE Final Year Project (Course 23CSP701), Anna University  
**Target Publication:** IEEE Access  

---

## 1. Multi-Tier Framework Architecture

VisionIQ is architected as an adaptive, multi-tier intelligence system decoupling query understanding, cost-aware dispatch, parallel specialist execution, and evidence-grounded synthesis.

```mermaid
graph TB
    subgraph Client_Layer ["Client Presentation Tier (React 18 + Vite + TypeScript)"]
        UI_Nav["Graphite Technical Navbar & System Telemetry"]
        UI_Workspace["Inspection Workspace (Image / Query / Canvas)"]
        UI_Video["Multi-Entity Video Tracking & Trajectory Canvas"]
        UI_Camera["Live Camera Continuous Ingestion"]
        UI_Trace["Multi-Agent Orchestration Trace & Latency Breakdown"]
        UI_RAG["Domain RAG Knowledge Evidence Viewer"]
        UI_Eval["Ablation & Benchmark Evaluation Dashboard (Recharts)"]
    end

    subgraph API_Gateway ["Gateway & Routing Tier (FastAPI + Pydantic v2)"]
        API_Health["GET /api/v1/health (Diagnostics & Model Availability)"]
        API_Image["POST /api/v1/analyze/image (Synchronous Multi-Agent Analysis)"]
        API_Video["POST /api/v1/analyze/video (Batch & Sampled Frame Tracking)"]
        API_Camera["POST /api/v1/analyze/camera-frame (Sub-100ms Inference)"]
        API_Reports["GET /api/v1/reports/{id}/pdf (ReportLab PDF Generation)"]
        API_History["GET /api/v1/history (Session History & Audit Trail)"]
    end

    subgraph Orchestration_Engine ["LangGraph Dynamic Orchestration Tier"]
        Intent_Analyzer["Intent Analysis & Entity Extraction Node"]
        Router_Core["Cost-Aware Routing Engine\nS(a, q) = α R(a,q) + δ E(a) - β C(a) - γ L(a)\nα=0.50, β=0.20, γ=0.15, δ=0.15, τ=0.45"]
        Dispatch_Filter["Threshold Gating (Score ≥ τ ? SELECT : SKIP)"]
        Evidence_Aggregator["Multi-Agent Evidence Fusion & Cross-Verification"]
        Synthesis_Node["Evidence-Grounded Synthesis & Hallucination Mitigation"]
    end

    subgraph Specialist_Agents ["Specialist Vision Agent Pool (11 Modules)"]
        A1["Object Detection Agent (YOLO11n)"]
        A2["Instance Segmentation Agent (YOLO11-seg)"]
        A3["Zero-Shot Classification Agent (CLIP / MobileNet)"]
        A4["Facial Analysis Agent (Haar / MediaPipe)"]
        A5["Optical Character Recognition Agent (EasyOCR)"]
        A6["Pose & Ergonomics Agent (YOLO11-pose)"]
        A7["Multi-Object Tracking Agent (ByteTRACK / BoT-SORT)"]
        A8["Defect & Anomaly Agent (Structural / Texture)"]
        A9["Spatial Heatmap Agent (GradCAM / Density)"]
        A10["Domain Vector RAG Agent (FAISS + MiniLM)"]
        A11["Cross-Domain Synthesis Agent (Grounded Fusion)"]
    end

    subgraph Storage_Layer ["Data Persistence & Model Registry"]
        DB[(SQLite / PostgreSQL via SQLAlchemy ORM)]
        Model_Reg["Model Weights Registry (ml/model_registry/)"]
        Knowledge_Base["Industrial Domain Standards (OSHA / ISO / SOPs)"]
        Storage_Media["Artifacts & PDF Reports (storage/reports/)"]
    end

    Client_Layer <--> API_Gateway
    API_Gateway --> Orchestration_Engine
    Orchestration_Engine --> Dispatch_Filter
    Dispatch_Filter --> Specialist_Agents
    Specialist_Agents --> Evidence_Aggregator
    Evidence_Aggregator --> Synthesis_Node
    Synthesis_Node --> API_Gateway
    Orchestration_Engine <--> DB
    Specialist_Agents <--> Model_Reg
    A10 <--> Knowledge_Base
    Synthesis_Node --> Storage_Media
```

---

## 2. End-to-End Data-Flow Architecture

```mermaid
flowchart TD
    UserQuery["User Input (Natural Query + Image/Video Payload)"] --> ValidateContract["Pydantic v2 Schema Validation & Sanitization"]
    ValidateContract --> CreateSession["Initialize AnalysisSession in Database"]
    
    CreateSession --> IntentExtraction["Intent Extraction (PPE, Safety, Defect, Counting, Text, Spatial)"]
    IntentExtraction --> ComputeRouting["Compute Agent Utility Scores:\nS(a, q) = 0.50·R(a,q) + 0.15·E(a) - 0.20·C(a) - 0.15·L(a)"]
    
    ComputeRouting --> GatingDecision{"Utility S(a, q) ≥ 0.45?"}
    GatingDecision -- Yes --> DispatchAgent["Parallel Execution via Async Task Pool"]
    GatingDecision -- No --> RecordSkipTrace["Log Skipped Agent Trace (Reason: Below Threshold)"]
    
    DispatchAgent --> ModelInference["Model Inference (YOLO / OCR / ResNet / Heuristics)"]
    ModelInference --> ExtractFindings["Extract Findings (BBoxes, Confidence, Evidence, Labels)"]
    ExtractFindings --> RecordExecTrace["Record Agent Execution Trace (Latency, Status, Scores)"]
    
    RecordExecTrace --> RAGGrounding["Query RAG Knowledge Base (Vector Search FAISS)"]
    RAGGrounding --> EvidenceFusion["Evidence Fusion & Hallucination Mitigation Check"]
    
    EvidenceFusion --> ComputeSupportScore["Compute Evidence Support Score (0.0 to 1.0)"]
    ComputeSupportScore --> GenerateReport["Compile ReportLab PDF Report & Render Overlay BBoxes"]
    
    GenerateReport --> SaveDB["Persist Findings, Traces & PDF Path to Database"]
    SaveDB --> APIResponse["Return Typed AnalysisResponse (JSON + Static URLs)"]
    RecordSkipTrace --> SaveDB
```

---

## 3. Dynamic Orchestration Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Engineer / User
    participant Frontend as React Frontend
    participant API as FastAPI Gateway
    participant DB as SQLite / PostgreSQL
    participant Orchestrator as LangGraph Orchestrator
    participant Agents as Vision Specialist Agents
    participant RAG as FAISS Knowledge Engine
    participant PDF as ReportLab Generator

    User->>Frontend: Upload image + Enter inspection query
    Frontend->>API: POST /api/v1/analyze/image (FormData)
    API->>DB: INSERT INTO analysis_sessions (status='processing')
    API->>Orchestrator: Invoke Graph(image_path, query, session_id)
    
    critical Intent Analysis & Routing
        Orchestrator->>Orchestrator: Extract query intent & target features
        Orchestrator->>Orchestrator: Calculate S(a, q) for all 11 agents
        Orchestrator->>DB: INSERT INTO agent_execution_traces (decisions & scores)
    end

    par Parallel Specialist Execution
        Orchestrator->>Agents: Execute Selected Agent 1 (e.g. Object Detection)
        Agents-->>Orchestrator: Bounding boxes & confidence scores
    and
        Orchestrator->>Agents: Execute Selected Agent 2 (e.g. PPE / Defect)
        Agents-->>Orchestrator: Specialist findings & anomaly scores
    and
        Orchestrator->>RAG: Retrieve Regulatory Clauses (OSHA / ISO)
        RAG-->>Orchestrator: Top-k grounding excerpts & similarity scores
    end

    critical Evidence Grounding & PDF Compilation
        Orchestrator->>Orchestrator: Synthesize findings & verify evidence support
        Orchestrator->>PDF: Generate 2-Page Technical Inspection Report
        PDF-->>Orchestrator: PDF artifact path (/storage/reports/rep_XYZ.pdf)
    end

    Orchestrator->>DB: UPDATE analysis_sessions (status='completed', synthesis, latency)
    Orchestrator->>DB: INSERT INTO findings (all detected anomalies & objects)
    Orchestrator-->>API: AnalysisResponse payload
    API-->>Frontend: JSON response (session_id, findings, traces, pdf_url)
    Frontend-->>User: Render visual overlays, agent telemetry, and download link
```

---

## 4. Mathematical Grounding & Routing Formulation

The dynamic orchestration routing score for an agent $a \in \mathcal{A}$ given query $q$ is defined as:

$$S(a, q) = \alpha R(a, q) + \delta E(a) - \beta C(a) - \gamma L(a)$$

Where:
- $R(a, q) \in [0, 1]$ represents the semantic query-to-agent relevance.
- $E(a) \in [0, 1]$ represents the agent's historical evidence capability.
- $C(a) \in [0, 1]$ represents the normalized compute cost.
- $L(a) \in [0, 1]$ represents the normalized latency.
- Calibrated weights: $\alpha = 0.50$, $\beta = 0.20$, $\gamma = 0.15$, $\delta = 0.15$, satisfying $\sum \text{weights} = 1.0$.
- Gating threshold: $\tau = 0.45$. An agent is activated if and only if $S(a, q) \ge \tau$.
