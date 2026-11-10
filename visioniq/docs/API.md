# VisionIQ: REST API Reference Manual

VisionIQ provides a fully documented RESTful API built on **FastAPI** with OpenAPI 3.1 and JSON Schema support.

- **Base URL:** `http://localhost:8000/api/v1`
- **Interactive Documentation (Swagger):** `http://localhost:8000/docs`
- **Alternative Documentation (ReDoc):** `http://localhost:8000/redoc`

---

## 1. System Health & Diagnostics

### `GET /api/v1/health`
Returns system status, active database connectivity, loaded neural models, and framework versions.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "app_name": "VisionIQ Backend",
  "version": "1.0.0",
  "database": "connected",
  "models_loaded": {
    "yolo_general": true,
    "yolo_ppe": true,
    "defect_classifier": true,
    "ocr_engine": true
  },
  "timestamp": "2026-10-26T12:00:00Z"
}
```

---

## 2. Image Inspection & Dynamic Orchestration

### `POST /api/v1/analyze`
Executes multi-agent visual analysis with LangGraph dynamic orchestration and Equation (1) cost-aware routing.

#### Request (Multipart Form-Data)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | Binary File | Yes | Raw image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`). Max size: 25MB. |
| `query` | String | Yes | Inspection objective or query prompt (e.g., "Check worker PPE and hard hats"). |
| `intent` | String | No | Optional override intent (e.g., `ppe_safety_compliance`, `industrial_surface_defect`). |
| `pipeline_type` | String | No | Routing mode: `adaptive` (default), `rule_based`, or `fixed`. |

#### Response `200 OK`
```json
{
  "session_id": "ses_9e2f1807d9bb",
  "status": "completed",
  "input_type": "image",
  "query": "Check worker PPE compliance",
  "detected_intent": "ppe_safety_compliance",
  "intent_confidence": 0.95,
  "pipeline_type": "adaptive",
  "total_latency_ms": 2341.2,
  "selected_agents": ["agent_a_intent", "agent_b_object_detector", "agent_c_ppe_inspector", "agent_f_spatial_reasoner", "agent_g_validator"],
  "annotated_image_path": "/static/uploads/ann_ses_9e2f1807d9bb.jpg",
  "findings": [
    {
      "finding_uid": "F-001",
      "label": "Worker Without Hardhat",
      "bbox": [120.0, 45.0, 240.0, 290.0],
      "detection_confidence": 0.94,
      "score_interpretation": "High confidence identification (94.0%)",
      "model_name": "VisionIQ-PPE-YOLO11n",
      "model_version": "v1.0.0",
      "spatial_zone": "cranial",
      "observed_evidence": "Individual W-01 observed without protective cranial equipment.",
      "root_cause_hypothesis": "Worker removed helmet or bypassed entry checkpoint.",
      "alternative_explanations": "Helmet may be resting outside of camera line-of-sight.",
      "recommendation": "Stop work immediately and issue ANSI Z89.1 certified head protection.",
      "verification_procedure": "Safety supervisor physical badge and PPE verification audit.",
      "uncertainty_flag": false
    }
  ],
  "agent_traces": [
    {
      "agent_id": "agent_c_ppe_inspector",
      "agent_name": "PPE Safety Compliance Specialist",
      "status": "completed",
      "routing_score": 0.582,
      "formula_inputs": {
        "alpha": 0.50,
        "delta": 0.15,
        "beta": 0.20,
        "gamma": 0.15,
        "relevance": 0.95,
        "evidence_capability": 0.92,
        "compute_cost": 0.35,
        "latency_penalty": 0.40
      },
      "execution_time_ms": 421.5,
      "model_identifier": "yolo_ppe.pt"
    }
  ],
  "rag_sources": [
    {
      "document_title": "OSHA Standard 1926.100: Head Protection",
      "section_reference": "29 CFR 1926.100(a)",
      "similarity_score": 0.88,
      "excerpt": "Employees working in areas where there is a possible danger of head injury shall be protected by protective helmets."
    }
  ],
  "synthesis": {
    "direct_visual_observations": "1 individual identified without hard hat.",
    "model_predictions": "VisionIQ-PPE-YOLO11n confidence 94.0%.",
    "retrieved_standards": "OSHA 1926.100 Head Protection.",
    "causal_hypotheses": "Cranial PPE absent during active work.",
    "actionable_recommendations": "Issue ANSI Z89.1 head protection prior to shift resumption."
  }
}
```

---

## 3. Video & Live Camera Stream Inspection

### `POST /api/v1/analyze/video`
Processes multi-frame industrial video with configurable frame sampling intervals, IoU multi-object tracking, and deduplicated unique entity counting.

#### Request (Multipart Form-Data)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | Binary File | Yes | Video file (`.mp4`, `.avi`, `.mov`, `.mkv`). Max: 100MB. |
| `query` | String | Yes | Inspection prompt. |
| `sample_interval_sec` | Float | No | Frame sampling interval (default: `1.0` second). |

#### Response `200 OK`
Returns summary of sampled frames, unique tracked entities, persistence timeline, and aggregated violations.

### `POST /api/v1/analyze/live-frame`
Processes a single frame captured from a live browser camera stream.

#### Request (Multipart Form-Data)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | Binary File | Yes | Captured JPEG image frame. |
| `query` | String | Yes | Active live query. |
| `session_id` | String | No | Active camera session identifier. |

---

## 4. Retrieval-Augmented Generation (RAG)

### `GET /api/v1/rag/documents`
Lists all indexed engineering standards, manufacturer manuals, and regulatory documents in the persistent vector store.

### `POST /api/v1/rag/ingest`
Ingests and indexes a new document with section extraction and prompt injection sanitization.

#### Request (Multipart Form-Data)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | Binary File | Yes | Supported document: `.pdf`, `.md`, `.txt`, `.html`. Max: 20MB. |
| `title` | String | No | Human-readable document title. |
| `source_uri` | String | No | Canonical URL or URI for citation linking. |

#### Response `200 OK`
```json
{
  "status": "success",
  "message": "Successfully indexed 'ASTM_E165_NDT.md'",
  "chunks_indexed": 4,
  "document_id": "doc_e8f2a1b9"
}
```

### `POST /api/v1/rag/query`
Executes hybrid semantic search over indexed engineering documents.

#### Request `application/json`
```json
{
  "query": "What is the minimum dwell time for liquid penetrant testing?",
  "top_k": 3,
  "relevance_threshold": 0.25
}
```

### `POST /api/v1/rag/validate-citations`
Validates whether cited clauses and standard numbers match authentic documents in the index to prevent hallucination.

---

## 5. PDF Reports & Analysis History

### `GET /api/v1/history`
Retrieves past analysis sessions with optional query, date, and status filtering.

### `GET /api/v1/sessions/{session_id}`
Returns complete details, findings, traces, and visual artifacts for a specific session.

### `POST /api/v1/reports/generate/{session_id}`
Generates a formal 2-page research-grade PDF inspection report using ReportLab.

### `GET /api/v1/reports/{report_id}/download`
Securely downloads the generated PDF report. Path traversal and unauthorized filesystem access are strictly blocked.

---

## 6. Model Evaluation & Research Experiments

### `GET /api/v1/experiments/runs`
Returns all historical comparative benchmark runs recorded in the database.

### `POST /api/v1/experiments/run-benchmark`
Executes empirical benchmark comparison across Fixed, Rule-Based, and Adaptive pipelines on the held-out test set.

### `GET /api/v1/experiments/plots`
Returns URLs to publication figures:
- `fig1_latency_vs_f1.png`
- `fig2_agent_invocations.png`
- `fig3_latency_distribution.png`
- `fig4_ablation_breakdown.png`

---

## 7. Model Registry

### `GET /api/v1/models`
Returns list of registered neural models, version strings, parameter counts, and deployment status.
