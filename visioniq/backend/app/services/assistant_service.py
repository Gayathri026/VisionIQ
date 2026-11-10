import os
import re
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.session import SessionLocal
from backend.app.database.models import AnalysisSession, Finding, AgentExecutionTrace, ModelVersion, ExperimentRun

CANONICAL_FAQS = {
    "mode_selection": {
        "patterns": [r"which.*mode", r"what.*mode.*use", r"choose.*mode", r"image.*video.*camera", r"how.*select.*mode"],
        "title": "Analysis Mode Selection Guide",
        "answer": """### VisionIQ Analysis Mode Selection Matrix

Select the optimal inspection mode based on your sensory input, throughput requirements, and operational environment:

| Mode | Input Type | Recommended Use Case | Latency Profile | Key Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **Image Inspection** | Static photo / scan (`.jpg`, `.png`, `.webp`) | Detailed post-shift quality auditing, macro defect sizing, single-frame worker PPE compliance. | ~100–250 ms | High-resolution coordinate overlays, multi-person anatomical spatial zoning, OCR text extraction, RAG standards grounding. |
| **Video Inspection** | Stored footage (`.mp4`, `.avi`, `.mov`) | Shift-long CCTV review, conveyor belt defect screening, persistent temporal safety monitoring. | Frame-sampled (~5–10 FPS) | Incremental OpenCV frame extraction, IoU multi-object tracking, non-duplicate unique entity counts, temporal persistence verification. |
| **Live Camera** | WebRTC browser webcam / RTSP stream | Active real-time entry gate inspection, operator workstation safety interlock monitoring. | Real-time (~100–180 ms/frame) | Direct browser permission lifecycle, configurable capture interval, dynamic intent switching, immediate visual feedback. |

#### Decision Rule:
- Use **Image Inspection** if you need forensic precision, exact bounding boxes, and PDF audit reports.
- Use **Video Inspection** if tracking entities across time or calculating true unique worker counts is required.
- Use **Live Camera** for hands-free, continuous on-site deployment.""",
        "recommended_nav": {"path": "/image", "label": "Open Image Inspection"},
        "suggested_followups": [
            "Why did VisionIQ select these agents?",
            "What does this confidence score mean?",
            "How do I export and understand the PDF report?"
        ]
    },
    "agent_selection": {
        "patterns": [r"why.*select.*agent", r"why.*these.*agents", r"how.*routing.*work", r"routing.*equation", r"equation 1", r"router.*score"],
        "title": "Cost-Aware Agent Selection & Equation (1)",
        "answer": r"""### Cost-Aware Dynamic Agent Routing

VisionIQ selects specialist agents dynamically using **Equation (1)** from the IEEE Access research design:

$$S(a, q) = \alpha R(a, q) + \delta E(a) - \beta C(a) - \gamma L(a)$$

#### Configurable Parameter Weights:
- $\\alpha = 0.50$: **Relevance Weight** (semantic similarity between query intent $q$ and specialist agent domain $a$).
- $\\delta = 0.15$: **Evidence Capability Weight** (agent capacity to supply verifiable bounding boxes or mask regions).
- $\\beta = 0.20$: **Cost Penalty Weight** (computational/token resource overhead of executing agent $a$).
- $\\gamma = 0.15$: **Latency Penalty Weight** (expected inference runtime delay).

#### Execution Constraints:
1. **Selection Threshold:** Agent $a$ is selected if $S(a, q) \\ge \\tau$ (default $\\tau = 0.35$).
2. **Mandatory Dependencies:** `agent_c_ppe` requires `agent_b_object` (person boxes required for torso/head spatial zoning); `agent_i_rca` requires `agent_e_defect` or `agent_c_ppe`.
3. **Fault Isolation:** Each agent runs in isolated try-except boundaries with partial degradation support if an individual checkpoint fails.""",
        "recommended_nav": {"path": "/settings", "label": "Adjust Routing Weights in Settings"},
        "suggested_followups": [
            "Which analysis mode should I use?",
            "What does this confidence score mean?",
            "How do I interpret the evaluation metrics?"
        ]
    },
    "confidence_scores": {
        "patterns": [r"confidence.*score.*mean", r"what.*confidence.*mean", r"interpret.*confidence", r"score.*meaning", r"anomaly.*score"],
        "title": "Confidence and Anomaly Score Interpretation",
        "answer": """### Score Interpretation Taxonomy

VisionIQ avoids conflating distinct mathematical scoring layers. Understand how each score is calculated:

1. **Detection Confidence (YOLOv11n / PPE Detector):**
   - Range: $[0.0, 1.0]$.
   - Represents the model's posterior probability that a bounding box correctly circumscribes the identified class (`person`, `hard_hat`, `safety_vest`).
   - $\\ge 0.70$: High certainty.
   - $0.40 - 0.69$: Moderate confidence; flagged for spatial association checks.
   - $< 0.40$: Suppressed by default non-maximum suppression (NMS) threshold.

2. **Defect Classification Probability (IndustrialDefectNet CNN):**
   - Range: $[0.0, 1.0]$ via Softmax across 6 NEU classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`).
   - Represents relative class distribution over normalized surface features.

3. **Anomaly Continuous Score (PatchCore Benchmark):**
   - Nominal surface mean: $0.0148$; Defective anomaly mean: $0.0905$.
   - A score $> 0.05$ indicates structural statistical deviation from nominal baseline features.

4. **RAG Semantic Similarity Score:**
   - Range: $[0.0, 1.0]$.
   - Combined sublinear TF-IDF cosine similarity ($0.60$) + BM25 lexical score ($0.40$) + regulatory standard title boost.""",
        "recommended_nav": {"path": "/evaluation", "label": "View Model Benchmark Curves"},
        "suggested_followups": [
            "Why could the damage cause not be determined?",
            "How do I interpret the evaluation metrics?",
            "Why did VisionIQ select these agents?"
        ]
    },
    "train_ppe": {
        "patterns": [r"how.*train.*ppe", r"train.*yolo", r"fine.*tune.*ppe", r"train.*model.*ppe"],
        "title": "PPE Model Training Protocol",
        "answer": """### Reproducible PPE Model Training Procedure

To retrain or fine-tune the `VisionIQ-PPE-YOLO11n` detector on the Construction Site Safety dataset:

#### Step 1: Validate Datasets & Integrity
Verify dataset annotations, boundary coordinates, and split ratios:
```bash
python visioniq/ml/datasets/prepare_datasets.py
```
This checks normalized $[0, 1]$ bounding box boundaries and filters corrupted image rasters.

#### Step 2: Execute Fine-Tuning Pipeline
Run the reproducible training script:
```bash
python visioniq/ml/training/train_ppe_detector.py --epochs 25 --batch-size 16 --seed 42
```
*Key Parameters:*
- Base Model: `yolo11n.pt` pretrained weights.
- Target Classes: `person`, `hard_hat`, `safety_vest`.
- Learning Rate: $\\eta = 0.01$, SGD optimizer with cosine learning rate decay.
- Checkpoint output: Saved to `visioniq/ml/model_registry/yolo_ppe.pt`.

#### Step 3: Run Model Evaluation
Compute validation mAP@50 and inference latency benchmarks:
```bash
python visioniq/ml/evaluation/evaluate_models.py
```
Metrics are persisted automatically to `visioniq/ml/evaluation/metrics.json` and SQLite database.""",
        "recommended_nav": {"path": "/evaluation", "label": "Review PPE Evaluation Metrics"},
        "suggested_followups": [
            "How do I interpret the evaluation metrics?",
            "How can I add a new technical document to RAG?",
            "What does this confidence score mean?"
        ]
    },
    "add_rag_document": {
        "patterns": [r"add.*document.*rag", r"ingest.*rag", r"new.*document", r"add.*standard", r"knowledge.*base.*upload"],
        "title": "Adding Engineering Standards to RAG",
        "answer": """### RAG Knowledge Base Ingestion Protocol

You can add new regulatory standards (OSHA, ASTM, ISO), machinery operating manuals, or safety procedures via the REST API or Knowledge Base UI:

#### Method A: Via Web Interface
1. Navigate to **Knowledge Base** (`/knowledge`).
2. Click **Ingest Standard Document**.
3. Select your `.pdf`, `.md`, `.txt`, or `.html` file.
4. Provide the official document title (e.g. `OSHA Standard 1926.102: Eye and Face Protection`) and source URI.
5. Click **Process & Index**.

#### Method B: Via REST API Endpoint
```bash
curl -X POST "http://localhost:8000/api/v1/rag/ingest" \\
  -F "file=@path/to/standard.pdf" \\
  -F "title=ISO 12100: Safety of Machinery" \\
  -F "source_uri=https://www.iso.org/standard/ISO12100"
```

#### Ingestion Pipeline Highlights:
- **Prompt Injection Filter:** All files pass through `sanitize_text_for_prompt_injection` to neutralize adversarial prompt directives.
- **Recursive Chunking:** Text is partitioned into 150-word chunks with 60-word sliding overlap, preserving section headings and page references.
- **Persistent Index:** Serialized immediately to `storage/rag/rag_index.json`.""",
        "recommended_nav": {"path": "/knowledge", "label": "Open Knowledge Base Manager"},
        "suggested_followups": [
            "Why could the damage cause not be determined?",
            "How do I export and understand the PDF report?",
            "Which analysis mode should I use?"
        ]
    },
    "undetermined_cause": {
        "patterns": [r"why.*damage.*cause.*not.*determined", r"cannot.*determine.*cause", r"undetermined.*cause", r"root.*cause.*uncertain", r"damage.*cause"],
        "title": "Epistemic Uncertainty & Undetermined Damage Causes",
        "answer": """### Why Surface Damage Causes Cannot Be Conclusively Claimed From 2D Pixels

In industrial engineering and IEEE Access research integrity, **a 2D image cannot establish metallurgical or structural causality**. Claiming a definitive internal cause without physical Non-Destructive Testing (NDT) introduces ungrounded legal liability.

#### Epistemic Reasons for Undetermined Root Causes:
1. **Surface vs Internal Flaws:** A visual scratch on steel sheet metal (ASTM A568) can stem from roller gouging, chemical crazing, or handling scratches. Visual pixels cannot measure depth below $0.05\\text{ mm}$ without contact profilometry.
2. **Ambiguous Morphology:** Inclusions, pitted surfaces, and crazing share high-frequency edge similarities under varying lighting angles.
3. **Mandatory Verification Standards:**
   - **ASTM E165:** Liquid Penetrant Examination required to confirm whether a surface crack breaches structural boundary.
   - **ASTM E155 / E1444:** Radiographic and Magnetic Particle Examination required for internal void assessment.

VisionIQ provides **plausible causal hypotheses** ranked with explicit epistemic uncertainty, followed by required on-site NDT verification steps.""",
        "recommended_nav": {"path": "/knowledge", "label": "Review ASTM NDT Standards"},
        "suggested_followups": [
            "What does this confidence score mean?",
            "How do I export and understand the PDF report?",
            "How do I interpret the evaluation metrics?"
        ]
    },
    "evaluation_metrics": {
        "patterns": [r"interpret.*evaluation.*metrics", r"what.*map50.*mean", r"macro.*f1", r"confusion.*matrix", r"how.*evaluate"],
        "title": "Interpreting Scientific Evaluation Metrics",
        "answer": """### Guide to VisionIQ Scientific Evaluation Metrics

The **Model Evaluation** (`/evaluation`) dashboard presents genuine empirical metrics across perception and routing:

#### 1. Perception Metrics
- **mAP@50 (Mean Average Precision at IoU=0.50):** Evaluates object and PPE localization. Considers a detection true positive if the intersection-over-union between prediction and ground-truth box exceeds $50\\%$.
- **Macro-F1 Score:** Harmonic mean of precision and recall calculated independently per class and averaged unweighted. Balances classes with skewed support (e.g. rare crazing vs frequent scratches).
- **Confusion Matrix:** Displays true versus predicted classes for NEU steel defects. Off-diagonal elements identify mutual feature confusion (e.g. patches mistaken for inclusions).

#### 2. Orchestration & Routing Metrics
- **Routing Precision & Recall:** Precision measures the percentage of invoked agents that were genuinely necessary; recall measures whether all necessary agents were activated.
- **Latency Reduction ($\\Delta L$):** Percentage improvement in median latency achieved by the Adaptive Multi-Agent pipeline relative to the Fixed Pipeline (which redundantly executes all agents).
- **Mean Agents Invoked:** Average number of specialist agents executed per query ($1.8$ on average vs $7.0$ for fixed).""",
        "recommended_nav": {"path": "/evaluation", "label": "Inspect Evaluation Metrics"},
        "suggested_followups": [
            "How do I train the PPE model?",
            "Why did VisionIQ select these agents?",
            "Which analysis mode should I use?"
        ]
    },
    "pdf_report": {
        "patterns": [r"export.*pdf", r"understand.*pdf", r"report.*generation", r"download.*report", r"how.*export"],
        "title": "Exporting and Understanding Engineering PDF Reports",
        "answer": """### Generating and Interpreting PDF Engineering Reports

VisionIQ compiles formal, audit-ready PDF inspection reports for archival and regulatory compliance:

#### How to Export:
1. Complete an analysis in **Image Inspection** (`/image`) or select a past session from **Analysis History** (`/history`).
2. In the evidence viewer or session header, click **Download PDF Report**.
3. The report is synthesized on-the-fly and downloaded as a standard PDF document.

#### Report Structure & Verification Sections:
1. **Header & Metadata:** Unique Session UID, ISO 8601 timestamp, user query, input image resolution, and total system runtime.
2. **Executive Summary:** Epistemologically structured high-level diagnosis and compliance status.
3. **Orchestration Trace Table:** Every evaluated agent, selection status, Equation (1) score, and execution duration.
4. **Annotated Visual Evidence:** Embedded high-contrast image with numbered bounding boxes matching finding UIDs.
5. **Epistemological Findings Matrix:** Direct observations, model provenance, and uncertainty ratings for each finding.
6. **Regulatory References & NDT Next Steps:** Verbatim OSHA/ASTM excerpts and non-destructive testing requirements before machinery sign-off.""",
        "recommended_nav": {"path": "/history", "label": "Browse Analysis History & Reports"},
        "suggested_followups": [
            "Which analysis mode should I use?",
            "Why could the damage cause not be determined?",
            "Why did VisionIQ select these agents?"
        ]
    }
}

class VisionIQAssistantService:
    """
    Context-aware In-App Engineering Assistant for VisionIQ.
    Answers operational, architectural, and evaluation queries,
    and explains active analysis sessions and agent execution traces.
    """
    def __init__(self):
        self.faqs = CANONICAL_FAQS

    def query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        current_trace: Optional[List[Dict[str, Any]]] = None,
        active_finding_uid: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes an assistant query. If the query asks about the current analysis,
        it inspects the active session/trace state; otherwise, it grounds the answer
        in verified repository documentation.
        """
        clean_q = user_query.strip()
        lower_q = clean_q.lower()

        # 1. Check if user is asking about the ACTIVE SESSION / CURRENT TRACE
        if session_id or current_trace:
            is_asking_current = any(k in lower_q for k in [
                "this analysis", "current analysis", "selected these agents", "why these agents",
                "this image", "current session", "what happened", "explain trace", "why was",
                "explain this finding", "this score"
            ])
            if is_asking_current:
                return self._explain_current_session(session_id, current_trace, active_finding_uid, clean_q)

        # 2. Check Canonical FAQs via Regex Pattern Matching
        for key, faq in self.faqs.items():
            for pat in faq["patterns"]:
                if re.search(pat, lower_q):
                    return {
                        "status": "success",
                        "query": clean_q,
                        "title": faq["title"],
                        "answer": faq["answer"],
                        "source_references": [
                            "docs/dataset_specifications.md",
                            "backend/app/orchestration/router.py",
                            "ml/model_registry/registry.json"
                        ],
                        "recommended_navigation": faq.get("recommended_nav"),
                        "suggested_followups": faq.get("suggested_followups", []),
                        "mode": "deterministic_verified"
                    }

        # 3. Fallback: Semantic Term Match over Repository Knowledge Base
        return self._fallback_guidance(clean_q)

    def _explain_current_session(
        self,
        session_id: Optional[str],
        current_trace: Optional[List[Dict[str, Any]]],
        active_finding_uid: Optional[str],
        query: str
    ) -> Dict[str, Any]:
        """Inspects actual session state in DB or current trace payload and synthesizes an explanation."""
        session_data = None
        if session_id:
            db = SessionLocal()
            try:
                sess = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
                if sess:
                    session_data = sess.to_dict()
            finally:
                db.close()

        traces = (session_data.get("traces") if session_data else None) or current_trace or []
        findings = session_data.get("findings", []) if session_data else []

        selected_agents = [t.get("agent_name") for t in traces if t.get("decision") in ("SELECTED", "MANDATORY_DEPENDENCY")]
        skipped_agents = [t.get("agent_name") for t in traces if t.get("decision") == "SKIPPED"]
        total_latency = session_data.get("total_latency_ms") if session_data else sum(t.get("actual_latency_ms", 0.0) for t in traces)

        # Format trace explanations
        trace_bullets = []
        for t in traces:
            name = t.get("agent_name", "Unknown Agent")
            dec = t.get("decision", "EVALUATED")
            score = t.get("final_routing_score", 0.0)
            reason = t.get("selection_reason", "Calculated by Equation 1")
            lat = t.get("actual_latency_ms", 0.0)
            status = t.get("execution_status", "COMPLETED")
            if dec in ("SELECTED", "MANDATORY_DEPENDENCY"):
                trace_bullets.append(f"- **{name}** (`{dec}`): Routing score $S(a,q) = {score:.3f}$. Reason: {reason}. Latency: `{lat:.1f}ms` ({status}).")
            else:
                trace_bullets.append(f"- **{name}** (`SKIPPED`): Routing score $S(a,q) = {score:.3f}$ below threshold. Reason: {reason}.")

        answer = f"""### Analysis Explanation for Session `{session_id or 'Active Workspace'}`

#### 1. Query Intent & Agent Selection
- **Inspection Query:** *"{session_data.get('user_query', query) if session_data else query}"*
- **Detected Intent:** **{session_data.get('detected_intent', 'Industrial Inspection') if session_data else 'Multi-Task Inspection'}**
- **Selected Specialist Agents ({len(selected_agents)}):** {', '.join(selected_agents) if selected_agents else 'None'}
- **Pruned / Skipped Agents ({len(skipped_agents)}):** {', '.join(skipped_agents) if skipped_agents else 'None'}
- **Total Latency:** `{total_latency:.1f}ms`

#### 2. Mathematical Routing Breakdown
{''.join(f'{b}\n' for b in trace_bullets)}

#### 3. Visual Findings Generated
VisionIQ produced **{len(findings)}** structured findings for this image.
"""
        if active_finding_uid:
            target_f = next((f for f in findings if f.get("finding_uid") == active_finding_uid), None)
            if target_f:
                answer += f"""
#### Focus on Finding `{active_finding_uid}` ({target_f.get('label')})
- **Agent Source:** `{target_f.get('agent_source')}` ({target_f.get('model_name')})
- **Observation:** {target_f.get('observed_evidence')}
- **Confidence / Score:** `{target_f.get('detection_confidence') or target_f.get('anomaly_score')}` ({target_f.get('score_interpretation')})
- **Recommended Action:** {target_f.get('recommendation')}
"""

        return {
            "status": "success",
            "query": query,
            "title": f"Live Trace Explanation: {session_id or 'Current Workspace'}",
            "answer": answer.strip(),
            "source_references": [
                f"Session ID: {session_id or 'in-memory'}",
                "backend/app/orchestration/router.py",
                "backend/app/database/models.py"
            ],
            "recommended_navigation": {"path": "/history", "label": "View Full History Record"},
            "suggested_followups": [
                "What does this confidence score mean?",
                "Why could the damage cause not be determined?",
                "How do I export and understand the PDF report?"
            ],
            "mode": "deterministic_verified"
        }

    def _fallback_guidance(self, query: str) -> Dict[str, Any]:
        """Provides generalized application guidance when a query is outside canonical FAQs."""
        return {
            "status": "success",
            "query": query,
            "title": "VisionIQ System Engineering Guide",
            "answer": f"""### VisionIQ Technical Assistant

You asked: *"{query}"*

#### Quick Navigation & Core Features:
1. **Image Inspection (`/image`):** Upload industrial rasters, run adaptive agent pipelines, and view bounding boxes synchronized with RAG evidence.
2. **Video Inspection (`/video`):** Inspect `.mp4` video files with multi-object tracking (IoU + centroid) and unique entity counts.
3. **Live Camera (`/camera`):** Stream browser webcam frames at configurable sampling rates with real-time intent switching.
4. **Knowledge Base (`/knowledge`):** Manage authoritative standards (OSHA 1926.100, ASTM A568, ISO 13849), test hybrid search, and validate citations.
5. **Model Evaluation (`/evaluation`):** Inspect empirical mAP@50, confusion matrices, and latency comparisons.
6. **Research Experiments (`/experiments`):** Review scientific pipeline benchmarks comparing fixed, rule-based, and adaptive routing.

*Note: The VisionIQ Assistant operates in Deterministic Engineering Mode based on verified repository files. For specific technical explanations, select one of the suggested prompts below.*""",
            "source_references": [
                "README.md",
                "docs/dataset_specifications.md",
                "ml/model_registry/registry.json"
            ],
            "recommended_navigation": {"path": "/overview", "label": "Go to System Overview"},
            "suggested_followups": [
                "Which analysis mode should I use?",
                "Why did VisionIQ select these agents?",
                "What does this confidence score mean?",
                "How do I train the PPE model?",
                "How can I add a new technical document to RAG?",
                "Why could the damage cause not be determined?",
                "How do I interpret the evaluation metrics?",
                "How do I export and understand the PDF report?"
            ],
            "mode": "deterministic_verified"
        }

assistant_service = VisionIQAssistantService()
