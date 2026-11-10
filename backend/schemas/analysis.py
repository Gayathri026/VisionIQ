from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class FindingSchema(BaseModel):
    finding_uid: str = Field(..., description="Unique finding ID, e.g. F-OBJ-01")
    agent_source: str = Field(..., description="Agent that produced the finding")
    label: str = Field(..., description="Class or defect name")
    bbox: Optional[List[float]] = Field(None, description="[x1, y1, x2, y2] bounding box coordinates")
    
    # Explicitly separated confidence metrics
    detection_confidence: Optional[float] = Field(None, description="Perception model confidence in [0, 1]")
    anomaly_score: Optional[float] = Field(None, description="Statistical anomaly score in [0, 1]")
    routing_score: Optional[float] = Field(None, description="Routing eligibility score")
    
    observed_evidence: str = Field(..., description="Direct visual evidence observed in the image/frame")
    root_cause_hypothesis: Optional[str] = Field(None, description="Ranked causal hypothesis if applicable")
    alternative_explanations: Optional[str] = Field(None, description="Alternative explanations or caveats")
    recommendation: Optional[str] = Field(None, description="Actionable recommendation or corrective action")
    verification_procedure: Optional[str] = Field(None, description="Inspection verification step")
    human_review_required: bool = Field(False, description="Whether qualified human inspection is required")
    frame_timestamp: Optional[float] = Field(None, description="Timestamp in video (seconds)")
    track_id: Optional[int] = Field(None, description="Persistent tracking identifier")

class AgentExecutionTraceSchema(BaseModel):
    agent_name: str
    relevance_score: float = Field(..., description="R(a, q) - Query relevance")
    evidence_capability: float = Field(..., description="E(a) - Expected evidence capability")
    normalized_cost: float = Field(..., description="C(a) - Normalized computational cost")
    expected_latency: float = Field(..., description="L(a) - Normalized expected latency")
    final_routing_score: float = Field(..., description="S(a, q) - Final score")
    decision: str = Field(..., description="SELECTED, SKIPPED, FALLBACK, or MANDATORY_DEPENDENCY")
    selection_reason: Optional[str] = None
    execution_status: str = Field(..., description="SUCCESS, FAILED, or SKIPPED")
    actual_latency_ms: float = Field(..., description="Measured execution duration in milliseconds")
    error_message: Optional[str] = None

class RAGReferenceSchema(BaseModel):
    document_title: str
    section_reference: str
    excerpt: str
    similarity_score: float
    source_uri: Optional[str] = None

class AnalysisResponse(BaseModel):
    session_id: str
    created_at: str
    user_query: str
    input_type: str
    input_path: str
    annotated_path: Optional[str] = None
    detected_intent: Optional[str] = None
    intent_confidence: float = 1.0
    selected_agents: List[str]
    executed_agents: List[str]
    skipped_agents: List[str]
    failed_agents: List[str]
    status: str
    total_latency_ms: float
    synthesis_text: Optional[str] = None
    evidence_support_score: float = 1.0
    pdf_report_path: Optional[str] = None
    findings: List[FindingSchema]
    traces: List[AgentExecutionTraceSchema]
    rag_references: List[RAGReferenceSchema]

class VideoProcessingRequest(BaseModel):
    query: str
    sample_fps: float = 1.0
    max_frames: int = 30
    enable_tracking: bool = True

class CameraFrameRequest(BaseModel):
    query: str
    frame_base64: str
    timestamp: float

class EvaluationRunSchema(BaseModel):
    run_id: str
    created_at: str
    pipeline_type: str
    dataset_name: str
    total_queries: int
    detection_map50: float
    ocr_wer: float
    ocr_cer: float
    routing_precision: float
    routing_recall: float
    routing_f1: float
    median_latency_ms: float
    p95_latency_ms: float
    latency_reduction_pct: float
    mean_agents_invoked: float
    failure_recovery_rate: float
    raw_metrics_json: Dict[str, Any] = {}
