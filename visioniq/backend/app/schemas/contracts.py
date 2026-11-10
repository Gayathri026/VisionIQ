from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "online"})
    project: str = Field(..., json_schema_extra={"example": "VisionIQ: Adaptive Multi-Agent Vision Intelligence"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    database_connected: bool = True
    models: Dict[str, bool] = Field(..., description="Availability status of registered vision models")
    routing_parameters: Dict[str, float] = Field(..., description="Active routing weights and threshold")

class FindingContract(BaseModel):
    model_config = {"protected_namespaces": ()}

    finding_uid: str = Field(..., json_schema_extra={"example": "F-OBJ-01"})
    agent_source: str = Field(..., json_schema_extra={"example": "object"})
    label: str = Field(..., json_schema_extra={"example": "Worker"})
    bbox: Optional[List[float]] = Field(None, json_schema_extra={"example": [120.0, 85.0, 310.0, 480.0]})
    detection_confidence: Optional[float] = Field(None, json_schema_extra={"example": 0.92})
    anomaly_score: Optional[float] = None
    routing_score: Optional[float] = Field(None, json_schema_extra={"example": 0.552})
    observed_evidence: str = Field(..., json_schema_extra={"example": "Worker detected with high-visibility vest."})
    root_cause_hypothesis: Optional[str] = None
    alternative_explanations: Optional[str] = None
    recommendation: Optional[str] = None
    verification_procedure: Optional[str] = None
    human_review_required: bool = False
    frame_timestamp: Optional[float] = None
    track_id: Optional[int] = None
    model_name: Optional[str] = Field(None, json_schema_extra={"example": "Ultralytics YOLO11n"})
    model_version: Optional[str] = Field(None, json_schema_extra={"example": "v1.0.0"})
    score_interpretation: Optional[str] = Field(None, json_schema_extra={"example": "High confidence visual match (92.0%)"})
    person_id: Optional[str] = Field(None, json_schema_extra={"example": "Worker #1 (W-01)"})
    is_confirmed_defect: Optional[bool] = Field(None, description="True for model-classified defects; False for heuristic anomaly cues")

class AgentTraceContract(BaseModel):
    agent_name: str = Field(..., json_schema_extra={"example": "ppe"})
    relevance_score: float = Field(..., json_schema_extra={"example": 0.98}, description="R(a, q) - Query relevance")
    evidence_capability: float = Field(..., json_schema_extra={"example": 0.90}, description="E(a) - Evidence capability")
    normalized_cost: float = Field(..., json_schema_extra={"example": 0.20}, description="C(a) - Normalized compute cost")
    expected_latency: float = Field(..., json_schema_extra={"example": 0.22}, description="L(a) - Normalized latency")
    final_routing_score: float = Field(..., json_schema_extra={"example": 0.552}, description="S(a, q) - Formula 1 result")
    decision: str = Field(..., json_schema_extra={"example": "SELECTED"})
    selection_reason: Optional[str] = Field(None, json_schema_extra={"example": "Score 0.552 >= threshold 0.45"})
    execution_status: str = Field(..., json_schema_extra={"example": "SUCCESS"})
    actual_latency_ms: float = Field(..., json_schema_extra={"example": 42.5})
    error_message: Optional[str] = None

class RAGSourceContract(BaseModel):
    document_title: str = Field(..., json_schema_extra={"example": "OSHA Standard 1926.100 - Head Protection"})
    section_reference: str = Field(..., json_schema_extra={"example": "29 CFR 1926.100(a)"})
    excerpt: str = Field(..., json_schema_extra={"example": "Employees working in areas where there is a possible danger..."})
    similarity_score: float = Field(..., json_schema_extra={"example": 0.88})
    source_uri: Optional[str] = None

class AnalysisResponse(BaseModel):
    session_id: str
    created_at: str
    user_query: str
    input_type: str = "image"
    input_path: str
    input_url: Optional[str] = None
    annotated_path: Optional[str] = None
    annotated_url: Optional[str] = None
    detected_intent: Optional[str] = None
    intent_confidence: float = 1.0
    selected_agents: List[str]
    executed_agents: List[str]
    skipped_agents: List[str]
    failed_agents: List[str]
    status: str = "completed"
    total_latency_ms: float
    synthesis_text: Optional[str] = None
    evidence_support_score: float = 1.0
    pdf_report_path: Optional[str] = None
    pdf_url: Optional[str] = None
    findings: List[FindingContract]
    traces: List[AgentTraceContract]
    rag_sources: List[RAGSourceContract] = []

class VideoProcessingRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Track personnel and verify PPE compliance"})
    sample_fps: float = Field(1.0, ge=0.2, le=5.0)
    max_frames: int = Field(25, ge=1, le=120)

class VideoProcessingResponse(BaseModel):
    session_id: Optional[str] = None
    video_path: str
    duration_sec: float
    total_video_frames: int
    sampled_frames_processed: int
    sample_fps: float
    unique_tracked_entities: Dict[str, int]
    total_unique_entities: int
    frame_results: List[Dict[str, Any]]
    temporal_violations: Optional[List[Dict[str, Any]]] = None
    detected_intent: Optional[str] = None
    selected_agents: Optional[List[str]] = None
    synthesis_text: Optional[str] = None
    total_latency_ms: float

class CameraFrameRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Continuous PPE monitoring"})
    frame_base64: str = Field(..., description="Base64 encoded JPEG/PNG frame payload")
    timestamp: float = Field(0.0)

class CameraFrameResponse(BaseModel):
    status: str
    session_id: Optional[str] = None
    timestamp: float
    latency_ms: float
    detected_intent: Optional[str] = None
    selected_agents: List[str] = []
    executed_agents: List[str] = []
    findings: List[FindingContract] = []
    annotated_frame_base64: Optional[str] = None
    synthesis: Optional[str] = None

class HistorySessionContract(BaseModel):
    id: str
    created_at: str
    user_query: str
    input_type: str
    detected_intent: Optional[str]
    status: str
    total_latency_ms: float
    evidence_support_score: float
    pdf_report_path: Optional[str] = None

class HistoryResponse(BaseModel):
    status: str = "success"
    total: int
    sessions: List[HistorySessionContract]

class OrchestrationPlanRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Count all visible people."})
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0)
    beta: Optional[float] = Field(None, ge=0.0, le=1.0)
    gamma: Optional[float] = Field(None, ge=0.0, le=1.0)
    delta: Optional[float] = Field(None, ge=0.0, le=1.0)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

class OrchestrationPlanResponse(BaseModel):
    query: str
    detected_intent: str
    intent_confidence: float
    selected_agents: List[str]
    skipped_agents: List[str]
    considered_agents: List[str]
    threshold: float
    traces: Dict[str, AgentTraceContract]
    latency_ms: float

class SessionTraceResponse(BaseModel):
    session_id: str
    user_query: str
    detected_intent: Optional[str]
    selected_agents: List[str]
    executed_agents: List[str]
    skipped_agents: List[str]
    failed_agents: List[str]
    total_latency_ms: float
    traces: List[AgentTraceContract]
