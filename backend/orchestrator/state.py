from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict, total=False):
    # Inputs
    image_path: str
    user_query: str
    session_id: Optional[str]
    input_type: str  # "image", "video", "camera"
    
    # Intent & Routing
    detected_intent: Optional[str]
    intent_confidence: float
    routing_scores: Dict[str, Dict[str, Any]]
    selected_agents: List[str]
    executed_agents: List[str]
    skipped_agents: List[str]
    failed_agents: List[str]
    
    # Agent Outputs
    object_result: Optional[Dict[str, Any]]
    ppe_result: Optional[Dict[str, Any]]
    ocr_result: Optional[Dict[str, Any]]
    defect_result: Optional[Dict[str, Any]]
    scene_result: Optional[Dict[str, Any]]
    evidence_validation_result: Optional[Dict[str, Any]]
    rag_result: Optional[Dict[str, Any]]
    rca_result: Optional[Dict[str, Any]]
    recommendation_result: Optional[Dict[str, Any]]
    
    # Synthesized Results
    findings: List[Dict[str, Any]]
    rag_references: List[Dict[str, Any]]
    synthesis_text: Optional[str]
    evidence_support_score: float
    annotated_path: Optional[str]
    pdf_report_path: Optional[str]
    final_response: Optional[Dict[str, Any]]
    
    # Orchestration Traces & Timings
    execution_log: List[Dict[str, Any]]
    errors: List[str]
    start_time: float
    total_latency_ms: float
