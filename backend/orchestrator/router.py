import time
from typing import Dict, Any, List, Tuple
from backend.config import settings

# Available specialist agents considered by the router
AVAILABLE_AGENTS = [
    "object",
    "ppe",
    "ocr",
    "defect",
    "scene",
    "evidence_validation",
    "rag",
    "rca",
    "recommendation"
]

def classify_intent_deterministic(query: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Deterministic intent classifier and relevance estimator.
    Returns: (primary_intent, intent_confidence, dict of relevance scores R(a, q) in [0, 1])
    """
    q = query.lower().strip()
    
    # Initialize baseline relevance
    relevance: Dict[str, float] = {a: 0.10 for a in AVAILABLE_AGENTS}
    intent = "general_scene_inquiry"
    confidence = 0.85
    
    # Intent 1: PPE Safety Compliance
    if any(k in q for k in ["ppe", "helmet", "hard hat", "vest", "safety gear", "compliant", "violation", "worker safety"]):
        intent = "ppe_safety_compliance"
        confidence = 0.95
        relevance["ppe"] = 0.98
        relevance["object"] = 0.85
        relevance["scene"] = 0.60
        relevance["rag"] = 0.75  # OSHA standards
        relevance["recommendation"] = 0.80
        relevance["evidence_validation"] = 0.85

    # Intent 2: Industrial Defect / Anomaly Inspection
    elif any(k in q for k in ["defect", "damage", "crack", "scratch", "pitted", "crazing", "inclusion", "leak", "broken", "anomaly", "surface flaw", "rust"]):
        intent = "defect_inspection"
        confidence = 0.95
        relevance["defect"] = 0.98
        relevance["object"] = 0.75
        relevance["scene"] = 0.40
        relevance["rca"] = 0.85
        relevance["rag"] = 0.75  # ASTM defect standards
        relevance["recommendation"] = 0.80
        relevance["evidence_validation"] = 0.85

    # Intent 3: Root Cause Analysis / Failure Diagnosis
    elif any(k in q for k in ["why", "cause", "root cause", "failure", "halted", "stopped", "diagnose", "breakdown", "tripped"]):
        intent = "root_cause_analysis"
        confidence = 0.95
        relevance["rca"] = 0.98
        relevance["defect"] = 0.85
        relevance["object"] = 0.80
        relevance["scene"] = 0.70
        relevance["ocr"] = 0.65  # Check error codes or machine IDs
        relevance["rag"] = 0.90  # Maintenance manuals
        relevance["recommendation"] = 0.90
        relevance["evidence_validation"] = 0.90

    # Intent 4: OCR / Machine ID / Text Extraction
    elif any(k in q for k in ["text", "machine id", "serial", "code", "label", "read", "sign", "number", "ocr", "tag"]):
        intent = "ocr_text_extraction"
        confidence = 0.95
        relevance["ocr"] = 0.98
        relevance["object"] = 0.50
        relevance["evidence_validation"] = 0.80

    # Intent 5: Object Counting & Detection
    elif any(k in q for k in ["count", "how many", "detect objects", "locate", "what objects", "tools", "machines"]):
        intent = "object_counting_detection"
        confidence = 0.95
        relevance["object"] = 0.98
        relevance["scene"] = 0.50
        relevance["evidence_validation"] = 0.75

    # Intent 6: Scene Understanding & Environmental Context
    elif any(k in q for k in ["scene", "environment", "surroundings", "overview", "what is happening", "context", "describe"]):
        intent = "scene_understanding"
        confidence = 0.90
        relevance["scene"] = 0.98
        relevance["object"] = 0.75
        relevance["evidence_validation"] = 0.70

    # Intent 7: Regulatory / RAG Standards Inquiry
    elif any(k in q for k in ["standard", "regulation", "osha", "iso", "astm", "manual", "guideline"]):
        intent = "contextual_compliance_research"
        confidence = 0.90
        relevance["rag"] = 0.98
        relevance["ppe"] = 0.70
        relevance["defect"] = 0.70
        relevance["recommendation"] = 0.75

    # Fallback General Inquiry
    else:
        intent = "general_analysis"
        confidence = 0.60
        relevance["object"] = 0.75
        relevance["scene"] = 0.70
        relevance["evidence_validation"] = 0.60

    return intent, confidence, relevance

def compute_routing_scores(
    relevance: Dict[str, float],
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
    delta: float = None,
    threshold: float = None
) -> Dict[str, Dict[str, Any]]:
    """
    Computes routing scores for all candidate agents using the research formula:
    S(a, q) = alpha * R(a, q) + delta * E(a) - beta * C(a) - gamma * L(a)
    """
    a_w = alpha if alpha is not None else settings.ROUTING_ALPHA
    b_w = beta if beta is not None else settings.ROUTING_BETA
    g_w = gamma if gamma is not None else settings.ROUTING_GAMMA
    d_w = delta if delta is not None else settings.ROUTING_DELTA
    tau = threshold if threshold is not None else settings.ROUTING_THRESHOLD

    traces = {}
    for agent in AVAILABLE_AGENTS:
        r = relevance.get(agent, 0.10)
        e = settings.AGENT_CAPABILITIES.get(agent, 0.80)
        c = settings.AGENT_COSTS.get(agent, 0.20)
        l = settings.AGENT_LATENCIES.get(agent, 0.20)
        
        # Exact Formula
        score = (a_w * r) + (d_w * e) - (b_w * c) - (g_w * l)
        
        # Decision based on threshold tau
        decision = "SELECTED" if score >= tau else "SKIPPED"
        reason = f"Score {score:.3f} >= threshold {tau:.2f}" if score >= tau else f"Score {score:.3f} < threshold {tau:.2f}"
        
        traces[agent] = {
            "relevance_score": round(r, 4),
            "evidence_capability": round(e, 4),
            "normalized_cost": round(c, 4),
            "expected_latency": round(l, 4),
            "final_routing_score": round(score, 4),
            "threshold": tau,
            "decision": decision,
            "selection_reason": reason
        }
        
    return traces

def resolve_dependencies(
    traces: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Enforces mandatory agent dependencies and minimum sufficient evidence constraints.
    Returns: (selected_agents_list, updated_traces)
    """
    selected = set([a for a, data in traces.items() if data["decision"] == "SELECTED"])
    
    # Rule 1: RCA requires at least one perceptual source
    if "rca" in selected:
        if not (selected & {"object", "defect", "ocr", "ppe"}):
            # Add object detector as mandatory dependency
            selected.add("object")
            traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["object"]["selection_reason"] = "Mandatory dependency for Root Cause Analysis (perceptual ground)"

    # Rule 2: Recommendation requires RCA or defect or ppe
    if "recommendation" in selected:
        if not (selected & {"rca", "defect", "ppe"}):
            selected.add("rca")
            traces["rca"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["rca"]["selection_reason"] = "Mandatory dependency for Recommendation formulation"

    # Rule 3: PPE safety agent requires object detector for spatial association
    if "ppe" in selected:
        if "object" not in selected:
            selected.add("object")
            traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["object"]["selection_reason"] = "Mandatory dependency for PPE person-level spatial association"

    # Rule 4: Defect agent benefits from object localization
    if "defect" in selected:
        if "object" not in selected:
            selected.add("object")
            traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["object"]["selection_reason"] = "Mandatory dependency for object-bounded defect inspection"

    # Rule 5: Minimum sufficient evidence constraint
    # If no perception agent was selected, activate deterministic fallback
    perception_agents = {"object", "ppe", "ocr", "defect", "scene"}
    if not (selected & perception_agents):
        selected.add("object")
        traces["object"]["decision"] = "FALLBACK"
        traces["object"]["selection_reason"] = "Minimum sufficient evidence constraint: default perceptual anchor"
        selected.add("scene")
        traces["scene"]["decision"] = "FALLBACK"
        traces["scene"]["selection_reason"] = "Minimum sufficient evidence constraint: scene context anchor"

    # Rule 6: Evidence validation should run whenever perception agents are active
    if selected & perception_agents:
        if "evidence_validation" not in selected:
            selected.add("evidence_validation")
            traces["evidence_validation"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["evidence_validation"]["selection_reason"] = "Enforcing evidence provenance and hallucination checking"

    ordered_selected = [a for a in AVAILABLE_AGENTS if a in selected]
    return ordered_selected, traces

def route_query(
    query: str,
    alpha: float = None,
    beta: float = None,
    gamma: float = None,
    delta: float = None,
    threshold: float = None
) -> Dict[str, Any]:
    """
    Main entry point for Cost-Aware Dynamic Query Routing.
    1. Classifies intent and relevance.
    2. Computes research equation scores.
    3. Resolves dependencies.
    4. Produces verifiable trace.
    """
    intent, confidence, relevance = classify_intent_deterministic(query)
    traces = compute_routing_scores(relevance, alpha, beta, gamma, delta, threshold)
    selected_agents, updated_traces = resolve_dependencies(traces)
    
    skipped_agents = [a for a in AVAILABLE_AGENTS if a not in selected_agents]

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "selected_agents": selected_agents,
        "skipped_agents": skipped_agents,
        "traces": updated_traces
    }
