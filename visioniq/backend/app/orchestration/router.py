"""
VisionIQ Dynamic Cost-Aware Query Router
Implements Equation 1:
S(a, q) = alpha * R(a, q) + delta * E(a) - beta * C(a) - gamma * L(a)

Enforces configurable weights, normalized metrics, operating threshold tau,
mandatory structural dependencies, and minimum sufficient evidence constraints.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
from backend.app.core.config import settings

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
    Deterministic intent classifier and relevance estimator R(a, q).
    Classifies queries across detection, counting, OCR, PPE, defect analysis,
    anomaly localization, scene understanding, video tracking, and combined tasks.

    Returns:
        (primary_intent, intent_confidence, relevance_dict)
    """
    q = query.lower().strip()
    relevance: Dict[str, float] = {a: 0.10 for a in AVAILABLE_AGENTS}

    # Combined Query: Scene + Damage / Defect
    if ("scene" in q or "describe" in q or "overview" in q) and any(d in q for d in ["damage", "defect", "flaw", "crack", "scratch"]):
        intent = "combined_scene_and_defect"
        confidence = 0.95
        relevance["scene"] = 0.98
        relevance["defect"] = 0.90
        relevance["object"] = 0.80
        relevance["rca"] = 0.85
        relevance["evidence_validation"] = 0.85
        relevance["rag"] = 0.70
        return intent, confidence, relevance

    # Intent 1: OCR / Machine Serial / Text Extraction
    if any(k in q for k in ["serial", "read", "text", "machine id", "ocr", "number on", "label", "code on", "tag", "sign"]):
        intent = "ocr_text_extraction"
        confidence = 0.96
        relevance["ocr"] = 0.98
        relevance["object"] = 0.50
        relevance["evidence_validation"] = 0.80
        return intent, confidence, relevance

    # Intent 2: Counting & Quantifying Entities
    if any(k in q for k in ["count", "how many", "number of people", "number of workers", "total count", "enumerate"]):
        intent = "object_counting_detection"
        confidence = 0.95
        relevance["object"] = 0.98
        relevance["scene"] = 0.50
        relevance["evidence_validation"] = 0.75
        # Do not activate OCR or Defect for counting!
        relevance["ocr"] = 0.10
        relevance["defect"] = 0.10
        return intent, confidence, relevance

    # Intent 3: PPE Safety Compliance
    if any(k in q for k in ["ppe", "helmet", "hard hat", "vest", "safety gear", "compliant", "violation", "worker safety", "protective"]):
        intent = "ppe_safety_compliance"
        confidence = 0.98
        relevance["ppe"] = 0.98
        relevance["object"] = 0.85
        relevance["scene"] = 0.60
        relevance["rag"] = 0.75
        relevance["recommendation"] = 0.80
        relevance["evidence_validation"] = 0.85
        return intent, confidence, relevance

    # Intent 4: Industrial Defect / Surface Anomaly Localization
    if any(k in q for k in ["crack", "scratch", "defect", "damage", "pitted", "crazing", "inclusion", "surface flaw", "rust", "anomaly"]):
        intent = "defect_inspection"
        confidence = 0.96
        relevance["defect"] = 0.98
        relevance["object"] = 0.75
        relevance["scene"] = 0.40
        relevance["rca"] = 0.85
        relevance["rag"] = 0.75
        relevance["recommendation"] = 0.80
        relevance["evidence_validation"] = 0.85
        return intent, confidence, relevance

    # Intent 5: Root Cause Analysis / Failure Diagnosis
    if any(k in q for k in ["why", "cause", "root cause", "failure", "halted", "stopped", "diagnose", "breakdown", "tripped", "why did"]):
        intent = "root_cause_analysis"
        confidence = 0.95
        relevance["rca"] = 0.98
        relevance["defect"] = 0.85
        relevance["object"] = 0.80
        relevance["scene"] = 0.70
        relevance["rag"] = 0.90
        relevance["recommendation"] = 0.90
        relevance["evidence_validation"] = 0.90
        return intent, confidence, relevance

    # Intent 6: Video Tracking & Trajectory Analysis
    if any(k in q for k in ["track", "trajectory", "video", "movement", "path", "continuous monitoring"]):
        intent = "video_tracking"
        confidence = 0.92
        relevance["object"] = 0.98
        relevance["scene"] = 0.80
        relevance["ppe"] = 0.70
        relevance["evidence_validation"] = 0.80
        return intent, confidence, relevance

    # Intent 7: Scene Understanding & Environmental Context
    if any(k in q for k in ["scene", "environment", "surroundings", "overview", "what is happening", "context", "describe"]):
        intent = "scene_understanding"
        confidence = 0.90
        relevance["scene"] = 0.98
        relevance["object"] = 0.75
        relevance["evidence_validation"] = 0.70
        return intent, confidence, relevance

    # Intent 8: Regulatory Standards Inquiry
    if any(k in q for k in ["standard", "regulation", "osha", "iso", "astm", "manual", "guideline", "code"]):
        intent = "contextual_compliance_research"
        confidence = 0.90
        relevance["rag"] = 0.98
        relevance["ppe"] = 0.70
        relevance["defect"] = 0.70
        relevance["recommendation"] = 0.75
        return intent, confidence, relevance

    # Deterministic Fallback Rule: Ambiguous or general inquiry
    intent = "general_scene_inquiry"
    confidence = 0.60
    relevance["object"] = 0.75
    relevance["scene"] = 0.70
    relevance["evidence_validation"] = 0.60
    return intent, confidence, relevance

def compute_routing_scores(
    relevance: Dict[str, float],
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    delta: Optional[float] = None,
    threshold: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Computes Equation 1 scores for all candidate agents:
    S(a, q) = alpha * R(a, q) + delta * E(a) - beta * C(a) - gamma * L(a)
    """
    a_w = alpha if alpha is not None else settings.ROUTING_ALPHA
    b_w = beta if beta is not None else settings.ROUTING_BETA
    g_w = gamma if gamma is not None else settings.ROUTING_GAMMA
    d_w = delta if delta is not None else settings.ROUTING_DELTA
    tau = threshold if threshold is not None else settings.ROUTING_THRESHOLD

    traces: Dict[str, Dict[str, Any]] = {}
    for agent in AVAILABLE_AGENTS:
        r = relevance.get(agent, 0.10)
        e = settings.AGENT_CAPABILITIES.get(agent, 0.80)
        c = settings.AGENT_COSTS.get(agent, 0.20)
        l = settings.AGENT_LATENCIES.get(agent, 0.20)

        # Exact Formula
        score = (a_w * r) + (d_w * e) - (b_w * c) - (g_w * l)

        if score >= tau:
            decision = "SELECTED"
            reason = f"Routing score {score:.3f} >= threshold {tau:.2f} (high relevance {r:.2f} justifies compute cost {c:.2f})"
        else:
            decision = "SKIPPED"
            reason = f"Routing score {score:.3f} < threshold {tau:.2f} (insufficient query relevance {r:.2f} to warrant execution)"

        traces[agent] = {
            "agent_name": agent,
            "relevance_score": round(r, 4),
            "evidence_capability": round(e, 4),
            "normalized_cost": round(c, 4),
            "expected_latency": round(l, 4),
            "final_routing_score": round(score, 4),
            "threshold": tau,
            "decision": decision,
            "selection_reason": reason,
            "execution_status": "PENDING" if decision == "SELECTED" else "SKIPPED",
            "actual_latency_ms": 0.0,
            "error_message": None
        }

    return traces

def resolve_dependencies(
    traces: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Enforces mandatory structural dependencies and minimum sufficient evidence constraints:
    1. PPE agent strictly requires Object Detection for person-level spatial bounding.
    2. Defect agent benefits from Object Detection for component-bounded localization.
    3. RCA agent strictly requires at least one perceptual source.
    4. Recommendation agent requires RCA, Defect, or PPE.
    5. Evidence Validation runs whenever any perception agent is active.
    6. Minimum sufficient evidence constraint: defaults to Object and Scene if none selected.
    """
    selected = set([a for a, data in traces.items() if data["decision"] == "SELECTED"])

    # Rule 1: PPE requires Object detection
    if "ppe" in selected and "object" not in selected:
        selected.add("object")
        traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
        traces["object"]["selection_reason"] = "Mandatory dependency for PPE person-level spatial association"
        traces["object"]["execution_status"] = "PENDING"

    # Rule 2: Defect inspection requires Object detection
    if "defect" in selected and "object" not in selected:
        selected.add("object")
        traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
        traces["object"]["selection_reason"] = "Mandatory dependency for object-bounded defect inspection"
        traces["object"]["execution_status"] = "PENDING"

    # Rule 3: RCA requires at least one perceptual source
    if "rca" in selected:
        perceptual = selected & {"object", "defect", "ocr", "ppe"}
        if not perceptual:
            selected.add("object")
            traces["object"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["object"]["selection_reason"] = "Mandatory dependency: RCA requires perceptual anchor"
            traces["object"]["execution_status"] = "PENDING"

    # Rule 4: Recommendation requires RCA or defect or ppe
    if "recommendation" in selected:
        if not (selected & {"rca", "defect", "ppe"}):
            selected.add("rca")
            traces["rca"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["rca"]["selection_reason"] = "Mandatory dependency for CAPA recommendation formulation"
            traces["rca"]["execution_status"] = "PENDING"

    # Rule 5: Minimum sufficient evidence constraint
    perception_agents = {"object", "ppe", "ocr", "defect", "scene"}
    if not (selected & perception_agents):
        selected.add("object")
        traces["object"]["decision"] = "FALLBACK"
        traces["object"]["selection_reason"] = "Minimum sufficient evidence constraint: default perceptual anchor"
        traces["object"]["execution_status"] = "PENDING"
        selected.add("scene")
        traces["scene"]["decision"] = "FALLBACK"
        traces["scene"]["selection_reason"] = "Minimum sufficient evidence constraint: scene context anchor"
        traces["scene"]["execution_status"] = "PENDING"

    # Rule 6: Evidence validation runs whenever perception agents are active
    if selected & perception_agents:
        if "evidence_validation" not in selected:
            selected.add("evidence_validation")
            traces["evidence_validation"]["decision"] = "MANDATORY_DEPENDENCY"
            traces["evidence_validation"]["selection_reason"] = "Enforcing evidence provenance and hallucination mitigation"
            traces["evidence_validation"]["execution_status"] = "PENDING"

    ordered_selected = [a for a in AVAILABLE_AGENTS if a in selected]
    return ordered_selected, traces

def route_query(
    query: str,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    delta: Optional[float] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Main entry point for Cost-Aware Dynamic Query Routing.
    Returns structured plan, considered agents, decisions, and detailed traces.
    """
    intent, confidence, relevance = classify_intent_deterministic(query)
    traces = compute_routing_scores(relevance, alpha, beta, gamma, delta, threshold)
    selected_agents, updated_traces = resolve_dependencies(traces)
    skipped_agents = [a for a in AVAILABLE_AGENTS if a not in selected_agents]

    return {
        "intent": intent,
        "intent_confidence": round(confidence, 4),
        "selected_agents": selected_agents,
        "skipped_agents": skipped_agents,
        "considered_agents": AVAILABLE_AGENTS,
        "threshold": threshold if threshold is not None else settings.ROUTING_THRESHOLD,
        "traces": updated_traces
    }
