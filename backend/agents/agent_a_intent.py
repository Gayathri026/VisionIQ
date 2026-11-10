import time
from typing import Dict, Any, List
from backend.config import settings
from backend.orchestrator.router import classify_intent_deterministic

class QueryIntentAgent:
    """
    Agent A: Query Intent Agent
    Interprets natural language queries, classifies intent into capability categories,
    computes uncertainty, and extracts requested entities or focus areas.
    """
    def __init__(self):
        self.agent_name = "intent"

    def analyze(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Determine intent using deterministic classifier or LLM if configured
        intent, confidence, relevance = classify_intent_deterministic(query)
        uncertainty = round(1.0 - confidence, 4)
        
        # Capability mapping
        required_capabilities = []
        if intent in ["object_counting_detection"]:
            required_capabilities.extend(["object_detection", "spatial_localization"])
        elif intent in ["ppe_safety_compliance"]:
            required_capabilities.extend(["person_detection", "ppe_detection", "spatial_association"])
        elif intent in ["defect_inspection"]:
            required_capabilities.extend(["surface_defect_detection", "anomaly_scoring"])
        elif intent in ["ocr_text_extraction"]:
            required_capabilities.extend(["text_detection", "text_recognition"])
        elif intent in ["root_cause_analysis"]:
            required_capabilities.extend(["visual_evidence_synthesis", "causal_reasoning", "retrieval_grounding"])
        elif intent in ["scene_understanding"]:
            required_capabilities.extend(["scene_context_analysis", "density_estimation"])
        else:
            required_capabilities.extend(["general_object_detection", "scene_context"])

        duration_ms = (time.time() - start_time) * 1000.0

        return {
            "intent": intent,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "required_capabilities": required_capabilities,
            "relevance_vector": relevance,
            "latency_ms": round(duration_ms, 2)
        }

def run_intent_agent(state: dict) -> dict:
    query = state.get("user_query", "")
    agent = QueryIntentAgent()
    result = agent.analyze(query)
    return {
        "detected_intent": result["intent"],
        "intent_confidence": result["confidence"],
        "intent_result": result
    }
