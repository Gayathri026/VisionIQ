import time
from typing import Dict, Any
from backend.orchestrator.router import route_query

class AdaptiveOrchestratorAgent:
    """
    Agent L: Adaptive Orchestrator Agent
    Controls the execution lifecycle, manages conditional branches,
    tracks per-agent execution times, and ensures partial failure recovery.
    """
    def __init__(self):
        self.agent_name = "orchestrator"
        self.version = "LangGraph-AdaptiveOrchestrator-v1.0"

    def plan(self, query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        routing_result = route_query(
            query=query,
            alpha=config.get("alpha") if config else None,
            beta=config.get("beta") if config else None,
            gamma=config.get("gamma") if config else None,
            delta=config.get("delta") if config else None,
            threshold=config.get("threshold") if config else None
        )
        duration_ms = (time.time() - start_time) * 1000.0
        routing_result["latency_ms"] = round(duration_ms, 2)
        return routing_result

def run_orchestrator_planning(state: dict) -> dict:
    query = state.get("user_query", "")
    agent = AdaptiveOrchestratorAgent()
    plan = agent.plan(query)
    
    # Initialize execution trace log
    log = []
    for agent_name, trace_data in plan["traces"].items():
        log.append({
            "agent_name": agent_name,
            "relevance_score": trace_data["relevance_score"],
            "evidence_capability": trace_data["evidence_capability"],
            "normalized_cost": trace_data["normalized_cost"],
            "expected_latency": trace_data["expected_latency"],
            "final_routing_score": trace_data["final_routing_score"],
            "decision": trace_data["decision"],
            "selection_reason": trace_data["selection_reason"],
            "execution_status": "PENDING" if trace_data["decision"] in ["SELECTED", "MANDATORY_DEPENDENCY", "FALLBACK"] else "SKIPPED",
            "actual_latency_ms": 0.0,
            "error_message": None
        })

    return {
        "detected_intent": plan["intent"],
        "intent_confidence": plan["intent_confidence"],
        "selected_agents": plan["selected_agents"],
        "skipped_agents": plan["skipped_agents"],
        "routing_scores": plan["traces"],
        "execution_log": log
    }
