import time
from typing import Dict, Any
from backend.app.orchestration.router import route_query
from backend.app.agents.base import BaseSpecialistAgent, AgentInput, AgentOutput, ExecutionStatus

class AdaptiveOrchestratorAgent(BaseSpecialistAgent):
    """
    Agent L: Adaptive Orchestrator Agent
    Controls the execution lifecycle, manages conditional branches,
    tracks per-agent execution times, and ensures partial failure recovery.
    """
    def __init__(self):
        super().__init__(agent_name="orchestrator", model_version="LangGraph-AdaptiveOrchestrator-v1.0")

    def _execute_internal(self, agent_input: AgentInput) -> AgentOutput:
        start_time = time.time()
        query = agent_input.user_query
        cfg = agent_input.context_data
        routing_result = route_query(
            query=query,
            alpha=cfg.get("alpha"),
            beta=cfg.get("beta"),
            gamma=cfg.get("gamma"),
            delta=cfg.get("delta"),
            threshold=cfg.get("threshold")
        )
        duration_ms = (time.time() - start_time) * 1000.0
        return AgentOutput(
            agent_name=self.agent_name,
            status=ExecutionStatus.SUCCESS,
            measured_runtime_ms=round(duration_ms, 2),
            data=routing_result,
            provenance={"router": "Eq1_CostAware", "timestamp": time.time()}
        )

    def plan(self, query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        agent_input = AgentInput(
            session_id="adhoc",
            user_query=query,
            context_data=config or {}
        )
        output = self.execute(agent_input)
        res = output.data
        res["latency_ms"] = output.measured_runtime_ms
        return res

def run_orchestrator_planning(state: dict) -> dict:
    query = state.get("user_query", "")
    agent = AdaptiveOrchestratorAgent()
    plan = agent.plan(query)

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
