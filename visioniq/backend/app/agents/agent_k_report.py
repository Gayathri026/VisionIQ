import os
import time
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings
from backend.app.agents.base import BaseSpecialistAgent, AgentInput, AgentOutput, ExecutionStatus
from backend.app.reporting.pdf_generator import generate_inspection_pdf

class ReportGeneratorAgent(BaseSpecialistAgent):
    """
    Agent K: Engineering PDF Report Generation Module
    Uses ReportLab and NumberedCanvas to produce a formal, verifiable engineering inspection report
    containing annotated visual evidence, finding tables, routing traces,
    RAG citations, root cause hypotheses, and reproducibility metadata.
    """
    def __init__(self):
        super().__init__(agent_name="report", model_version="VisionIQ-PDF-Engine-v1.0")

    def _execute_internal(self, agent_input: AgentInput) -> AgentOutput:
        start_time = time.time()
        ctx = agent_input.context_data
        session_id = agent_input.session_id
        query = agent_input.user_query
        input_path = agent_input.image_path or ""
        annotated_path = ctx.get("annotated_path", "")
        findings = ctx.get("findings", [])
        traces = ctx.get("execution_log", [])
        rag_references = ctx.get("rag_references", [])
        rca_result = ctx.get("rca_result")
        recommendation_result = ctx.get("recommendation_result")
        evidence_support_score = ctx.get("evidence_support_score", 1.0)
        total_latency_ms = ctx.get("total_latency_ms", 0.0)

        pdf_path = self.generate_pdf(
            session_id=session_id,
            query=query,
            input_path=input_path,
            annotated_path=annotated_path,
            findings=findings,
            traces=traces,
            rag_references=rag_references,
            rca_result=rca_result,
            recommendation_result=recommendation_result,
            evidence_support_score=evidence_support_score,
            total_latency_ms=total_latency_ms
        )

        duration_ms = (time.time() - start_time) * 1000.0

        return AgentOutput(
            agent_name=self.agent_name,
            status=ExecutionStatus.SUCCESS,
            measured_runtime_ms=round(duration_ms, 2),
            findings=[],
            data={"pdf_report_path": pdf_path},
            provenance={"format": "PDF/A-1b", "engine": "ReportLab 5.1.0", "timestamp": time.time()}
        )

    def generate_pdf(
        self,
        session_id: str,
        query: str,
        input_path: str,
        annotated_path: str,
        findings: List[Dict[str, Any]],
        traces: List[Dict[str, Any]],
        rag_references: List[Dict[str, Any]],
        rca_result: Dict[str, Any] = None,
        recommendation_result: Dict[str, Any] = None,
        evidence_support_score: float = 1.0,
        total_latency_ms: float = 0.0,
        representative_frames: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        return generate_inspection_pdf(
            session_id=session_id,
            query=query,
            input_path=input_path,
            annotated_path=annotated_path,
            findings=findings,
            traces=traces,
            rag_references=rag_references,
            rca_result=rca_result,
            recommendation_result=recommendation_result,
            evidence_support_score=evidence_support_score,
            total_latency_ms=total_latency_ms,
            representative_frames=representative_frames
        )

def run_report_agent(state: dict) -> dict:
    session_id = state.get("session_id", "sess-default")
    query = state.get("user_query", "")
    input_path = state.get("image_path", "")
    annotated_path = state.get("annotated_path", "")
    findings = state.get("findings", [])
    traces = state.get("execution_log", [])
    rag_refs = state.get("rag_references", [])
    rca = state.get("rca_result")
    recom = state.get("recommendation_result")
    evidence_score = state.get("evidence_support_score", 1.0)
    total_latency = state.get("total_latency_ms", 0.0)

    agent = ReportGeneratorAgent()
    pdf_path = agent.generate_pdf(
        session_id=session_id,
        query=query,
        input_path=input_path,
        annotated_path=annotated_path,
        findings=findings,
        traces=traces,
        rag_references=rag_refs,
        rca_result=rca,
        recommendation_result=recom,
        evidence_support_score=evidence_score,
        total_latency_ms=total_latency
    )
    return {"pdf_report_path": pdf_path}
