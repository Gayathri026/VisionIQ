from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import AnalysisSession, Finding, AgentExecutionTrace, RAGReference, EvaluationRun

class AnalysisRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(self, session_data: Dict[str, Any]) -> AnalysisSession:
        session = AnalysisSession(
            id=session_data.get("id"),
            user_query=session_data["user_query"],
            input_type=session_data.get("input_type", "image"),
            input_path=session_data["input_path"],
            annotated_path=session_data.get("annotated_path"),
            detected_intent=session_data.get("detected_intent"),
            intent_confidence=session_data.get("intent_confidence", 1.0),
            selected_agents=session_data.get("selected_agents", []),
            executed_agents=session_data.get("executed_agents", []),
            skipped_agents=session_data.get("skipped_agents", []),
            failed_agents=session_data.get("failed_agents", []),
            status=session_data.get("status", "completed"),
            total_latency_ms=session_data.get("total_latency_ms", 0.0),
            synthesis_text=session_data.get("synthesis_text"),
            evidence_support_score=session_data.get("evidence_support_score", 1.0),
            pdf_report_path=session_data.get("pdf_report_path")
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def add_findings(self, session_id: str, findings: List[Dict[str, Any]]):
        for f in findings:
            finding = Finding(
                session_id=session_id,
                finding_uid=f.get("finding_uid", "F-00"),
                agent_source=f.get("agent_source", "unknown"),
                label=f.get("label", "Unknown"),
                bbox=f.get("bbox"),
                detection_confidence=f.get("detection_confidence"),
                anomaly_score=f.get("anomaly_score"),
                routing_score=f.get("routing_score"),
                observed_evidence=f.get("observed_evidence", ""),
                root_cause_hypothesis=f.get("root_cause_hypothesis"),
                alternative_explanations=f.get("alternative_explanations"),
                recommendation=f.get("recommendation"),
                verification_procedure=f.get("verification_procedure"),
                human_review_required=f.get("human_review_required", False),
                frame_timestamp=f.get("frame_timestamp"),
                track_id=f.get("track_id")
            )
            self.db.add(finding)
        self.db.commit()

    def add_traces(self, session_id: str, traces: List[Dict[str, Any]]):
        for t in traces:
            trace = AgentExecutionTrace(
                session_id=session_id,
                agent_name=t.get("agent_name", "unknown"),
                relevance_score=t.get("relevance_score", 0.0),
                evidence_capability=t.get("evidence_capability", 0.0),
                normalized_cost=t.get("normalized_cost", 0.0),
                expected_latency=t.get("expected_latency", 0.0),
                final_routing_score=t.get("final_routing_score", 0.0),
                decision=t.get("decision", "SELECTED"),
                selection_reason=t.get("selection_reason", ""),
                execution_status=t.get("execution_status", "SUCCESS"),
                actual_latency_ms=t.get("actual_latency_ms", 0.0),
                error_message=t.get("error_message")
            )
            self.db.add(trace)
        self.db.commit()

    def add_rag_references(self, session_id: str, references: List[Dict[str, Any]]):
        for r in references:
            ref = RAGReference(
                session_id=session_id,
                document_title=r.get("document_title", "Technical Documentation"),
                section_reference=r.get("section_reference", "General"),
                excerpt=r.get("excerpt", ""),
                similarity_score=r.get("similarity_score", 0.0),
                source_uri=r.get("source_uri")
            )
            self.db.add(ref)
        self.db.commit()

    def update_pdf_path(self, session_id: str, pdf_path: str):
        session = self.db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        if session:
            session.pdf_report_path = pdf_path
            self.db.commit()

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self.db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()

    def list_sessions(self, limit: int = 50) -> List[AnalysisSession]:
        return self.db.query(AnalysisSession).order_by(AnalysisSession.created_at.desc()).limit(limit).all()

    def create_evaluation_run(self, data: Dict[str, Any]) -> EvaluationRun:
        run = EvaluationRun(**data)
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def list_evaluation_runs(self, limit: int = 20) -> List[EvaluationRun]:
        return self.db.query(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit).all()
