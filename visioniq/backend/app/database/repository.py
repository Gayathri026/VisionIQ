from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_
from backend.app.database.models import (
    AnalysisSession, Finding, AgentExecutionTrace, RAGSource, ReportRecord, ExperimentRun
)

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
                track_id=f.get("track_id"),
                model_name=f.get("model_name"),
                model_version=f.get("model_version"),
                score_interpretation=f.get("score_interpretation"),
                person_id=f.get("person_id"),
                is_confirmed_defect=f.get("is_confirmed_defect")
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

    def add_rag_sources(self, session_id: str, sources: List[Dict[str, Any]]):
        for s in sources:
            src = RAGSource(
                session_id=session_id,
                document_title=s.get("document_title", "Technical Documentation"),
                section_reference=s.get("section_reference", "General"),
                excerpt=s.get("excerpt", ""),
                similarity_score=s.get("similarity_score", 0.0),
                source_uri=s.get("source_uri")
            )
            self.db.add(src)
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

    def search_sessions(
        self,
        query: Optional[str] = None,
        input_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        order_desc: bool = True
    ) -> List[AnalysisSession]:
        q = self.db.query(AnalysisSession)
        if query:
            search_pattern = f"%{query.strip()}%"
            q = q.filter(
                or_(
                    AnalysisSession.user_query.ilike(search_pattern),
                    AnalysisSession.detected_intent.ilike(search_pattern),
                    AnalysisSession.id.ilike(search_pattern)
                )
            )
        if input_type and input_type.lower() != "all":
            q = q.filter(AnalysisSession.input_type == input_type.lower())
        if status and status.lower() != "all":
            q = q.filter(AnalysisSession.status == status.lower())

        if order_desc:
            q = q.order_by(AnalysisSession.created_at.desc())
        else:
            q = q.order_by(AnalysisSession.created_at.asc())

        return q.limit(limit).all()

    def create_report_record(
        self,
        session_id: str,
        pdf_path: str,
        file_size_bytes: int = 0,
        report_format: str = "PDF/A-1b"
    ) -> ReportRecord:
        record = ReportRecord(
            session_id=session_id,
            pdf_path=pdf_path,
            file_size_bytes=file_size_bytes,
            format=report_format
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_report_records(self, session_id: str) -> List[ReportRecord]:
        return self.db.query(ReportRecord).filter(ReportRecord.session_id == session_id).order_by(ReportRecord.created_at.desc()).all()

    def create_experiment_run(self, data: Dict[str, Any]) -> ExperimentRun:
        run = ExperimentRun(**data)
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def list_experiment_runs(self, limit: int = 20) -> List[ExperimentRun]:
        return self.db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).limit(limit).all()
