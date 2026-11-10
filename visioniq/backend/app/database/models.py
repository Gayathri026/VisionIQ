import datetime
import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class AnalysisSession(Base):
    """Stores high-level analysis requests, input references, and execution summaries."""
    __tablename__ = "analysis_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_query = Column(Text, nullable=False)
    input_type = Column(String, default="image")  # "image", "video", "camera"
    input_path = Column(String, nullable=False)
    annotated_path = Column(String, nullable=True)
    detected_intent = Column(String, nullable=True)
    intent_confidence = Column(Float, default=1.0)
    
    # Dynamic Agent Orchestration Sets
    selected_agents = Column(JSON, default=list)
    executed_agents = Column(JSON, default=list)
    skipped_agents = Column(JSON, default=list)
    failed_agents = Column(JSON, default=list)
    
    status = Column(String, default="completed")  # "completed", "failed", "partial"
    total_latency_ms = Column(Float, default=0.0)
    synthesis_text = Column(Text, nullable=True)
    evidence_support_score = Column(Float, default=1.0)
    pdf_report_path = Column(String, nullable=True)
    
    # Relationships
    findings = relationship("Finding", back_populates="session", cascade="all, delete-orphan")
    traces = relationship("AgentExecutionTrace", back_populates="session", cascade="all, delete-orphan")
    rag_sources = relationship("RAGSource", back_populates="session", cascade="all, delete-orphan")
    reports = relationship("ReportRecord", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_query": self.user_query,
            "input_type": self.input_type,
            "input_path": self.input_path,
            "annotated_path": self.annotated_path,
            "detected_intent": self.detected_intent,
            "intent_confidence": self.intent_confidence,
            "selected_agents": self.selected_agents or [],
            "executed_agents": self.executed_agents or [],
            "skipped_agents": self.skipped_agents or [],
            "failed_agents": self.failed_agents or [],
            "status": self.status,
            "total_latency_ms": self.total_latency_ms,
            "synthesis_text": self.synthesis_text,
            "evidence_support_score": self.evidence_support_score,
            "pdf_report_path": self.pdf_report_path,
            "findings": [f.to_dict() for f in self.findings] if self.findings else [],
            "traces": [t.to_dict() for t in self.traces] if self.traces else [],
            "rag_sources": [r.to_dict() for r in self.rag_sources] if self.rag_sources else []
        }

class Finding(Base):
    """Fine-grained specialist findings (detections, PPE violations, defects, OCR strings)."""
    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False)
    finding_uid = Column(String, nullable=False)  # e.g., "F-OBJ-01", "F-PPE-02"
    agent_source = Column(String, nullable=False)  # "object", "ppe", "defect", "ocr", "scene"
    label = Column(String, nullable=False)
    bbox = Column(JSON, nullable=True)  # [x1, y1, x2, y2]
    
    # Strictly separated metrics
    detection_confidence = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    routing_score = Column(Float, nullable=True)
    
    observed_evidence = Column(Text, nullable=False)
    root_cause_hypothesis = Column(Text, nullable=True)
    alternative_explanations = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
    verification_procedure = Column(Text, nullable=True)
    human_review_required = Column(Boolean, default=False)
    frame_timestamp = Column(Float, nullable=True)
    track_id = Column(Integer, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    score_interpretation = Column(Text, nullable=True)
    person_id = Column(String, nullable=True)
    is_confirmed_defect = Column(Boolean, nullable=True)

    session = relationship("AnalysisSession", back_populates="findings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "finding_uid": self.finding_uid,
            "agent_source": self.agent_source,
            "label": self.label,
            "bbox": self.bbox,
            "detection_confidence": self.detection_confidence,
            "anomaly_score": self.anomaly_score,
            "routing_score": self.routing_score,
            "observed_evidence": self.observed_evidence,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "alternative_explanations": self.alternative_explanations,
            "recommendation": self.recommendation,
            "verification_procedure": self.verification_procedure,
            "human_review_required": self.human_review_required,
            "frame_timestamp": self.frame_timestamp,
            "track_id": self.track_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "score_interpretation": self.score_interpretation,
            "person_id": self.person_id,
            "is_confirmed_defect": self.is_confirmed_defect
        }

class AgentExecutionTrace(Base):
    """Trace logs for every candidate agent evaluated under Equation 1."""
    __tablename__ = "agent_execution_traces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False)
    agent_name = Column(String, nullable=False)
    
    # Eq 1: S(a,q) = alpha*R + delta*E - beta*C - gamma*L
    relevance_score = Column(Float, default=0.0)      # R(a, q)
    evidence_capability = Column(Float, default=0.0)  # E(a)
    normalized_cost = Column(Float, default=0.0)      # C(a)
    expected_latency = Column(Float, default=0.0)     # L(a)
    final_routing_score = Column(Float, default=0.0)  # S(a, q)
    
    decision = Column(String, nullable=False)  # "SELECTED", "SKIPPED", "FALLBACK", "MANDATORY_DEPENDENCY"
    selection_reason = Column(Text, nullable=True)
    execution_status = Column(String, default="SUCCESS")  # "SUCCESS", "FAILED", "SKIPPED"
    actual_latency_ms = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    session = relationship("AnalysisSession", back_populates="traces")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "relevance_score": self.relevance_score,
            "evidence_capability": self.evidence_capability,
            "normalized_cost": self.normalized_cost,
            "expected_latency": self.expected_latency,
            "final_routing_score": self.final_routing_score,
            "decision": self.decision,
            "selection_reason": self.selection_reason,
            "execution_status": self.execution_status,
            "actual_latency_ms": self.actual_latency_ms,
            "error_message": self.error_message
        }

class ModelVersion(Base):
    """Registry tracking machine learning model checkpoints, versions, and metrics."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)        # "YOLOv11n", "VisionIQ-PPE", "NEU-Defect"
    version = Column(String, nullable=False)           # "v1.0.0"
    task_type = Column(String, nullable=False)         # "object_detection", "ppe_safety", "defect_classification"
    weights_path = Column(String, nullable=False)
    checksum_sha256 = Column(String, nullable=True)
    val_map50 = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    device_target = Column(String, default="cpu")      # "cpu", "cuda:0"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "version": self.version,
            "task_type": self.task_type,
            "weights_path": self.weights_path,
            "val_map50": self.val_map50,
            "val_accuracy": self.val_accuracy,
            "device_target": self.device_target,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class RAGSource(Base):
    """Authenticated engineering standards, OSHA codes, and technical manuals."""
    __tablename__ = "rag_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=True)
    document_title = Column(String, nullable=False)
    section_reference = Column(String, nullable=False)
    excerpt = Column(Text, nullable=False)
    similarity_score = Column(Float, default=0.0)
    source_uri = Column(String, nullable=True)

    session = relationship("AnalysisSession", back_populates="rag_sources")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "document_title": self.document_title,
            "section_reference": self.section_reference,
            "excerpt": self.excerpt,
            "similarity_score": self.similarity_score,
            "source_uri": self.source_uri
        }

class ReportRecord(Base):
    """Downloadable PDF engineering inspection reports metadata."""
    __tablename__ = "report_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False)
    report_uid = Column(String, unique=True, default=lambda: f"REP-{uuid.uuid4().hex[:8].upper()}")
    pdf_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer, default=0)
    format = Column(String, default="PDF/A-1b")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("AnalysisSession", back_populates="reports")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "report_uid": self.report_uid,
            "session_id": self.session_id,
            "pdf_path": self.pdf_path,
            "file_size_bytes": self.file_size_bytes,
            "format": self.format,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ExperimentRun(Base):
    """Scientific evaluation benchmark records across Fixed, Rule-based, and Adaptive VisionIQ."""
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, unique=True, default=lambda: f"EXP-{uuid.uuid4().hex[:8]}")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    pipeline_type = Column(String, nullable=False)     # "fixed", "rule_based", "adaptive"
    dataset_name = Column(String, default="VisionIQ-Benchmark")
    total_queries = Column(Integer, default=0)
    
    # Perception metrics
    detection_map50 = Column(Float, default=0.0)
    ocr_wer = Column(Float, default=0.0)
    ocr_cer = Column(Float, default=0.0)
    
    # Routing quality
    routing_precision = Column(Float, default=0.0)
    routing_recall = Column(Float, default=0.0)
    routing_f1 = Column(Float, default=0.0)
    
    # Latency & efficiency
    median_latency_ms = Column(Float, default=0.0)
    p95_latency_ms = Column(Float, default=0.0)
    latency_reduction_pct = Column(Float, default=0.0)
    mean_agents_invoked = Column(Float, default=0.0)
    
    failure_recovery_rate = Column(Float, default=1.0)
    raw_metrics_json = Column(JSON, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "pipeline_type": self.pipeline_type,
            "dataset_name": self.dataset_name,
            "total_queries": self.total_queries,
            "detection_map50": self.detection_map50,
            "ocr_wer": self.ocr_wer,
            "ocr_cer": self.ocr_cer,
            "routing_precision": self.routing_precision,
            "routing_recall": self.routing_recall,
            "routing_f1": self.routing_f1,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "latency_reduction_pct": self.latency_reduction_pct,
            "mean_agents_invoked": self.mean_agents_invoked,
            "failure_recovery_rate": self.failure_recovery_rate,
            "raw_metrics_json": self.raw_metrics_json or {}
        }
