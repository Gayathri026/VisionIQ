import time
import os
import uuid
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from backend.orchestrator.state import AgentState
from backend.agents.agent_l_orchestrator import run_orchestrator_planning
from backend.agents.agent_b_object import run_object_agent
from backend.agents.agent_c_ppe import run_ppe_agent
from backend.agents.agent_d_ocr import run_ocr_agent
from backend.agents.agent_e_defect import run_defect_agent
from backend.agents.agent_f_scene import run_scene_agent
from backend.agents.agent_g_evidence_validation import run_evidence_validation_agent
from backend.agents.agent_h_rag import run_rag_agent
from backend.agents.agent_i_rca import run_rca_agent
from backend.agents.agent_j_recommendation import run_recommendation_agent
from backend.agents.agent_k_report import run_report_agent
from backend.services.annotation_service import annotation_service
from backend.database.session import SessionLocal
from backend.database.repository import AnalysisRepository

def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Evaluates query intent, scores agents using Eq. (1), and plans branches."""
    start_time = time.time()
    plan_updates = run_orchestrator_planning(state)
    
    # Session ID assignment
    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())

    plan_updates["start_time"] = start_time
    plan_updates["findings"] = []
    plan_updates["executed_agents"] = []
    plan_updates["failed_agents"] = []
    plan_updates["errors"] = []
    return plan_updates

def vision_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Executes selected perceptual vision specialists with fault isolation."""
    selected = state.get("selected_agents", [])
    findings = list(state.get("findings", []))
    executed = list(state.get("executed_agents", []))
    failed = list(state.get("failed_agents", []))
    log = list(state.get("execution_log", []))
    errors = list(state.get("errors", []))
    updates: Dict[str, Any] = {}

    def update_trace(agent_name: str, status: str, duration_ms: float, err: str = None):
        for entry in log:
            if entry["agent_name"] == agent_name:
                entry["execution_status"] = status
                entry["actual_latency_ms"] = round(duration_ms, 2)
                entry["error_message"] = err
                break

    # 1. Object Agent
    if "object" in selected:
        t0 = time.time()
        try:
            res = run_object_agent(state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("object")
            update_trace("object", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("object")
            errors.append(f"ObjectAgent error: {str(e)}")
            update_trace("object", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 2. PPE Agent
    if "ppe" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_ppe_agent(temp_state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("ppe")
            update_trace("ppe", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("ppe")
            errors.append(f"PPEAgent error: {str(e)}")
            update_trace("ppe", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 3. Defect Agent
    if "defect" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_defect_agent(temp_state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("defect")
            update_trace("defect", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("defect")
            errors.append(f"DefectAgent error: {str(e)}")
            update_trace("defect", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 4. OCR Agent
    if "ocr" in selected:
        t0 = time.time()
        try:
            res = run_ocr_agent(state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("ocr")
            update_trace("ocr", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("ocr")
            errors.append(f"OCRAgent error: {str(e)}")
            update_trace("ocr", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 5. Scene Understanding Agent
    if "scene" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_scene_agent(temp_state)
            updates.update(res)
            findings.extend(res.get("new_findings", []))
            executed.append("scene")
            update_trace("scene", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("scene")
            errors.append(f"SceneAgent error: {str(e)}")
            update_trace("scene", "FAILED", (time.time() - t0) * 1000.0, str(e))

    updates["findings"] = findings
    updates["executed_agents"] = executed
    updates["failed_agents"] = failed
    updates["execution_log"] = log
    updates["errors"] = errors
    return updates

def validation_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Evidence validation agent checks provenance and cross-references claims."""
    t0 = time.time()
    log = list(state.get("execution_log", []))
    try:
        val_updates = run_evidence_validation_agent(state)
        dur = (time.time() - t0) * 1000.0
        for entry in log:
            if entry["agent_name"] == "evidence_validation":
                entry["execution_status"] = "SUCCESS"
                entry["actual_latency_ms"] = round(dur, 2)
                break
        val_updates["execution_log"] = log
        return val_updates
    except Exception as e:
        dur = (time.time() - t0) * 1000.0
        for entry in log:
            if entry["agent_name"] == "evidence_validation":
                entry["execution_status"] = "FAILED"
                entry["actual_latency_ms"] = round(dur, 2)
                entry["error_message"] = str(e)
                break
        return {"evidence_support_score": 0.80, "execution_log": log}

def cognitive_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: RAG standards retrieval, Root Cause Analysis, and CAPA recommendations."""
    selected = state.get("selected_agents", [])
    executed = list(state.get("executed_agents", []))
    failed = list(state.get("failed_agents", []))
    log = list(state.get("execution_log", []))
    updates: Dict[str, Any] = {}

    def update_trace(agent_name: str, status: str, duration_ms: float, err: str = None):
        for entry in log:
            if entry["agent_name"] == agent_name:
                entry["execution_status"] = status
                entry["actual_latency_ms"] = round(duration_ms, 2)
                entry["error_message"] = err
                break

    # 1. RAG Knowledge Retrieval
    if "rag" in selected:
        t0 = time.time()
        try:
            res = run_rag_agent(state)
            updates.update(res)
            executed.append("rag")
            update_trace("rag", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("rag")
            update_trace("rag", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 2. Root Cause Analysis
    if "rca" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_rca_agent(temp_state)
            updates.update(res)
            executed.append("rca")
            update_trace("rca", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("rca")
            update_trace("rca", "FAILED", (time.time() - t0) * 1000.0, str(e))

    # 3. Recommendation Planning
    if "recommendation" in selected:
        t0 = time.time()
        try:
            temp_state = dict(state)
            temp_state.update(updates)
            res = run_recommendation_agent(temp_state)
            updates.update(res)
            executed.append("recommendation")
            update_trace("recommendation", "SUCCESS", (time.time() - t0) * 1000.0)
        except Exception as e:
            failed.append("recommendation")
            update_trace("recommendation", "FAILED", (time.time() - t0) * 1000.0, str(e))

    updates["executed_agents"] = executed
    updates["failed_agents"] = failed
    updates["execution_log"] = log
    return updates

def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Consolidates findings, renders annotations, generates PDF report, persists to DB."""
    start_time = state.get("start_time", time.time())
    total_latency_ms = (time.time() - start_time) * 1000.0
    session_id = state.get("session_id", str(uuid.uuid4()))
    image_path = state.get("image_path", "")
    findings = state.get("findings", [])
    traces = state.get("execution_log", [])
    rag_refs = state.get("rag_references", [])
    evidence_score = state.get("evidence_support_score", 1.0)

    # 1. Generate visual annotation overlay
    annotated_path = ""
    if image_path and os.path.exists(image_path):
        try:
            annotated_path = annotation_service.annotate_image(
                image_path=image_path,
                findings=findings,
                output_filename=f"annotated_{session_id[:8]}_{os.path.basename(image_path)}"
            )
        except Exception as e:
            annotated_path = image_path

    # 2. Synthesize clear narrative summary
    intent = state.get("detected_intent", "Analysis")
    rca = state.get("rca_result")
    ppe = state.get("ppe_result")
    defect = state.get("defect_result")
    
    summary_parts = [f"Analysis completed for query '{state.get('user_query')}'. Intent classified as {intent}."]
    if ppe:
        summary_parts.append(
            f"PPE Evaluation: {ppe.get('compliant_count', 0)} compliant, "
            f"{ppe.get('non_compliant_count', 0)} non-compliant, {ppe.get('uncertain_count', 0)} uncertain."
        )
    if defect and defect.get("defects_detected", 0) > 0:
        summary_parts.append(f"Identified {defect['defects_detected']} surface defect region(s).")
    if rca and rca.get("ranked_hypotheses"):
        top_h = rca["ranked_hypotheses"][0]
        summary_parts.append(f"Primary Root-Cause Hypothesis: {top_h['hypothesis']} (Est. Probability: {top_h['estimated_cause_probability']*100:.0f}%).")

    synthesis_text = " ".join(summary_parts)

    # 3. Generate downloadable PDF report
    pdf_report_path = ""
    try:
        report_state = dict(state)
        report_state["session_id"] = session_id
        report_state["annotated_path"] = annotated_path
        report_state["total_latency_ms"] = total_latency_ms
        pdf_res = run_report_agent(report_state)
        pdf_report_path = pdf_res.get("pdf_report_path", "")
    except Exception as e:
        pdf_report_path = ""

    # 4. Construct Final API Response
    final_response = {
        "session_id": session_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_query": state.get("user_query", ""),
        "input_type": state.get("input_type", "image"),
        "input_path": image_path,
        "annotated_path": annotated_path,
        "detected_intent": intent,
        "intent_confidence": state.get("intent_confidence", 1.0),
        "selected_agents": state.get("selected_agents", []),
        "executed_agents": state.get("executed_agents", []),
        "skipped_agents": state.get("skipped_agents", []),
        "failed_agents": state.get("failed_agents", []),
        "status": "completed" if not state.get("failed_agents") else "partial",
        "total_latency_ms": round(total_latency_ms, 2),
        "synthesis_text": synthesis_text,
        "evidence_support_score": round(evidence_score, 3),
        "pdf_report_path": pdf_report_path,
        "findings": findings,
        "traces": traces,
        "rag_references": rag_refs
    }

    # 5. Persist to SQLite Database
    try:
        db = SessionLocal()
        repo = AnalysisRepository(db)
        repo.create_session({
            "id": session_id,
            "user_query": state.get("user_query", ""),
            "input_type": state.get("input_type", "image"),
            "input_path": image_path,
            "annotated_path": annotated_path,
            "detected_intent": intent,
            "intent_confidence": state.get("intent_confidence", 1.0),
            "selected_agents": state.get("selected_agents", []),
            "executed_agents": state.get("executed_agents", []),
            "skipped_agents": state.get("skipped_agents", []),
            "failed_agents": state.get("failed_agents", []),
            "status": final_response["status"],
            "total_latency_ms": total_latency_ms,
            "synthesis_text": synthesis_text,
            "evidence_support_score": evidence_score,
            "pdf_report_path": pdf_report_path
        })
        repo.add_findings(session_id, findings)
        repo.add_traces(session_id, traces)
        repo.add_rag_references(session_id, rag_refs)
        db.close()
    except Exception as e:
        print(f"Database persistence warning: {e}")

    return {
        "final_response": final_response,
        "annotated_path": annotated_path,
        "pdf_report_path": pdf_report_path,
        "total_latency_ms": total_latency_ms,
        "synthesis_text": synthesis_text
    }

def build_visioniq_graph():
    """Compiles the LangGraph StateGraph with conditional orchestration."""
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("vision_agents", vision_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("cognitive_agents", cognitive_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "vision_agents")
    workflow.add_edge("vision_agents", "validation")
    workflow.add_edge("validation", "cognitive_agents")
    workflow.add_edge("cognitive_agents", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()

# Global compiled application
vision_iq_app = build_visioniq_graph()
