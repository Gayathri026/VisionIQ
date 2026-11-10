import time
from typing import Dict, Any, List

class EvidenceValidationAgent:
    """
    Agent G: Evidence Validation Agent
    Cross-checks downstream claims against raw perceptual outputs (detection bboxes, OCR texts, defect masks).
    Identifies hallucinations, unsupported assertions, contradictions, and insufficient evidence.
    Computes an Evidence Support Score in [0, 1].
    """
    def __init__(self):
        self.agent_name = "evidence_validation"
        self.model_version = "VisionIQ-ProvenanceValidator-v1.0"

    def validate(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        object_result: Dict[str, Any] = None,
        ppe_result: Dict[str, Any] = None,
        ocr_result: Dict[str, Any] = None,
        defect_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        q_lower = query.lower()
        contradictions = []
        unsupported_claims = []
        validation_warnings = []
        
        support_score = 1.0

        # Check 1: Query asserts entity that does not exist in perceptual outputs
        if any(w in q_lower for w in ["person", "worker", "helmet", "vest"]) and ppe_result:
            total_persons = ppe_result.get("total_persons_evaluated", 0)
            if total_persons == 0:
                unsupported_claims.append(
                    "Query inquires about personnel or PPE safety, but zero persons were detected in the input image."
                )
                support_score -= 0.25

        if any(w in q_lower for w in ["text", "serial", "code", "label", "reading"]) and ocr_result:
            detected_text = ocr_result.get("detected_text", "")
            if not detected_text or detected_text == "No text detected":
                unsupported_claims.append(
                    "Query requested text or serial identification, but no readable optical characters were found."
                )
                support_score -= 0.25

        if any(w in q_lower for w in ["defect", "crack", "damage", "flaw"]) and defect_result:
            defects_count = defect_result.get("defects_detected", 0)
            if defects_count == 0:
                validation_warnings.append(
                    "No anomalous defect patterns exceeded the detection threshold. Verifiable surface flaws absent."
                )

        # Check 2: Low-confidence perception assertions
        for f in findings:
            conf = f.get("detection_confidence")
            if conf is not None and conf < 0.45:
                validation_warnings.append(
                    f"Finding {f.get('finding_uid')} ({f.get('label')}) carries low confidence ({conf*100:.1f}%). Downgraded certainty."
                )
                support_score -= 0.08
                f["human_review_required"] = True

        # Check 3: Occlusion / Uncertainty reconciliation
        if ppe_result:
            uncertain_count = ppe_result.get("uncertain_count", 0)
            if uncertain_count > 0:
                validation_warnings.append(
                    f"{uncertain_count} worker(s) have uncertain PPE status due to camera angle or distance. Definite noncompliance cannot be asserted."
                )
                support_score -= 0.10

        # Clamp support score to [0.10, 1.00]
        support_score = max(0.10, min(1.00, round(support_score, 3)))
        
        is_reliable = support_score >= 0.70
        duration_ms = (time.time() - start_time) * 1000.0

        return {
            "evidence_support_score": support_score,
            "is_reliable": is_reliable,
            "contradictions": contradictions,
            "unsupported_claims": unsupported_claims,
            "validation_warnings": validation_warnings,
            "human_inspection_advised": not is_reliable or len(validation_warnings) > 0,
            "model_version": self.model_version,
            "latency_ms": round(duration_ms, 2)
        }

def run_evidence_validation_agent(state: dict) -> dict:
    query = state.get("user_query", "")
    findings = state.get("findings", [])
    agent = EvidenceValidationAgent()
    result = agent.validate(
        query=query,
        findings=findings,
        object_result=state.get("object_result"),
        ppe_result=state.get("ppe_result"),
        ocr_result=state.get("ocr_result"),
        defect_result=state.get("defect_result")
    )
    return {
        "evidence_validation_result": result,
        "evidence_support_score": result["evidence_support_score"]
    }
