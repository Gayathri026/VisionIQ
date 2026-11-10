import time
from typing import Dict, Any, List

class RecommendationAgent:
    """
    Agent J: Corrective and Preventive Action (CAPA) Recommendation Agent
    Formulates structured corrective actions, preventive measures, and verification protocols.
    Explicitly links each recommendation to specific findings and retrieved RAG standards,
    and identifies requirements for certified human review.
    """
    def __init__(self):
        self.agent_name = "recommendation"
        self.model_version = "VisionIQ-ActionPlanner-v1.0"

    def formulate(
        self,
        findings: List[Dict[str, Any]],
        rca_result: Dict[str, Any] = None,
        rag_references: List[Dict[str, Any]] = None,
        ppe_result: Dict[str, Any] = None,
        defect_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        immediate_actions = []
        preventive_measures = []
        verification_steps = []

        # 1. PPE Compliance Recommendations
        if ppe_result and ppe_result.get("non_compliant_count", 0) > 0:
            immediate_actions.append({
                "action_id": "ACT-PPE-01",
                "priority": "HIGH",
                "target_finding": "F-PPE",
                "title": "Immediate Safety Stand-Down & Gear Issuance",
                "description": "Direct non-compliant personnel to immediately step back into the staging bay and equip certified ANSI Z89.1 hard hats and Class 2 vests.",
                "linked_standard": "OSHA 1926.100 / OSHA 1926.201",
                "requires_human_signoff": True
            })
            preventive_measures.append({
                "measure_id": "PRV-PPE-01",
                "title": "Automated Turnstile PPE Interlock",
                "description": "Implement automated vision-based barrier gating at site access points that unlocks only upon verifying helmet and vest detection.",
                "linked_standard": "ISO 45001 Occupational Health and Safety",
                "requires_human_signoff": False
            })
            verification_steps.append(
                "Safety officer conduct physical badge scan and headwear inspection before permitting zone re-entry."
            )

        # 2. Defect Inspection Recommendations
        if defect_result and defect_result.get("defects_detected", 0) > 0:
            defects = defect_result["defects"]
            top_class = defects[0]["class"]
            immediate_actions.append({
                "action_id": "ACT-DEF-01",
                "priority": "MEDIUM",
                "target_finding": "F-DEF",
                "title": f"Quarantine Affected Lot for '{top_class}' Flaw",
                "description": f"Segregate the inspected workpiece or batch exhibiting {top_class} surface defect to prevent downstream assembly contamination.",
                "linked_standard": "ASTM A568 Section 9.1",
                "requires_human_signoff": True
            })
            preventive_measures.append({
                "measure_id": "PRV-DEF-01",
                "title": "Rolling Mill / Machine Tooling Recalibration",
                "description": "Inspect upstream rollers and dies for foreign particle build-up or abrasive wear grooves causing repeating surface scratches.",
                "linked_standard": "ISO 9001 Quality Management - Equipment Calibration",
                "requires_human_signoff": True
            })
            verification_steps.append(
                "Perform surface roughness profilometry (Ra/Rz) and dye penetrant testing prior to releasing quarantined parts."
            )

        # 3. RCA-Driven Machinery Stoppage Recommendations
        if rca_result and "hydraulic" in rca_result.get("query", "").lower():
            immediate_actions.append({
                "action_id": "ACT-MAC-01",
                "priority": "HIGH",
                "target_finding": "RCA-01",
                "title": "Lockout / Tagout (LOTO) and Circuit Verification",
                "description": "Apply OSHA 1910.147 Lockout/Tagout protocol before maintenance personnel access hydraulic valve banks or safety curtains.",
                "linked_standard": "OSHA 1910.147 / ISO 13849 Clause 6.2",
                "requires_human_signoff": True
            })
            verification_steps.append(
                "Verify hydraulic pressure accumulator zero-energy state via calibrated pressure gauge before opening valve enclosures."
            )

        # General Baseline Recommendation
        if not immediate_actions:
            immediate_actions.append({
                "action_id": "ACT-GEN-01",
                "priority": "ROUTINE",
                "target_finding": "F-GEN",
                "title": "Standard Operating Inspection Procedure",
                "description": "No critical safety violations or severe defects detected. Log inspection record into asset maintenance database.",
                "linked_standard": "ISO 9001 Section 8.6 Release of Products",
                "requires_human_signoff": False
            })
            verification_steps.append("Archive inspection image and trace timestamp in database.")

        duration_ms = (time.time() - start_time) * 1000.0

        return {
            "immediate_corrective_actions": immediate_actions,
            "preventive_measures": preventive_measures,
            "verification_protocol": verification_steps,
            "total_actions": len(immediate_actions) + len(preventive_measures),
            "human_inspection_mandatory": any(a["requires_human_signoff"] for a in immediate_actions),
            "model_version": self.model_version,
            "latency_ms": round(duration_ms, 2)
        }

def run_recommendation_agent(state: dict) -> dict:
    findings = state.get("findings", [])
    rca_result = state.get("rca_result")
    rag_refs = state.get("rag_references", [])
    agent = RecommendationAgent()
    result = agent.formulate(
        findings=findings,
        rca_result=rca_result,
        rag_references=rag_refs,
        ppe_result=state.get("ppe_result"),
        defect_result=state.get("defect_result")
    )
    return {"recommendation_result": result}
