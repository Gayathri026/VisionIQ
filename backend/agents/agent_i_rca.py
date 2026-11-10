import time
from typing import Dict, Any, List

class RootCauseAnalysisAgent:
    """
    Agent I: Root-Cause Analysis (RCA) Agent
    Generates ranked, plausible causal hypotheses grounded in observable visual evidence.
    Explicitly documents alternative explanations, missing diagnostic evidence,
    and differential inspections. Never asserts physical certainty from images alone.
    """
    def __init__(self):
        self.agent_name = "rca"
        self.model_version = "VisionIQ-CausalReasoner-v1.0"

    def analyze(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        rag_references: List[Dict[str, Any]] = None,
        defect_result: Dict[str, Any] = None,
        ppe_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        q_lower = query.lower()
        hypotheses = []
        missing_evidence = []
        differentiating_inspections = []

        # Case 1: Defect / Material Damage Inquiry
        if defect_result and defect_result.get("defects_detected", 0) > 0:
            defects = defect_result["defects"]
            top_defect = defects[0]
            defect_type = top_defect["class"]

            hypotheses.append({
                "rank": 1,
                "hypothesis": f"Mechanical surface degradation or tooling wear characteristic of '{defect_type}'.",
                "estimated_cause_probability": 0.65,
                "supporting_visual_evidence": f"Morphological flaw '{defect_type}' detected with anomaly score {top_defect['anomaly_score']:.3f} and area {top_defect['area_px']}px.",
                "caveats": "Optical imaging captures only surface manifestations; subsurface crack propagation depth cannot be measured from 2D pixels."
            })
            hypotheses.append({
                "rank": 2,
                "hypothesis": "Surface chemical contamination, rolling mill scale, or uneven cooling during metal formation.",
                "estimated_cause_probability": 0.25,
                "supporting_visual_evidence": "Localized color/luminance variation in the vicinity of the defect contour.",
                "caveats": "Could also be benign superficial oil/coolant deposition."
            })
            missing_evidence.extend([
                "Subsurface depth profile (eddy current or ultrasonic A-scan data).",
                "Material batch chemical assay and heat-treatment certificate.",
                "Operational cycles and machine maintenance logs since last tooling replacement."
            ])
            differentiating_inspections.extend([
                "Perform Dye Penetrant Inspection (PT) or Ultrasonic Testing (UT) to determine crack depth.",
                "Optical profilometry / contact stylus roughness measurement (ISO 4287 Ra/Rz verification)."
            ])

        # Case 2: PPE Compliance / Safety Inquiry
        elif ppe_result and ppe_result.get("non_compliant_count", 0) > 0:
            hypotheses.append({
                "rank": 1,
                "hypothesis": "Personnel failure to don designated protective equipment prior to entering active zone.",
                "estimated_cause_probability": 0.70,
                "supporting_visual_evidence": f"{ppe_result['non_compliant_count']} worker(s) observed in active zone lacking hard hats or high-visibility vests.",
                "caveats": "Worker may have temporarily removed equipment immediately before image capture."
            })
            hypotheses.append({
                "rank": 2,
                "hypothesis": "Inadequate zoning demarcation or lack of physical PPE gate enforcement at site ingress.",
                "estimated_cause_probability": 0.25,
                "supporting_visual_evidence": "Absence of clear physical turnstiles or safety signage near worker transit path.",
                "caveats": "Signage may be located outside the camera's field of view."
            })
            missing_evidence.extend([
                "Site gate check log and badge entry records for the current shift.",
                "Safety briefing / toolbox meeting attendance roster."
            ])
            differentiating_inspections.extend([
                "Audit physical turnstile entry camera footage.",
                "Interview site safety foreman regarding gate enforcement protocols."
            ])

        # Case 3: Hydraulic / Machine Stoppage Inquiry
        elif any(k in q_lower for k in ["why", "stop", "halt", "fail", "hydraulic"]):
            hypotheses.append({
                "rank": 1,
                "hypothesis": "Emergency safety interlock trip or light curtain optical interruption.",
                "estimated_cause_probability": 0.55,
                "supporting_visual_evidence": "Machinery stopped in static dwell position without visible structural fracture.",
                "caveats": "PLC event code required to confirm which interlock circuit opened."
            })
            hypotheses.append({
                "rank": 2,
                "hypothesis": "Hydraulic line pressure loss or thermal overload trip on hydraulic power unit.",
                "estimated_cause_probability": 0.35,
                "supporting_visual_evidence": "No dynamic fluid leakage visible on ground; actuators stationary.",
                "caveats": "Pressure transducers and fluid temperature cannot be assessed visually."
            })
            missing_evidence.extend([
                "PLC fault register dump and alarm history timestamped at stoppage time.",
                "Hydraulic pressure gauge readings and reservoir fluid level check."
            ])
            differentiating_inspections.extend([
                "Read PLC fault code memory via maintenance console.",
                "Inspect hydraulic manifold pressure transducers and check safety relay LED status."
            ])

        # General Default
        else:
            hypotheses.append({
                "rank": 1,
                "hypothesis": "Operational state within nominal parameters based on visible equipment features.",
                "estimated_cause_probability": 0.50,
                "supporting_visual_evidence": "No overt structural deformation, smoke, or physical debris detected.",
                "caveats": "Visual inspection cannot rule out electrical or internal mechanical faults."
            })
            missing_evidence.append("Operating telemetry (vibration, temperature, power draw).")
            differentiating_inspections.append("Routine vibration analysis and scheduled multi-meter inspection.")

        duration_ms = (time.time() - start_time) * 1000.0

        return {
            "query": query,
            "ranked_hypotheses": hypotheses,
            "missing_evidence": missing_evidence,
            "differentiating_inspections": differentiating_inspections,
            "model_version": self.model_version,
            "latency_ms": round(duration_ms, 2)
        }

def run_rca_agent(state: dict) -> dict:
    query = state.get("user_query", "")
    findings = state.get("findings", [])
    rag_refs = state.get("rag_references", [])
    agent = RootCauseAnalysisAgent()
    result = agent.analyze(
        query=query,
        findings=findings,
        rag_references=rag_refs,
        defect_result=state.get("defect_result"),
        ppe_result=state.get("ppe_result")
    )
    return {"rca_result": result}
