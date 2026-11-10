import time
import os
from typing import Dict, Any, List
import easyocr
import numpy as np
from backend.app.agents.base import BaseSpecialistAgent, AgentInput, AgentOutput, ExecutionStatus

class OCRAgent(BaseSpecialistAgent):
    """
    Agent D: Optical Character Recognition (OCR) Agent
    Uses EasyOCR to extract visible text, serial numbers, error codes, and labels.
    Preserves raw uncertain text and outputs bounding polygon coordinates.
    """
    _reader_instance = None

    def __init__(self, languages: List[str] = None):
        super().__init__(agent_name="ocr", model_version="EasyOCR-1.7.2")
        self.languages = languages or ["en"]

    def _get_reader(self):
        if OCRAgent._reader_instance is None:
            OCRAgent._reader_instance = easyocr.Reader(self.languages, gpu=False)
        return OCRAgent._reader_instance

    def _execute_internal(self, agent_input: AgentInput) -> AgentOutput:
        start_time = time.time()
        image_path = agent_input.image_path
        if not image_path or not os.path.exists(image_path):
            return AgentOutput(
                agent_name=self.agent_name,
                status=ExecutionStatus.FAILED,
                error_details=f"Image file not found: {image_path}",
                provenance={"agent": self.agent_name, "version": self.model_version}
            )

        reader = self._get_reader()
        results = reader.readtext(image_path)

        text_regions: List[Dict[str, Any]] = []
        findings: List[Dict[str, Any]] = []
        extracted_lines: List[str] = []

        for idx, item in enumerate(results):
            bbox_points, text, confidence = item
            pts = [[float(p[0]), float(p[1])] for p in bbox_points]
            min_x = min(p[0] for p in pts)
            min_y = min(p[1] for p in pts)
            max_x = max(p[0] for p in pts)
            max_y = max(p[1] for p in pts)

            finding_uid = f"F-OCR-{idx+1:02d}"
            region_dict = {
                "finding_uid": finding_uid,
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": [round(min_x, 1), round(min_y, 1), round(max_x, 1), round(max_y, 1)],
                "polygon": pts
            }
            text_regions.append(region_dict)
            extracted_lines.append(text)

            is_uncertain = confidence < 0.60
            score_desc = f"{confidence*100:.1f}% OCR recognition confidence: {'Clear' if not is_uncertain else 'Degraded / low contrast'} optical text character recognition."
            
            findings.append({
                "finding_uid": finding_uid,
                "agent_source": "ocr",
                "label": f"Text: '{text}'" if not is_uncertain else f"Uncertain Text: '{text}'",
                "bbox": [round(min_x, 1), round(min_y, 1), round(max_x, 1), round(max_y, 1)],
                "detection_confidence": round(float(confidence), 4),
                "anomaly_score": None,
                "routing_score": None,
                "observed_evidence": f"OCR extracted text '{text}' with confidence {confidence*100:.1f}%. Raw uncorrected string preserved.",
                "root_cause_hypothesis": None,
                "alternative_explanations": "Optical distortion, low contrast, or motion blur may affect character clarity." if is_uncertain else None,
                "recommendation": "Verify physical nameplate or barcode scanner log if critical for asset registry." if is_uncertain else None,
                "verification_procedure": "Cross-reference text against digital asset registry or maintenance inventory.",
                "human_review_required": is_uncertain,
                "model_name": "EasyOCR Text Recognizer",
                "model_version": "v1.7.2",
                "score_interpretation": score_desc,
                "person_id": None,
                "is_confirmed_defect": None
            })

        duration_ms = (time.time() - start_time) * 1000.0

        return AgentOutput(
            agent_name=self.agent_name,
            status=ExecutionStatus.SUCCESS,
            measured_runtime_ms=round(duration_ms, 2),
            findings=findings,
            data={
                "detected_text": " | ".join(extracted_lines) if extracted_lines else "No text detected",
                "text_regions": text_regions,
                "total_lines": len(extracted_lines)
            },
            provenance={
                "model": "EasyOCR",
                "languages": self.languages,
                "timestamp": time.time()
            }
        )

    def analyze(self, image_path: str) -> Dict[str, Any]:
        agent_input = AgentInput(
            session_id="adhoc",
            user_query="ocr text extraction",
            image_path=image_path
        )
        output = self.execute(agent_input)
        return {
            "detected_text": output.data.get("detected_text", "No text detected"),
            "text_regions": output.data.get("text_regions", []),
            "findings": output.findings,
            "languages": self.languages,
            "model_version": self.model_version,
            "latency_ms": output.measured_runtime_ms,
            "status": output.status.value,
            "error": output.error_details
        }

def run_ocr_agent(state: dict) -> dict:
    image_path = state.get("image_path")
    agent = OCRAgent()
    result = agent.analyze(image_path)
    return {
        "ocr_result": result,
        "new_findings": result.get("findings", [])
    }
