import time
import os
from typing import Dict, Any, List
import easyocr
import numpy as np

class OCRAgent:
    """
    Agent D: Optical Character Recognition (OCR) Agent
    Uses EasyOCR to extract visible text, serial numbers, error codes, and labels.
    Preserves raw uncertain text and outputs bounding polygon coordinates.
    """
    _reader_instance = None

    def __init__(self, languages: List[str] = None):
        self.agent_name = "ocr"
        self.languages = languages or ["en"]
        self.model_version = "EasyOCR-1.7.2"

    def _get_reader(self):
        if OCRAgent._reader_instance is None:
            # CPU mode fallback enabled by default
            OCRAgent._reader_instance = easyocr.Reader(self.languages, gpu=False)
        return OCRAgent._reader_instance

    def analyze(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found: {image_path}",
                "text_regions": [],
                "findings": [],
                "latency_ms": 0.0
            }

        try:
            reader = self._get_reader()
            results = reader.readtext(image_path)
            
            text_regions: List[Dict[str, Any]] = []
            findings: List[Dict[str, Any]] = []
            extracted_lines: List[str] = []

            for idx, item in enumerate(results):
                bbox_points, text, confidence = item
                # bbox_points is 4 corners [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
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

                # Flag uncertain text if confidence is below 0.60
                is_uncertain = confidence < 0.60
                findings.append({
                    "finding_uid": finding_uid,
                    "agent_source": "ocr",
                    "label": f"Text: '{text}'" if not is_uncertain else f"Uncertain Text: '{text}'",
                    "bbox": [round(min_x, 1), round(min_y, 1), round(max_x, 1), round(max_y, 1)],
                    "detection_confidence": round(float(confidence), 4),
                    "anomaly_score": None,
                    "routing_score": None,
                    "observed_evidence": f"OCR extracted text '{text}' with character confidence {confidence*100:.1f}%. Raw uncorrected string preserved.",
                    "root_cause_hypothesis": None,
                    "alternative_explanations": "Optical distortion, low contrast, or motion blur may cause character misinterpretation." if is_uncertain else None,
                    "recommendation": "Verify physical nameplate or barcode scanner log if critical for machine identity." if is_uncertain else None,
                    "verification_procedure": "Cross-reference text against digital asset registry or maintenance inventory.",
                    "human_review_required": is_uncertain
                })

            duration_ms = (time.time() - start_time) * 1000.0

            return {
                "detected_text": " | ".join(extracted_lines) if extracted_lines else "No text detected",
                "text_regions": text_regions,
                "findings": findings,
                "languages": self.languages,
                "model_version": self.model_version,
                "latency_ms": round(duration_ms, 2)
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000.0
            return {
                "error": f"OCR extraction failed: {str(e)}",
                "text_regions": [],
                "findings": [],
                "latency_ms": round(duration_ms, 2)
            }

def run_ocr_agent(state: dict) -> dict:
    image_path = state.get("image_path")
    agent = OCRAgent()
    result = agent.analyze(image_path)
    return {
        "ocr_result": result,
        "new_findings": result.get("findings", [])
    }
