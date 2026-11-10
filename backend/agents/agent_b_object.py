import time
import os
from typing import Dict, Any, List
from ultralytics import YOLO
from backend.config import settings

class ObjectDetectionAgent:
    """
    Agent B: Object Detection Agent
    Executes YOLOv11 object detector, extracting bounding boxes, confidence,
    class labels, model metadata, and inference latency.
    """
    _model_instance = None

    def __init__(self, model_path: str = None):
        self.agent_name = "object"
        self.model_path = model_path or settings.MODEL_YOLO_PATH
        self.model_version = "YOLOv11n (Ultralytics 8.3.0)"

    def _get_model(self):
        if ObjectDetectionAgent._model_instance is None:
            if os.path.exists(self.model_path):
                ObjectDetectionAgent._model_instance = YOLO(self.model_path)
            else:
                # Fallback to downloading or standard yolo11n
                ObjectDetectionAgent._model_instance = YOLO("yolo11n.pt")
        return ObjectDetectionAgent._model_instance

    def analyze(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found: {image_path}",
                "objects": [],
                "findings": [],
                "latency_ms": 0.0
            }

        try:
            model = self._get_model()
            results = model(image_path, verbose=False)
            
            detected_objects: List[Dict[str, Any]] = []
            findings: List[Dict[str, Any]] = []
            
            img_h, img_w = None, None
            if len(results) > 0 and results[0].orig_shape is not None:
                img_h, img_w = results[0].orig_shape

            for r_idx, result in enumerate(results):
                boxes = result.boxes
                for b_idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    label = model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                    
                    # Normalized coordinates [0, 1]
                    norm_bbox = None
                    if img_w and img_h and img_w > 0 and img_h > 0:
                        norm_bbox = [
                            round(xyxy[0] / img_w, 4),
                            round(xyxy[1] / img_h, 4),
                            round(xyxy[2] / img_w, 4),
                            round(xyxy[3] / img_h, 4)
                        ]

                    finding_uid = f"F-OBJ-{len(detected_objects) + 1:02d}"
                    obj_dict = {
                        "finding_uid": finding_uid,
                        "label": label,
                        "confidence": round(conf, 4),
                        "bbox": [round(c, 1) for c in xyxy],
                        "norm_bbox": norm_bbox
                    }
                    detected_objects.append(obj_dict)

                    # Create standard explainable finding
                    findings.append({
                        "finding_uid": finding_uid,
                        "agent_source": "object",
                        "label": label,
                        "bbox": [round(c, 1) for c in xyxy],
                        "detection_confidence": round(conf, 4),
                        "anomaly_score": None,
                        "routing_score": None,
                        "observed_evidence": f"Detected {label} with confidence {conf*100:.1f}% at bounding box [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}].",
                        "root_cause_hypothesis": None,
                        "alternative_explanations": None,
                        "recommendation": None,
                        "verification_procedure": None,
                        "human_review_required": False
                    })

            duration_ms = (time.time() - start_time) * 1000.0

            return {
                "objects": detected_objects,
                "findings": findings,
                "total_count": len(detected_objects),
                "model_version": self.model_version,
                "model_path": self.model_path,
                "latency_ms": round(duration_ms, 2)
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000.0
            return {
                "error": f"Object detection failed: {str(e)}",
                "objects": [],
                "findings": [],
                "latency_ms": round(duration_ms, 2)
            }

def run_object_agent(state: dict) -> dict:
    image_path = state.get("image_path")
    agent = ObjectDetectionAgent()
    result = agent.analyze(image_path)
    return {
        "object_result": result,
        "new_findings": result.get("findings", [])
    }
