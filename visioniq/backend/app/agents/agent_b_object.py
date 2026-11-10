import time
import os
from typing import Dict, Any, List
from ultralytics import YOLO
from backend.app.core.config import settings
from backend.app.agents.base import BaseSpecialistAgent, AgentInput, AgentOutput, ExecutionStatus

class ObjectDetectionAgent(BaseSpecialistAgent):
    """
    Agent B: Object Detection Agent
    Executes YOLOv11 object detector, extracting bounding boxes, confidence,
    class labels, model metadata, and inference latency.
    """
    _model_instance = None

    def __init__(self, model_path: str = None):
        super().__init__(agent_name="object", model_version="YOLOv11n (Ultralytics 8.3.0)")
        self.model_path = model_path or settings.MODEL_YOLO_PATH

    def _get_model(self):
        if ObjectDetectionAgent._model_instance is None:
            # Check configured path, then model_registry, then local root
            candidate_paths = [
                self.model_path,
                os.path.join(settings.BASE_DIR, "visioniq/ml/model_registry/yolo11n.pt"),
                os.path.join(settings.BASE_DIR, "ml/model_registry/yolo11n.pt"),
                "yolo11n.pt"
            ]
            loaded = False
            for p in candidate_paths:
                if p and os.path.exists(p):
                    ObjectDetectionAgent._model_instance = YOLO(p)
                    loaded = True
                    break
            if not loaded:
                ObjectDetectionAgent._model_instance = YOLO("yolo11n.pt")
        return ObjectDetectionAgent._model_instance

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

                clamped_xyxy = [
                    round(max(0.0, min(float(img_w or xyxy[0]), xyxy[0])), 1),
                    round(max(0.0, min(float(img_h or xyxy[1]), xyxy[1])), 1),
                    round(max(0.0, min(float(img_w or xyxy[2]), xyxy[2])), 1),
                    round(max(0.0, min(float(img_h or xyxy[3]), xyxy[3])), 1)
                ]

                norm_bbox = None
                if img_w and img_h and img_w > 0 and img_h > 0:
                    norm_bbox = [
                        round(clamped_xyxy[0] / img_w, 4),
                        round(clamped_xyxy[1] / img_h, 4),
                        round(clamped_xyxy[2] / img_w, 4),
                        round(clamped_xyxy[3] / img_h, 4)
                    ]

                finding_uid = f"F-OBJ-{len(detected_objects) + 1:02d}"
                score_desc = f"{conf*100:.1f}% detection confidence: {'High' if conf >= 0.80 else 'Moderate' if conf >= 0.50 else 'Low'} confidence identification for {label}."
                
                obj_dict = {
                    "finding_uid": finding_uid,
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": clamped_xyxy,
                    "norm_bbox": norm_bbox,
                    "score_interpretation": score_desc
                }
                detected_objects.append(obj_dict)

                findings.append({
                    "finding_uid": finding_uid,
                    "agent_source": "object",
                    "label": label,
                    "bbox": clamped_xyxy,
                    "detection_confidence": round(conf, 4),
                    "anomaly_score": None,
                    "routing_score": None,
                    "observed_evidence": f"Detected {label} with confidence {conf*100:.1f}% at bounding box [{int(clamped_xyxy[0])}, {int(clamped_xyxy[1])}, {int(clamped_xyxy[2])}, {int(clamped_xyxy[3])}].",
                    "root_cause_hypothesis": None,
                    "alternative_explanations": "Possible visual ambiguity with background or partial occlusion." if conf < 0.65 else None,
                    "recommendation": "Review high-resolution camera stream if critical asset identification is required." if conf < 0.60 else None,
                    "verification_procedure": "Cross-reference asset registry or conduct direct visual inspection.",
                    "human_review_required": conf < 0.50,
                    "model_name": "Ultralytics YOLO11n",
                    "model_version": "v1.0.0",
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
                "objects": detected_objects,
                "total_count": len(detected_objects),
                "model_version": self.model_version
            },
            provenance={
                "model": "YOLOv11n",
                "classes_available": len(model.names),
                "timestamp": time.time()
            }
        )

    def analyze(self, image_path: str) -> Dict[str, Any]:
        agent_input = AgentInput(
            session_id="adhoc",
            user_query="object detection",
            image_path=image_path
        )
        output = self.execute(agent_input)
        return {
            "objects": output.data.get("objects", []),
            "findings": output.findings,
            "total_count": output.data.get("total_count", 0),
            "model_version": self.model_version,
            "latency_ms": output.measured_runtime_ms,
            "status": output.status.value,
            "error": output.error_details
        }

def run_object_agent(state: dict) -> dict:
    image_path = state.get("image_path")
    agent = ObjectDetectionAgent()
    result = agent.analyze(image_path)
    return {
        "object_result": result,
        "new_findings": result.get("findings", [])
    }
