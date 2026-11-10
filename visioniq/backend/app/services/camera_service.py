import os
import cv2
import time
import uuid
import base64
from typing import Dict, Any, List
import numpy as np

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.orchestration.router import route_query
from backend.app.agents.agent_b_object import ObjectDetectionAgent
from backend.app.agents.agent_c_ppe import PPESafetyAgent
from backend.app.agents.agent_e_defect import IndustrialDefectAgent
from backend.app.agents.agent_d_ocr import OCRAgent
from backend.app.services.annotation_service import annotation_service

object_agent = ObjectDetectionAgent()
ppe_agent = PPESafetyAgent()
defect_agent = IndustrialDefectAgent()
ocr_agent = OCRAgent()

class CameraInferenceService:
    """
    Real-time browser camera frame inference engine. Decodes live webcam frames,
    dynamically routes to appropriate vision specialists via Equation (1),
    annotates the frame with high-contrast evidence overlays, and encodes
    back to base64 for low-latency visual display.
    """
    def __init__(self):
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
        os.makedirs(settings.ANNOTATIONS_DIR, exist_ok=True)

    def process_frame(
        self,
        frame_base64: str,
        query: str = "Continuous PPE safety compliance monitoring",
        timestamp: float = 0.0
    ) -> Dict[str, Any]:
        """Processes a single live camera frame through the actual vision intelligence pipeline."""
        t0 = time.time()
        session_id = str(uuid.uuid4())

        if not frame_base64 or len(frame_base64.strip()) == 0:
            raise ValueError("Empty camera frame payload received.")

        # Strip data URL header if present (e.g., 'data:image/jpeg;base64,')
        raw_b64 = frame_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise ValueError(f"Corrupted base64 image encoding: {str(e)}")

        if len(image_bytes) < 100:
            raise ValueError("Camera frame payload too small to be a valid visual raster.")

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.shape[0] < 5 or frame.shape[1] < 5:
            raise ValueError("Unable to decode visual raster from camera frame payload.")

        # Persist temporary frame to disk for agent execution
        temp_frame_path = os.path.join(settings.UPLOADS_DIR, f"cam_{session_id[:8]}.jpg")
        cv2.imwrite(temp_frame_path, frame)

        executed_agents: List[str] = []
        findings: List[Dict[str, Any]] = []

        try:
            # 1. Routing decision based on query
            plan = route_query(query)
            selected_agents = plan.get("selected_agents", ["object"])
            detected_intent = plan.get("intent", "object_counting_detection")

            # 2. Execute selected specialist agents
            if "object" in selected_agents:
                obj_res = object_agent.analyze(image_path=temp_frame_path)
                findings.extend(obj_res.get("findings", []))
                executed_agents.append("object")

            if "ppe" in selected_agents:
                ppe_res = ppe_agent.analyze(image_path=temp_frame_path, object_result={"findings": findings})
                findings.extend(ppe_res.get("findings", []))
                executed_agents.append("ppe")

            if "defect" in selected_agents:
                def_res = defect_agent.analyze(image_path=temp_frame_path)
                findings.extend(def_res.get("findings", []))
                executed_agents.append("defect")

            if "ocr" in selected_agents:
                ocr_res = ocr_agent.analyze(image_path=temp_frame_path)
                findings.extend(ocr_res.get("findings", []))
                executed_agents.append("ocr")

            # Set frame timestamp on all findings
            for f in findings:
                f["frame_timestamp"] = timestamp

            # 3. Annotate frame with high-contrast bounding boxes
            ann_path = annotation_service.annotate_image(
                image_path=temp_frame_path,
                findings=findings,
                output_filename=f"ann_cam_{session_id[:8]}.jpg"
            )

            # Re-encode annotated image to base64 for real-time frontend streaming
            annotated_frame_b64 = None
            if os.path.exists(ann_path):
                with open(ann_path, "rb") as f_ann:
                    b64_bytes = base64.b64encode(f_ann.read()).decode("utf-8")
                    annotated_frame_b64 = f"data:image/jpeg;base64,{b64_bytes}"

            latency_ms = round((time.time() - t0) * 1000.0, 2)
            synthesis = (
                f"Webcam frame at t={timestamp:.2f}s processed in {latency_ms:.1f}ms. "
                f"Executed {len(executed_agents)} specialists ({', '.join(executed_agents)}), "
                f"detecting {len(findings)} visual entities."
            )

            return {
                "status": "success",
                "session_id": session_id,
                "timestamp": timestamp,
                "latency_ms": latency_ms,
                "detected_intent": detected_intent,
                "selected_agents": selected_agents,
                "executed_agents": executed_agents,
                "findings": findings,
                "annotated_frame_base64": annotated_frame_b64,
                "synthesis": synthesis
            }

        finally:
            # Resource cleanup: remove temporary frame
            if os.path.exists(temp_frame_path):
                try:
                    os.remove(temp_frame_path)
                except Exception:
                    pass

camera_inference_service = CameraInferenceService()
