import os
import cv2
import base64
import time
import uuid
import numpy as np
from typing import Dict, Any
from backend.config import settings
from backend.orchestrator.graph import vision_iq_app

class CameraService:
    """
    Handles live camera streaming inference from browser webcam feeds.
    Decodes base64 JPEG payloads, runs selected specialist agents, and returns
    live bounding boxes, findings, and current agent activity traces.
    """
    def __init__(self):
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)

    def process_camera_frame(
        self,
        frame_base64: str,
        query: str,
        timestamp: float = 0.0
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # Strip header if present (e.g. data:image/jpeg;base64,...)
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",")[1]

        try:
            image_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode base64 image data")
        except Exception as e:
            return {"error": f"Invalid base64 camera frame: {str(e)}", "findings": []}

        temp_cam_file = os.path.join(settings.UPLOADS_DIR, f"cam_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(temp_cam_file, img)

        try:
            state = {
                "image_path": temp_cam_file,
                "user_query": query,
                "input_type": "camera"
            }
            res_state = vision_iq_app.invoke(state)
            resp = res_state.get("final_response", {})

            # Encode annotated image back to base64 for real-time frontend display
            annotated_path = resp.get("annotated_path")
            annotated_base64 = ""
            if annotated_path and os.path.exists(annotated_path):
                with open(annotated_path, "rb") as f:
                    annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

            total_latency = (time.time() - start_time) * 1000.0

            return {
                "status": "success",
                "timestamp": timestamp,
                "latency_ms": round(total_latency, 2),
                "detected_intent": resp.get("detected_intent"),
                "selected_agents": resp.get("selected_agents", []),
                "executed_agents": resp.get("executed_agents", []),
                "findings": resp.get("findings", []),
                "annotated_frame_base64": annotated_base64,
                "synthesis": resp.get("synthesis_text")
            }
        finally:
            if os.path.exists(temp_cam_file):
                try:
                    os.remove(temp_cam_file)
                except Exception:
                    pass

camera_service = CameraService()
