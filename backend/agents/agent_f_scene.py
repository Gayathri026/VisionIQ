import time
import os
import cv2
import numpy as np
from typing import Dict, Any, List

class SceneUnderstandingAgent:
    """
    Agent F: Spatial Scene Understanding Agent
    Analyzes overall environment, lighting, spatial density, and spatial relationships.
    Strictly separates direct objective observations from contextual interpretations.
    """
    def __init__(self):
        self.agent_name = "scene"
        self.model_version = "VisionIQ-SceneContext-v1.0"

    def analyze(self, image_path: str, object_result: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found: {image_path}",
                "direct_observations": [],
                "contextual_interpretations": [],
                "latency_ms": 0.0
            }

        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Failed to read image", "direct_observations": [], "contextual_interpretations": []}

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Illumination & Contrast Metrics
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        if mean_brightness < 60:
            lighting_desc = "Low-light / underexposed environment"
        elif mean_brightness > 190:
            lighting_desc = "High-key / direct glare illumination"
        else:
            lighting_desc = "Standard diffused industrial lighting"

        # 2. Object Spatial Distribution & Density
        objects = object_result.get("objects", []) if object_result else []
        obj_count = len(objects)
        
        # Spatial quadrant distribution
        quadrants = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
        total_bbox_area = 0.0

        for obj in objects:
            bx1, by1, bx2, by2 = obj["bbox"]
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            total_bbox_area += (bx2 - bx1) * (by2 - by1)

            if cx < w / 2 and cy < h / 2:
                quadrants["top_left"] += 1
            elif cx >= w / 2 and cy < h / 2:
                quadrants["top_right"] += 1
            elif cx < w / 2 and cy >= h / 2:
                quadrants["bottom_left"] += 1
            else:
                quadrants["bottom_right"] += 1

        density_fraction = min(1.0, total_bbox_area / float(h * w))
        if density_fraction > 0.40 or obj_count > 10:
            density_desc = "Congested / High-density working area"
        elif density_fraction > 0.15 or obj_count > 3:
            density_desc = "Moderate operational density"
        else:
            density_desc = "Sparse / Low-density layout"

        # 3. Direct Observations (Purely objective facts)
        direct_observations = [
            f"Image resolution: {w}x{h} pixels ({w*h/1e6:.2f} Megapixels).",
            f"Mean scene luminance: {mean_brightness:.1f}/255 ({lighting_desc}).",
            f"Identified {obj_count} visual entities covering {density_fraction*100:.1f}% of total image area.",
            f"Spatial entity distribution across quadrants: Top-Left: {quadrants['top_left']}, Top-Right: {quadrants['top_right']}, Bottom-Left: {quadrants['bottom_left']}, Bottom-Right: {quadrants['bottom_right']}."
        ]

        # 4. Contextual Interpretations (Qualified hypotheses)
        labels = [o["label"].lower() for o in objects]
        interpretations = []
        
        if any(l in ["person", "worker"] for l in labels) and any(l in ["truck", "car", "machinery"] for l in labels):
            interpretations.append("Active industrial site with shared worker-vehicle spatial transit paths.")
        elif any(l in ["person", "worker"] for l in labels):
            interpretations.append("Occupied human work cell under inspection.")
        else:
            interpretations.append("Static machinery or material surface inspection without detected personnel.")

        interpretations.append(f"Visual density classification: {density_desc}.")

        finding_uid = "F-SCN-01"
        finding = {
            "finding_uid": finding_uid,
            "agent_source": "scene",
            "label": "Spatial Scene Context Analysis",
            "bbox": None,
            "detection_confidence": 0.90,
            "anomaly_score": None,
            "routing_score": None,
            "observed_evidence": " | ".join(direct_observations),
            "root_cause_hypothesis": None,
            "alternative_explanations": "Variations in camera exposure or wide-angle lens distortion may influence peripheral quadrant counts.",
            "recommendation": "Maintain adequate 50-foot-candle illumination per OSHA 1926.56 standards." if mean_brightness < 70 else None,
            "verification_procedure": "Site safety supervisor light meter verification.",
            "human_review_required": False
        }

        duration_ms = (time.time() - start_time) * 1000.0

        return {
            "direct_observations": direct_observations,
            "contextual_interpretations": interpretations,
            "lighting": lighting_desc,
            "density": density_desc,
            "quadrants": quadrants,
            "findings": [finding],
            "model_version": self.model_version,
            "latency_ms": round(duration_ms, 2)
        }

def run_scene_agent(state: dict) -> dict:
    image_path = state.get("image_path")
    object_result = state.get("object_result", {})
    agent = SceneUnderstandingAgent()
    result = agent.analyze(image_path, object_result)
    return {
        "scene_result": result,
        "new_findings": result.get("findings", [])
    }
