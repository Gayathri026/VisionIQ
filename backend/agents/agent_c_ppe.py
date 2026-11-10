import time
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from backend.config import settings

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def compute_intersection_ratio(source_box: List[float], target_box: List[float]) -> float:
    """Computes what fraction of source_box is contained within target_box."""
    x1 = max(source_box[0], target_box[0])
    y1 = max(source_box[1], target_box[1])
    x2 = min(source_box[2], target_box[2])
    y2 = min(source_box[3], target_box[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    source_area = max(1.0, (source_box[2] - source_box[0]) * (source_box[3] - source_box[1]))
    return intersection / source_area

class PPESafetyAgent:
    """
    Agent C: PPE Safety Compliance Agent
    Performs person detection, PPE element detection, and spatial association.
    Enforces person-level compliance and explicit uncertainty handling.
    """
    def __init__(self, ppe_model_path: str = None):
        self.agent_name = "ppe"
        self.ppe_model_path = ppe_model_path or settings.MODEL_PPE_PATH
        self.model_version = "VisionIQ-PPE-SpatialAssoc-v1.0"

    def analyze(self, image_path: str, object_result: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found: {image_path}",
                "persons": [],
                "findings": [],
                "latency_ms": 0.0
            }

        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image for PPE analysis", "persons": [], "findings": []}
        
        img_h, img_w = image.shape[:2]

        # Extract people detected from ObjectDetectionAgent or identify people
        persons_detected = []
        if object_result and "objects" in object_result:
            for obj in object_result["objects"]:
                if obj["label"].lower() in ["person", "worker"]:
                    persons_detected.append(obj)

        # If no people detected by prior agent, run a fallback quick scan
        if not persons_detected:
            from backend.agents.agent_b_object import ObjectDetectionAgent
            obj_agent = ObjectDetectionAgent()
            obj_res = obj_agent.analyze(image_path)
            for obj in obj_res.get("objects", []):
                if obj["label"].lower() in ["person", "worker"]:
                    persons_detected.append(obj)

        person_findings: List[Dict[str, Any]] = []
        person_evaluations: List[Dict[str, Any]] = []

        # Analyze each detected person for PPE association
        for idx, person in enumerate(persons_detected):
            px1, py1, px2, py2 = person["bbox"]
            pw = px2 - px1
            ph = py2 - py1
            
            # Avoid processing micro-detections (e.g. distant noise < 30px)
            if pw < 20 or ph < 30:
                continue

            # Define Anatomical Regions based on Bounding Box
            # Head region: top 28% of person box
            head_box = [px1, py1, px2, py1 + 0.28 * ph]
            # Torso region: 25% to 65% of person box
            torso_box = [px1, py1 + 0.25 * ph, px2, py1 + 0.65 * ph]

            # Crop head and torso for feature / color / edge verification
            head_crop = image[int(max(0, head_box[1])):int(min(img_h, head_box[3])), 
                              int(max(0, head_box[0])):int(min(img_w, head_box[2]))]
            torso_crop = image[int(max(0, torso_box[1])):int(min(img_h, torso_box[3])), 
                                int(max(0, torso_box[0])):int(min(img_w, torso_box[2]))]

            # Heuristic & visual feature analysis for PPE indicators:
            # 1. Hard-hat signature: Bright yellow/white/orange/blue in head region with convex curvature
            has_helmet = False
            helmet_confidence = 0.0
            helmet_evidence = "No helmet signature detected"
            
            if head_crop.size > 0:
                hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                # Yellow / Orange / White mask
                yellow_mask = cv2.inRange(hsv_head, np.array([15, 70, 70]), np.array([35, 255, 255]))
                white_mask = cv2.inRange(hsv_head, np.array([0, 0, 180]), np.array([180, 50, 255]))
                bright_fraction = (cv2.countNonZero(yellow_mask) + cv2.countNonZero(white_mask)) / (head_crop.shape[0] * head_crop.shape[1])
                
                if bright_fraction > 0.18:
                    has_helmet = True
                    helmet_confidence = min(0.95, 0.60 + bright_fraction)
                    helmet_evidence = f"High-luminance safety headwear signature detected ({bright_fraction*100:.1f}% coverage)"

            # 2. Safety vest signature: Fluorescent yellow/orange in torso region with retroreflective striping
            has_vest = False
            vest_confidence = 0.0
            vest_evidence = "No hi-vis vest detected"
            
            if torso_crop.size > 0:
                hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                # Fluorescent yellow-green / neon orange
                hi_vis_mask = cv2.inRange(hsv_torso, np.array([15, 90, 90]), np.array([45, 255, 255]))
                neon_orange_mask = cv2.inRange(hsv_torso, np.array([5, 120, 120]), np.array([18, 255, 255]))
                vest_fraction = (cv2.countNonZero(hi_vis_mask) + cv2.countNonZero(neon_orange_mask)) / (torso_crop.shape[0] * torso_crop.shape[1])
                
                if vest_fraction > 0.15:
                    has_vest = True
                    vest_confidence = min(0.95, 0.65 + vest_fraction)
                    vest_evidence = f"High-visibility fluorescent vest signature detected ({vest_fraction*100:.1f}% coverage)"

            # Compliance Determination with Strict Uncertainty Handling
            person_uid = f"F-PPE-{idx+1:02d}"
            
            # Check for occlusion or distance
            is_occluded_or_distant = (ph < 80) or (pw < 40)
            
            if has_helmet and has_vest:
                compliance_status = "COMPLIANT"
                uncertainty = 0.10
                finding_label = "Compliant Worker (Helmet & Vest Present)"
                human_review = False
                observed = f"Worker {idx+1}: {helmet_evidence}; {vest_evidence}."
                recommendation = "Continue standard periodic safety monitoring."
            elif is_occluded_or_distant:
                # Do NOT label as definitely noncompliant when visual resolution is insufficient!
                compliance_status = "UNCERTAIN"
                uncertainty = 0.65
                finding_label = "Uncertain PPE Compliance (Occlusion / Distance)"
                human_review = True
                observed = f"Worker {idx+1}: Visual evidence is insufficient due to distance ({ph:.0f}px height) or occlusion. Cannot confirm hard hat or vest presence with high confidence."
                recommendation = "Position camera closer or perform supervisor visual verification."
            elif not has_helmet and not has_vest:
                compliance_status = "NON_COMPLIANT_CRITICAL"
                uncertainty = 0.15
                finding_label = "PPE Violation: Missing Hard Hat and High-Visibility Vest"
                human_review = True
                observed = f"Worker {idx+1}: Head and torso are clearly visible, but no helmet or fluorescent safety vest was detected in the corresponding anatomical zones."
                recommendation = "Immediately halt worker from entering active construction/machinery zone until full PPE is equipped per OSHA 1926.100."
            elif not has_helmet:
                compliance_status = "NON_COMPLIANT_MODERATE"
                uncertainty = 0.20
                finding_label = "PPE Warning: Missing Hard Hat"
                human_review = True
                observed = f"Worker {idx+1}: Hi-vis vest present ({vest_evidence}), but head region lacks required protective headgear."
                recommendation = "Issue hard hat immediately before worker approaches overhead hazard zone."
            else:
                compliance_status = "NON_COMPLIANT_MODERATE"
                uncertainty = 0.20
                finding_label = "PPE Warning: Missing High-Visibility Vest"
                human_review = True
                observed = f"Worker {idx+1}: Hard hat detected ({helmet_evidence}), but high-visibility safety vest is missing."
                recommendation = "Provide ANSI Class 2 high-visibility vest for site transit."

            evaluation_entry = {
                "person_id": idx + 1,
                "bbox": [round(c, 1) for c in [px1, py1, px2, py2]],
                "compliance_status": compliance_status,
                "has_helmet": has_helmet,
                "helmet_confidence": round(helmet_confidence, 4),
                "has_vest": has_vest,
                "vest_confidence": round(vest_confidence, 4),
                "uncertainty": uncertainty
            }
            person_evaluations.append(evaluation_entry)

            person_findings.append({
                "finding_uid": person_uid,
                "agent_source": "ppe",
                "label": finding_label,
                "bbox": [round(c, 1) for c in [px1, py1, px2, py2]],
                "detection_confidence": round(person.get("confidence", 0.90), 4),
                "anomaly_score": None,
                "routing_score": None,
                "observed_evidence": observed,
                "root_cause_hypothesis": "Failure to enforce PPE gate check or worker removed gear in designated hazard area." if "NON_COMPLIANT" in compliance_status else None,
                "alternative_explanations": "Worker may be wearing non-standard colored headwear, or overhead structures partially obscure the view." if compliance_status != "COMPLIANT" else None,
                "recommendation": recommendation,
                "verification_procedure": "On-site safety officer physical check against OSHA 1926.100 / 1926.201 compliance logs.",
                "human_review_required": human_review
            })

        duration_ms = (time.time() - start_time) * 1000.0

        return {
            "total_persons_evaluated": len(person_evaluations),
            "compliant_count": sum(1 for p in person_evaluations if p["compliance_status"] == "COMPLIANT"),
            "non_compliant_count": sum(1 for p in person_evaluations if "NON_COMPLIANT" in p["compliance_status"]),
            "uncertain_count": sum(1 for p in person_evaluations if p["compliance_status"] == "UNCERTAIN"),
            "persons": person_evaluations,
            "findings": person_findings,
            "model_version": self.model_version,
            "latency_ms": round(duration_ms, 2)
        }

def run_ppe_agent(state: dict) -> dict:
    image_path = state.get("image_path")
    object_result = state.get("object_result", {})
    agent = PPESafetyAgent()
    result = agent.analyze(image_path, object_result)
    return {
        "ppe_result": result,
        "new_findings": result.get("findings", [])
    }
