import time
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings
from backend.app.agents.base import BaseSpecialistAgent, AgentInput, AgentOutput, ExecutionStatus

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

class PPESafetyAgent(BaseSpecialistAgent):
    """
    Agent C: PPE Safety Compliance Agent
    Performs person detection, PPE element detection, and spatial association.
    Enforces person-level compliance and explicit uncertainty handling.
    """
    def __init__(self, ppe_model_path: str = None):
        super().__init__(agent_name="ppe", model_version="VisionIQ-PPE-SpatialAssoc-v1.0")
        self.ppe_model_path = ppe_model_path or settings.MODEL_PPE_PATH

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

        image = cv2.imread(image_path)
        if image is None:
            return AgentOutput(
                agent_name=self.agent_name,
                status=ExecutionStatus.FAILED,
                error_details="Could not decode image with OpenCV",
                provenance={"agent": self.agent_name, "version": self.model_version}
            )

        img_h, img_w = image.shape[:2]

        # Extract people detected from upstream context
        object_result = agent_input.context_data.get("object_result", {})
        persons_detected = []
        if object_result and "objects" in object_result:
            for obj in object_result["objects"]:
                if obj["label"].lower() in ["person", "worker"]:
                    persons_detected.append(obj)

        if not persons_detected:
            from backend.app.agents.agent_b_object import ObjectDetectionAgent
            obj_agent = ObjectDetectionAgent()
            obj_res = obj_agent.analyze(image_path)
            for obj in obj_res.get("objects", []):
                if obj["label"].lower() in ["person", "worker"]:
                    persons_detected.append(obj)

        # Deterministic sorting (left-to-right, then top-to-bottom) for stable worker IDs
        persons_detected.sort(key=lambda p: (p["bbox"][0], p["bbox"][1]))

        person_findings: List[Dict[str, Any]] = []
        person_evaluations: List[Dict[str, Any]] = []

        for idx, person in enumerate(persons_detected):
            px1, py1, px2, py2 = person["bbox"]
            pw = px2 - px1
            ph = py2 - py1

            if pw < 15 or ph < 25:
                continue

            stable_worker_id = f"Worker #{idx+1} (W-{idx+1:02d})"

            # Check mutual occlusion against all other persons
            is_occluded_by_other = False
            occluding_worker = None
            for jdx, other in enumerate(persons_detected):
                if idx != jdx:
                    iou = compute_box_iou(person["bbox"], other["bbox"])
                    if iou > 0.20:
                        is_occluded_by_other = True
                        occluding_worker = f"Worker #{jdx+1}"
                        break

            # Anatomical bounding: head (top 28%), torso (25% to 65%)
            head_box = [px1, py1, px2, py1 + 0.28 * ph]
            torso_box = [px1, py1 + 0.25 * ph, px2, py1 + 0.65 * ph]

            head_crop = image[int(max(0, head_box[1])):int(min(img_h, head_box[3])), 
                              int(max(0, head_box[0])):int(min(img_w, head_box[2]))]
            torso_crop = image[int(max(0, torso_box[1])):int(min(img_h, torso_box[3])), 
                                int(max(0, torso_box[0])):int(min(img_w, torso_box[2]))]

            has_helmet = False
            helmet_confidence = 0.0
            helmet_evidence = "No helmet signature detected"

            if head_crop.size > 0:
                hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                yellow_mask = cv2.inRange(hsv_head, np.array([15, 70, 70]), np.array([35, 255, 255]))
                white_mask = cv2.inRange(hsv_head, np.array([0, 0, 180]), np.array([180, 50, 255]))
                bright_fraction = (cv2.countNonZero(yellow_mask) + cv2.countNonZero(white_mask)) / (head_crop.shape[0] * head_crop.shape[1])

                if bright_fraction > 0.18:
                    has_helmet = True
                    helmet_confidence = min(0.95, 0.60 + bright_fraction)
                    helmet_evidence = f"High-luminance safety headwear signature detected ({bright_fraction*100:.1f}% coverage)"

            has_vest = False
            vest_confidence = 0.0
            vest_evidence = "No hi-vis vest detected"

            if torso_crop.size > 0:
                hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
                hi_vis_mask = cv2.inRange(hsv_torso, np.array([15, 90, 90]), np.array([45, 255, 255]))
                neon_orange_mask = cv2.inRange(hsv_torso, np.array([5, 120, 120]), np.array([18, 255, 255]))
                vest_fraction = (cv2.countNonZero(hi_vis_mask) + cv2.countNonZero(neon_orange_mask)) / (torso_crop.shape[0] * torso_crop.shape[1])

                if vest_fraction > 0.15:
                    has_vest = True
                    vest_confidence = min(0.95, 0.65 + vest_fraction)
                    vest_evidence = f"High-visibility fluorescent vest signature detected ({vest_fraction*100:.1f}% coverage)"

            person_uid = f"F-PPE-{idx+1:02d}"
            is_distant = (ph < 75) or (pw < 35)

            if is_occluded_by_other:
                compliance_status = "UNCERTAIN"
                uncertainty = 0.70
                finding_label = f"{stable_worker_id}: Uncertain PPE Compliance (Occluded by {occluding_worker})"
                human_review = True
                observed = f"{stable_worker_id}: Partially occluded by overlapping bounding box of {occluding_worker}. Cannot determine helmet or vest presence without ambiguity."
                recommendation = "Reposition camera angle or review adjacent frames to resolve occlusion."
                score_desc = "Uncertain compliance due to person-person occlusion. Human verification required."
            elif is_distant:
                compliance_status = "UNCERTAIN"
                uncertainty = 0.65
                finding_label = f"{stable_worker_id}: Uncertain PPE Compliance (Low Resolution / Distance)"
                human_review = True
                observed = f"{stable_worker_id}: Person height is {ph:.0f}px; insufficient spatial resolution for definitive helmet or vest verification."
                recommendation = "Use optical zoom or dispatch safety floor supervisor for verification."
                score_desc = "Uncertain compliance due to low pixel resolution. Automated penalty withheld."
            elif has_helmet and has_vest:
                compliance_status = "COMPLIANT"
                uncertainty = 0.10
                finding_label = f"{stable_worker_id}: Compliant Worker (Helmet & Vest Present)"
                human_review = False
                observed = f"{stable_worker_id}: {helmet_evidence}; {vest_evidence}."
                recommendation = "Continue standard periodic safety monitoring."
                score_desc = f"Compliant: Both protective hardhat ({helmet_confidence*100:.0f}%) and fluorescent vest ({vest_confidence*100:.0f}%) confirmed in anatomical zones."
            elif not has_helmet and not has_vest:
                compliance_status = "NON_COMPLIANT_CRITICAL"
                uncertainty = 0.15
                finding_label = f"{stable_worker_id}: PPE Violation: Missing Hard Hat & Vest"
                human_review = True
                observed = f"{stable_worker_id}: Head and torso regions unobstructed; neither hardhat nor safety vest detected in anatomical zones."
                recommendation = "Halt entry into active operational area until full PPE is equipped per OSHA 1926.100."
                score_desc = "Critical Non-Compliance: Missing both hardhat and high-visibility vest. Immediate intervention required."
            elif not has_helmet:
                compliance_status = "NON_COMPLIANT_MODERATE"
                uncertainty = 0.20
                finding_label = f"{stable_worker_id}: PPE Warning: Missing Hard Hat"
                human_review = True
                observed = f"{stable_worker_id}: High-visibility vest present, but head region lacks required protective headgear."
                recommendation = "Issue hard hat immediately before worker approaches overhead hazard zone."
                score_desc = "Moderate Non-Compliance: Vest present, but missing protective hardhat in head zone."
            else:
                compliance_status = "NON_COMPLIANT_MODERATE"
                uncertainty = 0.20
                finding_label = f"{stable_worker_id}: PPE Warning: Missing High-Visibility Vest"
                human_review = True
                observed = f"{stable_worker_id}: Hard hat detected, but high-visibility safety vest is missing."
                recommendation = "Provide ANSI Class 2 high-visibility vest for site transit."
                score_desc = "Moderate Non-Compliance: Hardhat present, but missing high-visibility safety vest."

            evaluation_entry = {
                "person_id": stable_worker_id,
                "bbox": [round(c, 1) for c in [px1, py1, px2, py2]],
                "compliance_status": compliance_status,
                "has_helmet": has_helmet,
                "helmet_confidence": round(helmet_confidence, 4),
                "has_vest": has_vest,
                "vest_confidence": round(vest_confidence, 4),
                "uncertainty": uncertainty,
                "is_occluded": is_occluded_by_other
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
                "alternative_explanations": "Worker may wear non-standard color gear, or overhead structures partially obscure the view." if compliance_status != "COMPLIANT" else None,
                "recommendation": recommendation,
                "verification_procedure": "On-site safety officer physical check against OSHA 1926.100 / 1926.201 compliance logs.",
                "human_review_required": human_review,
                "model_name": "VisionIQ-PPE-SpatialAssoc",
                "model_version": "v1.0.0",
                "score_interpretation": score_desc,
                "person_id": stable_worker_id,
                "is_confirmed_defect": None
            })

        duration_ms = (time.time() - start_time) * 1000.0

        return AgentOutput(
            agent_name=self.agent_name,
            status=ExecutionStatus.SUCCESS,
            measured_runtime_ms=round(duration_ms, 2),
            findings=person_findings,
            data={
                "total_persons_evaluated": len(person_evaluations),
                "compliant_count": sum(1 for p in person_evaluations if p["compliance_status"] == "COMPLIANT"),
                "non_compliant_count": sum(1 for p in person_evaluations if "NON_COMPLIANT" in p["compliance_status"]),
                "uncertain_count": sum(1 for p in person_evaluations if p["compliance_status"] == "UNCERTAIN"),
                "persons": person_evaluations
            },
            provenance={
                "method": "anatomical_spatial_association",
                "rules": "OSHA 1926.100 / 1926.201",
                "timestamp": time.time()
            }
        )

    def analyze(self, image_path: str, object_result: Dict[str, Any] = None) -> Dict[str, Any]:
        agent_input = AgentInput(
            session_id="adhoc",
            user_query="ppe compliance check",
            image_path=image_path,
            context_data={"object_result": object_result} if object_result else {}
        )
        output = self.execute(agent_input)
        return {
            "total_persons_evaluated": output.data.get("total_persons_evaluated", 0),
            "compliant_count": output.data.get("compliant_count", 0),
            "non_compliant_count": output.data.get("non_compliant_count", 0),
            "uncertain_count": output.data.get("uncertain_count", 0),
            "persons": output.data.get("persons", []),
            "findings": output.findings,
            "model_version": self.model_version,
            "latency_ms": output.measured_runtime_ms,
            "status": output.status.value,
            "error": output.error_details
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
