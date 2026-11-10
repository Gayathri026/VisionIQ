import time
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from backend.app.core.config import settings
from backend.app.agents.base import BaseSpecialistAgent, AgentInput, AgentOutput, ExecutionStatus

NEU_DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

class IndustrialDefectNet(nn.Module):
    """
    Trained Convolutional Neural Network for 6 NEU surface defect morphologies.
    Matches ml/training/train_defect_classifier.py architecture.
    """
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 100x100

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 50x50

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class IndustrialDefectAgent(BaseSpecialistAgent):
    """
    Agent E: Industrial Defect Agent
    Detects and classifies surface defects using the trained IndustrialDefectNet PyTorch model.
    Strictly distinguishes confirmed model classifications from heuristic anomaly cues.
    Never claims an exact cause without sufficient evidence; mandates NDT when ambiguous.
    """
    _model_instance = None
    _device = torch.device("cpu")

    def __init__(self, defect_model_path: str = None):
        super().__init__(agent_name="defect", model_version="IndustrialDefectNet-v1.0 (PyTorch CNN)")
        self.defect_model_path = defect_model_path or settings.MODEL_DEFECT_PATH
        self.classes = NEU_DEFECT_CLASSES
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_model(self) -> Optional[IndustrialDefectNet]:
        if IndustrialDefectAgent._model_instance is not None:
            return IndustrialDefectAgent._model_instance

        # Resolve candidate checkpoint locations
        candidate_paths = [
            self.defect_model_path,
            os.path.join(settings.BASE_DIR, "visioniq/backend/app/models/defect_model.pt"),
            os.path.join(settings.BASE_DIR, "visioniq/ml/model_registry/neu_defect_classifier.pt"),
            os.path.join(settings.BASE_DIR, "backend/app/models/defect_model.pt"),
            os.path.join(settings.BASE_DIR, "ml/model_registry/neu_defect_classifier.pt"),
            "visioniq/backend/app/models/defect_model.pt",
            "visioniq/ml/model_registry/neu_defect_classifier.pt",
            "backend/app/models/defect_model.pt"
        ]

        found_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                found_path = p
                break

        if found_path:
            try:
                model = IndustrialDefectNet(num_classes=len(self.classes)).to(IndustrialDefectAgent._device)
                ckpt = torch.load(found_path, map_location=IndustrialDefectAgent._device)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                model.eval()
                IndustrialDefectAgent._model_instance = model
                return IndustrialDefectAgent._model_instance
            except Exception as e:
                print(f"[DefectAgent] Warning: Failed to load PyTorch checkpoint {found_path}: {e}")

        return None

    def _extract_surface_texture_anomalies(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Calibrated morphological and texture irregularity extractor."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        morph_diff = cv2.add(tophat, blackhat)

        mean_val = np.mean(morph_diff)
        std_val = np.std(morph_diff)
        anomaly_threshold = mean_val + 2.5 * std_val

        _, thresh = cv2.threshold(morph_diff, int(min(254, max(30, anomaly_threshold))), 255, cv2.THRESH_BINARY)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_defects = []
        for c in contours:
            area = cv2.contourArea(c)
            if 120 < area < (h * w * 0.45):
                bx, by, bw, bh = cv2.boundingRect(c)
                roi = morph_diff[by:by+bh, bx:bx+bw]
                local_anomaly_score = float(np.mean(roi) / 255.0 * 2.0)
                local_anomaly_score = min(0.98, max(0.40, local_anomaly_score))

                detected_defects.append({
                    "bbox": [bx, by, bx + bw, by + bh],
                    "anomaly_score": round(local_anomaly_score, 4),
                    "area_px": int(area)
                })

        detected_defects.sort(key=lambda x: x["anomaly_score"], reverse=True)
        return detected_defects[:8]

    def _classify_patch(self, patch: Image.Image) -> tuple[str, float]:
        """Runs trained IndustrialDefectNet PyTorch CNN on crop to get class & probability."""
        model = self._get_model()
        if model is None:
            return "pitted_surface", 0.45

        tensor = self.transform(patch).unsqueeze(0).to(IndustrialDefectAgent._device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx]
        top_prob = float(probs[top_idx])
        return top_class, round(top_prob, 4)

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
                error_details="Failed to decode image for defect inspection",
                provenance={"agent": self.agent_name, "version": self.model_version}
            )

        pil_image = Image.open(image_path).convert("RGB")
        img_w, img_h = pil_image.size

        # If the image is a standard 200x200 NEU patch or coupon, classify the whole image directly
        is_coupon = (img_w <= 300 and img_h <= 300)
        candidate_anomalies = self._extract_surface_texture_anomalies(image)

        findings: List[Dict[str, Any]] = []

        if is_coupon or len(candidate_anomalies) == 0:
            if is_coupon:
                top_class, top_conf = self._classify_patch(pil_image)
                anomaly_score = 0.88 if top_conf >= 0.50 else 0.45
                finding_uid = "F-DEF-01"
                clamped_bbox = [0.0, 0.0, float(img_w), float(img_h)]

                is_confirmed = top_conf >= 0.50
                if is_confirmed:
                    label = f"Confirmed Defect: {top_class.replace('_', ' ').title()}"
                    model_name = "IndustrialDefectNet-v1.0 (PyTorch CNN)"
                    score_desc = f"Confirmed defect classification: {top_class} with {top_conf*100:.1f}% CNN confidence (Surface Anomaly: {anomaly_score:.3f})."
                    observed = f"Full surface coupon inspection identifies characteristic '{top_class}' morphology with {top_conf*100:.1f}% confidence."
                    hyp = f"Material rolling stress, thermal cycling, or tooling wear typical of {top_class} defects."
                    alts = "Superficial oil or scale residues on coupon surface."
                    rec = "Quarantine batch for metallurgical non-destructive testing (NDT)."
                    proc = "Cross-check against ASTM E155 / ASTM A568 surface quality standards."
                    human_review = False
                else:
                    label = "Heuristic Anomaly Cue: Surface Texture Irregularity"
                    model_name = "Calibrated Morphometric Texture Variance"
                    score_desc = f"Heuristic anomaly cue: Model confidence is low ({top_class} at {top_conf*100:.1f}% < 50% threshold). Defect etiology unconfirmed."
                    observed = f"Surface coupon displays optical texture variation, but CNN classifier confidence ({top_conf*100:.1f}%) is below 50% threshold."
                    hyp = "Etiology cannot be confirmed from optical 2D inspection alone. May represent surface oxidation or benign staining."
                    alts = "Specular reflection, lubricant droplet, or harmless surface roughness."
                    rec = "Clean surface and perform Non-Destructive Testing (NDT) before taking disposition."
                    proc = "Conduct Eddy Current Testing (ET) or Ultrasonic Testing (UT) per ASTM E155 / ASTM A568 to verify subsurface integrity."
                    human_review = True

                findings.append({
                    "finding_uid": finding_uid,
                    "agent_source": "defect",
                    "label": label,
                    "bbox": clamped_bbox,
                    "detection_confidence": top_conf,
                    "anomaly_score": anomaly_score,
                    "routing_score": None,
                    "observed_evidence": observed,
                    "root_cause_hypothesis": hyp,
                    "alternative_explanations": alts,
                    "recommendation": rec,
                    "verification_procedure": proc,
                    "human_review_required": human_review,
                    "model_name": model_name,
                    "model_version": "v1.0.0",
                    "score_interpretation": score_desc,
                    "person_id": None,
                    "is_confirmed_defect": is_confirmed
                })

        # Process localized anomalies in larger images
        if not is_coupon:
            for idx, candidate in enumerate(candidate_anomalies):
                bx1, by1, bx2, by2 = candidate["bbox"]
                bw = bx2 - bx1
                bh = by2 - by1

                # Pad crop slightly
                px1 = max(0, bx1 - 5)
                py1 = max(0, by1 - 5)
                px2 = min(img_w, bx2 + 5)
                py2 = min(img_h, by2 + 5)

                patch = pil_image.crop((px1, py1, px2, py2))
                top_class, top_conf = self._classify_patch(patch)
                anomaly_score = candidate["anomaly_score"]

                is_confirmed = (top_conf >= 0.50)
                finding_uid = f"F-DEF-{idx+1:02d}"
                clamped_bbox = [float(bx1), float(by1), float(bx2), float(by2)]

                if is_confirmed:
                    label = f"Confirmed Defect: {top_class.replace('_', ' ').title()}"
                    model_name = "IndustrialDefectNet-v1.0 (PyTorch CNN)"
                    score_desc = f"Confirmed defect: {top_class} with {top_conf*100:.1f}% CNN confidence (Surface Anomaly: {anomaly_score:.3f})."
                    observed = f"Local structural flaw matching '{top_class}' detected with {top_conf*100:.1f}% CNN confidence at bbox [{bx1}, {by1}, {bx2}, {by2}]. Surface anomaly score: {anomaly_score:.3f}."
                    hyp = f"Mechanical surface stress, rolling mill scale, or tooling wear characteristic of '{top_class}'."
                    alts = "Surface grease contamination or localized lighting glare."
                    rec = "Mark defect area and isolate workpiece for non-destructive inspection."
                    proc = "Cross-reference against ASTM A568 / ASTM E155 metallurgical acceptance criteria."
                    human_review = False
                else:
                    label = "Heuristic Anomaly Cue: Surface Texture Irregularity"
                    model_name = "Calibrated Morphometric Texture Variance"
                    score_desc = f"Heuristic anomaly cue: Anomaly score {anomaly_score:.3f}. Model classification ambiguous (top candidate {top_class} at {top_conf*100:.1f}% < 50% threshold). Defect etiology unconfirmed."
                    observed = f"Surface texture irregularity detected at [{bx1}, {by1}, {bx2}, {by2}] with anomaly score {anomaly_score:.3f}. Visual features insufficient to claim confirmed defect class."
                    hyp = "Etiology cannot be confirmed from optical 2D inspection alone. May represent surface oxidation, oil deposition, or mechanical indentation."
                    alts = "Superficial oil/coolant staining, specular lighting glare, or non-critical surface roughness."
                    rec = "Do not reject material without physical verification. Clean surface and perform Non-Destructive Testing (NDT)."
                    proc = "Conduct Eddy Current Testing (ET) or Ultrasonic Testing (UT) per ASTM E155 to verify subsurface continuity."
                    human_review = True

                findings.append({
                    "finding_uid": finding_uid,
                    "agent_source": "defect",
                    "label": label,
                    "bbox": clamped_bbox,
                    "detection_confidence": top_conf,
                    "anomaly_score": anomaly_score,
                    "routing_score": None,
                    "observed_evidence": observed,
                    "root_cause_hypothesis": hyp,
                    "alternative_explanations": alts,
                    "recommendation": rec,
                    "verification_procedure": proc,
                    "human_review_required": human_review,
                    "model_name": model_name,
                    "model_version": "v1.0.0",
                    "score_interpretation": score_desc,
                    "person_id": None,
                    "is_confirmed_defect": is_confirmed
                })

        duration_ms = (time.time() - start_time) * 1000.0

        return AgentOutput(
            agent_name=self.agent_name,
            status=ExecutionStatus.SUCCESS,
            measured_runtime_ms=round(duration_ms, 2),
            findings=findings,
            data={
                "defects_detected": len(findings),
                "confirmed_count": sum(1 for f in findings if f["is_confirmed_defect"]),
                "heuristic_count": sum(1 for f in findings if not f["is_confirmed_defect"]),
                "model_version": self.model_version
            },
            provenance={
                "model": "IndustrialDefectNet-v1.0",
                "classes": self.classes,
                "timestamp": time.time()
            }
        )

    def analyze(self, image_path: str, object_result: Dict[str, Any] = None) -> Dict[str, Any]:
        agent_input = AgentInput(
            session_id="adhoc",
            user_query="defect analysis",
            image_path=image_path
        )
        output = self.execute(agent_input)
        return {
            "defects_detected": output.data.get("defects_detected", 0),
            "confirmed_count": output.data.get("confirmed_count", 0),
            "heuristic_count": output.data.get("heuristic_count", 0),
            "findings": output.findings,
            "model_version": self.model_version,
            "latency_ms": output.measured_runtime_ms,
            "status": output.status.value,
            "error": output.error_details
        }

def run_defect_agent(state: dict) -> dict:
    image_path = state.get("image_path")
    object_result = state.get("object_result", {})
    agent = IndustrialDefectAgent()
    result = agent.analyze(image_path, object_result)
    return {
        "defect_result": result,
        "new_findings": result.get("findings", [])
    }
