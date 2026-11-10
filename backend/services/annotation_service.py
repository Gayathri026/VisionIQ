import os
import cv2
import numpy as np
from typing import Dict, Any, List
from backend.config import settings

def draw_styled_box(
    image: np.ndarray,
    box: List[float],
    label: str,
    color: tuple = (43, 108, 176),
    thickness: int = 2
) -> np.ndarray:
    """Draws a clean, industrial-styled bounding box with background badge for text."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Box rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Badge background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thick = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    
    badge_y1 = max(0, y1 - text_h - 6)
    badge_y2 = y1
    badge_x2 = min(w, x1 + text_w + 8)
    
    cv2.rectangle(image, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(
        image, label, (x1 + 4, y1 - 4),
        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
    )
    return image

class AnnotationService:
    """
    Renders multi-agent visual evidence (objects, PPE compliance, defects, OCR)
    directly onto images for provenance verification and PDF reports.
    """
    def __init__(self):
        os.makedirs(settings.ANNOTATIONS_DIR, exist_ok=True)

    def annotate_image(
        self,
        image_path: str,
        findings: List[Dict[str, Any]],
        output_filename: str = None
    ) -> str:
        if not os.path.exists(image_path):
            return ""

        image = cv2.imread(image_path)
        if image is None:
            return ""

        # Colors in BGR
        COLOR_OBJECT = (176, 108, 43)    # Steel Blue
        COLOR_PPE_OK = (40, 167, 69)     # Forest Green
        COLOR_PPE_WARN = (0, 140, 255)   # Amber / Orange
        COLOR_PPE_BAD = (30, 30, 220)    # Crimson Red
        COLOR_DEFECT = (0, 0, 210)       # Dark Red
        COLOR_OCR = (180, 50, 150)       # Purple

        for f in findings:
            bbox = f.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            agent_source = f.get("agent_source", "")
            label = f.get("label", "")
            uid = f.get("finding_uid", "")

            # Select styling based on finding type
            if agent_source == "ppe":
                if "Compliant" in label:
                    color = COLOR_PPE_OK
                    badge = f"{uid}: Compliant"
                elif "Uncertain" in label:
                    color = COLOR_PPE_WARN
                    badge = f"{uid}: Uncertain"
                else:
                    color = COLOR_PPE_BAD
                    badge = f"{uid}: PPE Violation"
            elif agent_source == "defect":
                color = COLOR_DEFECT
                anomaly = f.get("anomaly_score")
                anom_str = f"({anomaly:.2f})" if anomaly else ""
                badge = f"{uid}: {label} {anom_str}"
            elif agent_source == "ocr":
                color = COLOR_OCR
                badge = f"{uid}: OCR"
            else:
                color = COLOR_OBJECT
                conf = f.get("detection_confidence")
                conf_str = f" {conf*100:.0f}%" if conf else ""
                badge = f"{uid}: {label}{conf_str}"

            image = draw_styled_box(image, bbox, badge, color=color, thickness=2)

        if not output_filename:
            base_name = os.path.basename(image_path)
            output_filename = f"annotated_{base_name}"
            
        annotated_path = os.path.join(settings.ANNOTATIONS_DIR, output_filename)
        cv2.imwrite(annotated_path, image)
        return annotated_path

annotation_service = AnnotationService()
