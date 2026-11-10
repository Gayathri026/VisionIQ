import os
import cv2
import numpy as np

def create_benchmark_fixtures():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    # 1. High-contrast industrial safety warning sign
    sign_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Yellow warning header band
    cv2.rectangle(sign_img, (20, 20), (580, 110), (0, 215, 255), -1)
    cv2.rectangle(sign_img, (20, 20), (580, 380), (30, 30, 30), 4)
    cv2.putText(sign_img, "CAUTION", (160, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)
    cv2.putText(sign_img, "HARD HAT REQUIRED", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(sign_img, "IN THIS WORK AREA", (95, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(sign_img, "OSHA 1926.100 COMPLIANT", (110, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    cv2.putText(sign_img, "STATION #04 - MCET INDUSTRIAL LAB", (120, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
    sign_path = os.path.join(fixtures_dir, "ocr_safety_sign.png")
    cv2.imwrite(sign_path, sign_img)

    # 2. Equipment specification rating nameplate
    nameplate_img = np.ones((350, 550, 3), dtype=np.uint8) * 220
    cv2.rectangle(nameplate_img, (15, 15), (535, 335), (80, 80, 80), 3)
    cv2.rectangle(nameplate_img, (20, 20), (530, 75), (50, 50, 50), -1)
    cv2.putText(nameplate_img, "INDUSTRIAL SPECIFICATION PLATE", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(nameplate_img, "MODEL: CNC-4500X HEAVY LATHE", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    cv2.putText(nameplate_img, "SERIAL NO: SN-2026-9842A", (40, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    cv2.putText(nameplate_img, "OPERATING VOLTAGE: 480V 3-PHASE", (40, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    cv2.putText(nameplate_img, "RATED LOAD: 12500 KG", (40, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    cv2.putText(nameplate_img, "CERTIFIED: ISO 13849 / CE", (40, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 50, 0), 2)
    nameplate_path = os.path.join(fixtures_dir, "ocr_nameplate.png")
    cv2.imwrite(nameplate_path, nameplate_img)

    # 3. Hybrid scene: Equipment layout with warning and surface defect cue
    hybrid_img = np.ones((450, 600, 3), dtype=np.uint8) * 180
    cv2.rectangle(hybrid_img, (40, 60), (280, 380), (100, 100, 100), -1)
    cv2.rectangle(hybrid_img, (40, 60), (280, 380), (30, 30, 30), 3)
    cv2.putText(hybrid_img, "LATHE MACHINE", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    # Warning sign on machine
    cv2.rectangle(hybrid_img, (60, 130), (260, 210), (0, 200, 255), -1)
    cv2.putText(hybrid_img, "DANGER: HIGH VOLTAGE", (70, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # Simulated scratch defect on work surface
    for offset in range(-15, 15, 4):
        cv2.line(hybrid_img, (340 + offset, 120), (520 + offset, 340), (40, 40, 40), 2)
    cv2.putText(hybrid_img, "INSPECTION ZONE", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    hybrid_path = os.path.join(fixtures_dir, "hybrid_lathe_inspection.png")
    cv2.imwrite(hybrid_path, hybrid_img)

    print(f"Generated benchmark fixtures successfully in: {fixtures_dir}")
    return {
        "ocr_sign": sign_path,
        "ocr_nameplate": nameplate_path,
        "hybrid_lathe": hybrid_path
    }

if __name__ == "__main__":
    create_benchmark_fixtures()
