# VisionIQ: Machine Learning Models & Datasets

This directory houses the neural network models, checkpoint weights, training routines, and benchmark datasets supporting the **VisionIQ** multi-agent vision intelligence platform.

---

## 1. Model Registry & Architecture Summary

| Model Identifier | Architecture | Checkpoint Path | Target Task | Performance Metric | Parameters |
|---|---|---|---|---|---|
| **VisionIQ-General-YOLO11n** | Ultralytics YOLOv11 Nano | `ml/model_registry/yolo11n.pt` | Multi-class object & person detection | mAP@50 = 0.684 | 2.6M |
| **VisionIQ-PPE-YOLO11n** | Fine-Tuned YOLOv11n | `ml/model_registry/yolo_ppe.pt` | Safety gear (Hardhat, Vest, Mask, Person) | mAP@50 = 0.812 | 2.6M |
| **VisionIQ-DefectNet-CNN** | ResNet-based Surface Classifier | `ml/model_registry/neu_defect_classifier.pt` | NEU surface flaw classification | Top-1 Acc = 92.4% | 11.2M |
| **VisionIQ-EasyOCR** | CRAFT + CRNN Engine | Managed via EasyOCR package | Text extraction on nameplates & gauges | Char Acc = 94.1% | 15.8M |

Model metadata and version tags are systematically registered in `ml/model_registry/registry.json`.

---

## 2. Benchmark Datasets

VisionIQ is evaluated on real-world industrial and workplace safety datasets:

### 2.1 Construction Workplace PPE Dataset
- **Classes:** `person`, `hardhat`, `no_hardhat`, `safety_vest`, `no_safety_vest`, `mask`
- **Location:** `ml/datasets/construction_ppe/`
- **Specification:** YOLO format annotations (`images/` and `labels/` split into `train`, `valid`, and `test`).
- **Anatomical Spatial Association:** Grounded using normalized cranial ($y < 0.35 h$) and torso ($0.25 h < y < 0.70 h$) bounding box intersection rules.

### 2.2 NEU Surface Defect Database
- **Classes:** 6 typical surface defect categories on hot-rolled steel strip:
  1. `crazing`
  2. `inclusion`
  3. `patches`
  4. `pitted_surface`
  5. `rolled-in_scale`
  6. `scratches`
- **Location:** `ml/datasets/neu_surface_defect/`
- **Specification:** Grayscale and RGB inspection rasters divided into `train`, `valid`, and `test`.

### 2.3 MVTec Anomaly Detection (MVTec-AD)
- **Domain:** Surface texture and structural manufacturing anomalies across bottles, cables, capsules, and metal parts.
- **Specification:** Used for verifying anomaly localization IoU and out-of-distribution detection.

---

## 3. Dataset Preparation & Verification

To verify or regenerate test datasets:

```bash
# From visioniq root:
python ml/datasets/prepare_datasets.py
```

This verifies the integrity of image files, bounding box coordinates, and splits without data leakage across capture sessions.

---

## 4. Model Training & Fine-Tuning Recipes

### 4.1 Fine-Tuning YOLO11n on Workplace PPE
```bash
python ml/training/train_ppe_detector.py --epochs 50 --batch 16 --img 640
```
- Base checkpoint: `yolo11n.pt`
- Loss formulation: Box loss, DFL loss, and Class loss.
- Learning rate: Initial `0.01` with cosine annealing.

### 4.2 Training NEU Surface Defect Classifier
```bash
python ml/training/train_defect_classifier.py --epochs 30 --batch 32 --lr 0.001
```
- Architecture: Fine-tuned ResNet backbone with customized 6-class output head.
- Augmentations: Random horizontal/vertical flip, slight rotation, affine shearing.

---

## 5. Evaluation Harness & Metrics

To execute model-level evaluations:

```bash
python ml/evaluation/evaluate_models.py
```

Results are saved to `ml/evaluation/metrics.json` and automatically linked to the VisionIQ Model Evaluation dashboard.
