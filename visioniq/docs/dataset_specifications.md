# VisionIQ: Dataset Specifications & Academic Research Licensing

**Project Title:** VisionIQ: An Adaptive Multi-Agent Vision Intelligence Framework with Dynamic Orchestration and Cost-Aware Query Routing  
**Evaluation:** BE CSE Final Year Project (Course 23CSP701), Anna University  
**Target Publication:** IEEE Access  

---

## 1. Overview of Investigated Datasets

To ensure scientific rigor, reproducibility, and compliance with ethical and legal data standards, VisionIQ standardizes three specialized benchmark datasets for training and evaluating vision agents.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VisionIQ Dataset Matrix                         │
├─────────────────────────┬────────────────────────┬─────────────────────┤
│ 1. Construction Safety  │ 2. NEU Surface Defects │ 3. MVTec AD Anomaly │
│ Roboflow / Kaggle       │ Northeastern Univ (NEU)│ MVTec Research      │
│ Object Detection (BBox) │ Classification (6-Cls) │ Anomaly Localization│
│ License: CC BY 4.0      │ License: Academic Only │ License: CC BY-NC-SA│
└─────────────────────────┴────────────────────────┴─────────────────────┘
```

---

## 2. Dataset 1: Construction Site Safety Image Dataset

- **Source Provider:** Roboflow / Kaggle (Snehilsanyal)
- **Repository URL:** [https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  - *Permitted Use:* Free for academic research, modification, and commercial application provided proper attribution is given.
- **Task Formulation:** **Multi-Class Object Detection** (Bounding Boxes)
- **Annotation Format:** Ultralytics YOLO format:
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
  *(All coordinates normalized in range $[0.0, 1.0]$)*
- **Classes (10):**
  1. `Hardhat` (Class ID 0)
  2. `Mask` (Class ID 1)
  3. `NO-Hardhat` (Class ID 2)
  4. `NO-Mask` (Class ID 3)
  5. `NO-Safety Vest` (Class ID 4)
  6. `Person` (Class ID 5)
  7. `Safety Cone` (Class ID 6)
  8. `Safety Vest` (Class ID 7)
  9. `machinery` (Class ID 8)
  10. `vehicle` (Class ID 9)
- **Directory Layout:**
  ```
  construction_ppe/
  ├── images/
  │   ├── train/
  │   ├── valid/
  │   └── test/
  └── labels/
      ├── train/
      ├── valid/
      └── test/
  ```
- **Leakage Prevention Protocol:** Images extracted from continuous camera feeds are clustered by capture sequence timestamp prefix. All frames within a continuous 30-second window are assigned exclusively to either `train` or `test` to prevent near-identical visual overlap across splits.

---

## 3. Dataset 2: NEU Surface Defect Database

- **Source Provider:** Northeastern University, China (K. Song and Y. Yan, 2013)
- **Repository URL:** [https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- **License:** **Academic Research Use Permitted**
  - *Reference Publication:* K. Song and Y. Yan, "A surface defect detection method based on saliency linear feature for hot-rolled steel strip," *Applied Surface Science*, vol. 285, pp. 578–587, 2013.
- **Task Formulation:** **Surface Defect Image Classification** (6 Categories)
- **Crucial Methodological Rule:** *Do NOT fabricate synthetic bounding boxes for classification images.* NEU is evaluated as multi-class categorization of hot-rolled steel strip flaws.
- **Image Specifications:** 1,800 hot-rolled steel strip images (300 samples per class), $200 \times 200$ grayscale resolution.
- **Classes (6):**
  1. `crazing` (Class ID 0): Surface network of fine branching micro-cracks.
  2. `inclusion` (Class ID 1): Non-metallic impurities trapped during metallurgical casting.
  3. `patches` (Class ID 2): Irregular localized oxidation or scale deposition.
  4. `pitted_surface` (Class ID 3): Localized cavities resulting from rolling de-scaling.
  5. `rolled-in_scale` (Class ID 4): Mill scale pressed into the strip surface during rolling.
  6. `scratches` (Class ID 5): Linear mechanical abrasions from guide roller friction.
- **Directory Layout:**
  ```
  neu_surface_defect/
  ├── crazing/
  ├── inclusion/
  ├── patches/
  ├── pitted_surface/
  ├── rolled-in_scale/
  └── scratches/
  ```
- **Splits:** 70% Train (1,260 images), 15% Validation (270 images), 15% Test (270 images). Stratified across all 6 classes.

---

## 4. Dataset 3: MVTec Anomaly Detection Benchmark (MVTec AD)

- **Source Provider:** MVTec Software GmbH (P. Bergmann et al., CVPR 2019)
- **Repository URL:** [https://www.mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- **License:** **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**
  - *Permitted Use:* Free for non-commercial academic research and benchmarking. Commercial exploitation requires licensing.
- **Task Formulation:** **Unsupervised Industrial Anomaly Detection & Pixel-Level Localization**
- **Categories:** 15 industrial domains (5 textures: carpet, grid, leather, tile, wood; 10 objects: bottle, cable, capsule, hazelnut, metal nut, pill, screw, toothbrush, transistor, zipper).
- **Evaluation Metrics:**
  - Image-Level: Area Under Receiver Operating Characteristic Curve (**Image AUROC**).
  - Pixel-Level: Pixel-wise AUROC and Per-Region Overlap (**PRO score**).
- **VisionIQ Application:** Calibration of continuous surface texture gradients and morphological variance distributions without inferring unsubstantiated causal claims.
