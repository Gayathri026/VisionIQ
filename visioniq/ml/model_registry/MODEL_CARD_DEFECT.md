# Model Card: VisionIQ Industrial Surface Defect Classifier

## Model Details
- **Model Name:** VisionIQ Industrial Surface Defect Classifier (`visioniq-neu-defectnet-v1`)
- **Version:** 1.0.0
- **Model Date:** October 2026
- **Model Type:** Multi-Class Deep Convolutional Neural Network
- **Architecture:** `IndustrialDefectNet`
  - 3 Convolutional Blocks ($3\times 3$ kernel, stride 1, padding 1, ReLU, Batch Normalization)
  - 2 Max-Pooling Layers ($2\times 2$, stride 2) reducing spatial dimensions
  - Adaptive Average Pooling ($4 \times 4$ fixed feature lattice)
  - Dense Fully-Connected Classification Head ($2048 \to 128 \to 6$ with $p=0.30$ Dropout)
- **Parameters:** 298,438 (~298K)
- **Target Platform:** Industrial Embedded IPCs, Edge Gateways, and Cloud Quality Control Nodes
- **Framework:** PyTorch 2.4.1+
- **License:** Academic Research / Open Source

## Intended Use
- **Primary Use Case:** Automated visual defect categorization for hot-rolled steel strips in metallurgical production lines and post-rolling cold-finishing facilities.
- **Surface Defect Categories (6 classes):**
  1. `crazing` (Micro-crack networks formed during high-temperature thermal cycling)
  2. `inclusion` (Non-metallic exogenous oxide/slag inclusions entrapped during ingot casting)
  3. `patches` (Oxidation layer remnants and mechanical friction deposits)
  4. `pitted_surface` (Cavitations and localized oxidation pitting)
  5. `rolled-in_scale` (Iron oxide scale pressed into the strip matrix during rolling)
  6. `scratches` (Longitudinal physical abrasions caused by guide roller misalignment)
- **Out-of-Scope Use Cases:** General multi-label image tagging, biological microscopy, or non-ferrous casting inspections without domain re-training.

## Training Data & Methodology
- **Dataset:** Northeastern University (NEU) Surface Defect Database
- **Source Link:** `https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database`
- **Data Modality:** $200 \times 200$ grayscale hot-rolled steel strip surface samples.
- **Modality Compliance:** Treated strictly as whole-patch multi-class visual recognition. No synthetic bounding boxes or ungrounded coordinates are fabricated.
- **Split Distribution:** 70% Training / 15% Validation / 15% Held-Out Test grouped systematically to preserve balance across all 6 defect morphologies.
- **Optimization:** AdamW ($\eta = 0.001$, $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Loss Function:** Multi-Class Cross-Entropy Loss ($\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log \hat{y}_i$).
- **Reproducibility:** Random seed fixed to $42$ across Python runtime, NumPy, and PyTorch seeds.

## Evaluation Metrics & Expected Performance
- **Evaluation Criteria:**
  - Categorical Accuracy ($Acc$)
  - Precision, Recall, and Macro-Averaged $F_1$-Score
  - Full $6 \times 6$ Confusion Matrix identifying inter-class confusion patterns (e.g., between `rolled-in_scale` and `pitted_surface`).
- **Inference Latency:** $< 15\text{ ms}$ on standard multi-core CPU, $< 2\text{ ms}$ on CUDA tensor cores.

## Limitations & Industrial Quality Assurance
- **Grease and Lubricant Contamination:** Thick rolling oils and cooling emulsion puddles can create specular reflections that mask underlying fine crazing cracks.
- **NDT Integration:** High-criticality aerospace or pressure-vessel steels must combine VisionIQ optical classification with ultrasonic or eddy-current non-destructive testing (NDT).
