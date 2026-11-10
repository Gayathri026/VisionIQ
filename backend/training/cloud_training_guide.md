# VisionIQ Cloud Training & High-Performance GPU Guide

This guide documents the reproducible training protocol for VisionIQ specialist models using cloud GPU resources (Google Colab / Kaggle Notebooks / AWS EC2).

---

## 1. Datasets Used

1. **Construction Site Safety Image Dataset (Roboflow / Kaggle)**:
   - **URL**: [https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
   - **Classes (10)**: `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`.
   - **Splits**: 70% Train, 20% Validation, 10% Test.

2. **NEU Surface Defect Database (Northeastern University)**:
   - **URL**: [https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
   - **Classes (6)**: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.
   - **Splits**: 1,800 grayscale images (300 per class), 200x200 resolution.

3. **MVTec AD (Industrial Anomaly Detection)**:
   - **URL**: [https://www.mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
   - **Application**: Anomaly score distribution calibration.

---

## 2. 1-Click Google Colab / Kaggle Execution

In a Colab notebook with GPU runtime enabled (T4 or V100):

```bash
# Clone or upload VisionIQ repository
git clone https://github.com/your-username/visioniq.git
cd visioniq

# Install pinned dependencies
pip install -r requirements.txt

# Run Dataset Manifest Generation
python -m backend.training.prepare_datasets

# Execute 50-epoch Fine-Tuning for PPE Detection
python -m backend.training.train_ppe_model --epochs 50 --batch 16 --device 0
```

---

## 3. Training Hyperparameters

- **Architecture**: YOLOv11 (Ultralytics)
- **Optimizer**: AdamW
- **Initial Learning Rate**: 0.001
- **Weight Decay**: 0.0005
- **Image Resolution**: 640x640
- **Random Seed**: 42 (fixed for scientific reproducibility)
- **Data Augmentation**: Mosaic (1.0), Mixup (0.15), HSV jitter (h=0.015, s=0.7, v=0.4).

---

## 4. Citation Requirements

When publishing or referencing these models, cite:
```bibtex
@article{sanyal2023construction,
  title={Construction Site Safety Dataset},
  author={Sanyal, Snehil and Roboflow},
  year={2023}
}

@article{song2013neu,
  title={A surface defect detection method based on saliency linear feature for hot-rolled steel strip},
  author={Song, K. and Yan, Y.},
  journal={Applied Surface Science},
  volume={285},
  pages={578--587},
  year={2013}
}
```
