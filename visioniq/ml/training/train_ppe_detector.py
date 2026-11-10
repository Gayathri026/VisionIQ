"""
VisionIQ PPE Detector Fine-Tuning Pipeline
Fine-tunes Ultralytics YOLO baseline on Construction Site Safety Bounding Boxes.
Enforces fixed random seeds, parameter tracking, and checkpoint persistence.
"""

import os
import sys
import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add monorepo to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.app.core.config import settings

def train_ppe_detector(
    epochs: int = 1,
    batch_size: int = 2,
    imgsz: int = 640,
    lr: float = 0.001,
    device: str = None,
    seed: int = 42
) -> str:
    """
    Fine-tunes YOLO model on construction site safety dataset.
    Returns path to exported fine-tuned checkpoint.
    """
    # 1. Enforce strict scientific reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")
    print(f"[VisionIQ PPE Trainer] Target device: {target_device} | Seed: {seed} | Epochs: {epochs}")

    data_yaml = PROJECT_ROOT / "ml" / "datasets" / "ppe_safety.yaml"
    if not data_yaml.exists():
        from ml.datasets.prepare_datasets import seed_sample_verification_splits
        seed_sample_verification_splits()

    base_weights = PROJECT_ROOT / "ml" / "model_registry" / "yolo11n.pt"
    if not base_weights.exists():
        base_weights = Path("yolo11n.pt")

    model = YOLO(str(base_weights))

    project_dir = PROJECT_ROOT / "storage" / "training_logs"
    os.makedirs(project_dir, exist_ok=True)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr,
        device=target_device,
        project=str(project_dir),
        name="ppe_training_run",
        exist_ok=True,
        seed=seed,
        verbose=True
    )

    # Export best / last checkpoint to model registry
    target_weights = PROJECT_ROOT / "ml" / "model_registry" / "yolo_ppe.pt"
    os.makedirs(target_weights.parent, exist_ok=True)

    possible_checkpoints = [
        project_dir / "ppe_training_run" / "weights" / "best.pt",
        project_dir / "ppe_training_run" / "weights" / "last.pt"
    ]

    saved = False
    for ckpt in possible_checkpoints:
        if ckpt.exists():
            shutil.copyfile(str(ckpt), str(target_weights))
            print(f"[VisionIQ PPE Trainer] Checkpoint persisted to: {target_weights} ({ckpt.stat().st_size / 1e6:.2f} MB)")
            saved = True
            break

    if not saved:
        # Fallback copy base weights as baseline checkpoint
        shutil.copyfile(str(base_weights), str(target_weights))
        print(f"[VisionIQ PPE Trainer] Baseline model copied as checkpoint: {target_weights}")

    # Copy to backend app models directory as well
    backend_model_path = PROJECT_ROOT / "backend" / "app" / "models" / "yolo_ppe.pt"
    os.makedirs(backend_model_path.parent, exist_ok=True)
    shutil.copyfile(str(target_weights), str(backend_model_path))

    return str(target_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionIQ PPE Detector Trainer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppe_detector(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr=args.lr,
        seed=args.seed
    )
