import os
import torch
import numpy as np
from ultralytics import YOLO
from backend.config import settings

def train_ppe_model(
    epochs: int = 5,
    imgsz: int = 640,
    batch_size: int = 4,
    device: str = None
):
    """
    Reproducible YOLO training script for Construction PPE compliance.
    Saves weights to backend/models/yolo_ppe.pt upon completion.
    """
    # 1. Fix random seeds for scientific reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device_str = device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Initializing PPE Model Training on device: {device_str}")

    data_yaml = "datasets/ppe_safety.yaml"
    if not os.path.exists(data_yaml):
        from backend.training.prepare_datasets import generate_sample_verification_splits
        generate_sample_verification_splits()

    # Load baseline model weights
    base_model = settings.MODEL_YOLO_PATH if os.path.exists(settings.MODEL_YOLO_PATH) else "yolo11n.pt"
    model = YOLO(base_model)

    # Execute training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device_str,
        project="storage/training_logs",
        name="ppe_training_run",
        exist_ok=True,
        seed=42,
        verbose=True
    )

    # Save target checkpoint
    best_weights = os.path.join("storage/training_logs", "ppe_training_run", "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs(os.path.dirname(settings.MODEL_PPE_PATH), exist_ok=True)
        import shutil
        shutil.copyfile(best_weights, settings.MODEL_PPE_PATH)
        print(f"Saved fine-tuned PPE model to: {settings.MODEL_PPE_PATH}")

    return results

if __name__ == "__main__":
    train_ppe_model(epochs=1, batch_size=2)
