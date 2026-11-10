"""
VisionIQ Surface Defect Classifier Training Pipeline
Trains a multi-class deep convolutional network for the 6 NEU surface defect classes:
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
Enforces fixed seeds, AdamW optimization, cross-entropy loss, and checkpoint export.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

class NEUSurfaceDataset(Dataset):
    """Loads steel surface defect patches organized in class folders."""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        for cls_idx, cls_name in enumerate(NEU_CLASSES):
            cls_folder = self.root_dir / cls_name
            if cls_folder.exists():
                for img_path in cls_folder.glob("*.*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append((str(img_path), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class IndustrialDefectNet(nn.Module):
    """
    Lightweight convolutional feature extractor and classifier
    tailored for $200 \times 200$ grayscale industrial steel strip defect inspection.
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

def train_defect_classifier(
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 0.001,
    device: str = None,
    seed: int = 42
) -> str:
    """Trains surface defect network and exports checkpoint to model registry."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[VisionIQ Defect Trainer] Target device: {target_device} | Seed: {seed} | Epochs: {epochs}")

    train_dir = PROJECT_ROOT / "ml" / "datasets" / "neu_surface_defect" / "train"
    val_dir = PROJECT_ROOT / "ml" / "datasets" / "neu_surface_defect" / "valid"

    if not train_dir.exists() or len(list(train_dir.rglob("*.*"))) == 0:
        from ml.datasets.prepare_datasets import seed_sample_verification_splits
        seed_sample_verification_splits()

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NEUSurfaceDataset(str(train_dir), transform=transform)
    val_dataset = NEUSurfaceDataset(str(val_dir), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = IndustrialDefectNet(num_classes=len(NEU_CLASSES)).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(target_device), labels.to(target_device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(target_device), labels.to(target_device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(1, val_total)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc*100:.1f}% - Val Acc: {val_acc*100:.1f}%")

    # Export checkpoint
    target_weights = PROJECT_ROOT / "ml" / "model_registry" / "neu_defect_classifier.pt"
    os.makedirs(target_weights.parent, exist_ok=True)
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "classes": NEU_CLASSES,
        "val_acc": val_acc,
        "architecture": "IndustrialDefectNet-v1.0"
    }, str(target_weights))

    print(f"[VisionIQ Defect Trainer] Checkpoint persisted to: {target_weights}")

    # Copy to backend app models directory
    backend_weights = PROJECT_ROOT / "backend" / "app" / "models" / "defect_model.pt"
    os.makedirs(backend_weights.parent, exist_ok=True)
    torch.save(model.state_dict(), str(backend_weights))

    return str(target_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionIQ Surface Defect Classifier Trainer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_defect_classifier(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed
    )
