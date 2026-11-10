import os
import json
import yaml
import shutil
from typing import Dict, Any, List

DATASET_ROOT = "datasets"

PPE_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle"
]

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

def prepare_dataset_directories():
    """Sets up standardized YOLO dataset folder hierarchy."""
    for ds_name in ["construction_ppe", "neu_surface_defect"]:
        for split in ["train", "valid", "test"]:
            os.makedirs(os.path.join(DATASET_ROOT, ds_name, "images", split), exist_ok=True)
            os.makedirs(os.path.join(DATASET_ROOT, ds_name, "labels", split), exist_ok=True)

def generate_yaml_configs():
    """Generates standard Ultralytics YAML manifests."""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    
    # 1. Construction Site Safety YAML
    ppe_yaml_path = os.path.join(DATASET_ROOT, "ppe_safety.yaml")
    ppe_config = {
        "path": os.path.abspath(os.path.join(DATASET_ROOT, "construction_ppe")),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(PPE_CLASSES),
        "names": PPE_CLASSES
    }
    with open(ppe_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(ppe_config, f, default_flow_style=False)

    # 2. NEU Surface Defect YAML
    neu_yaml_path = os.path.join(DATASET_ROOT, "neu_defect.yaml")
    neu_config = {
        "path": os.path.abspath(os.path.join(DATASET_ROOT, "neu_surface_defect")),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(NEU_CLASSES),
        "names": NEU_CLASSES
    }
    with open(neu_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(neu_config, f, default_flow_style=False)

    print(f"Generated YAML configurations: {ppe_yaml_path}, {neu_yaml_path}")

def generate_sample_verification_splits():
    """
    Seeds initial validation images from local sample_data to allow
    immediate CPU training validation without requiring 10GB external downloads.
    """
    prepare_dataset_directories()
    generate_yaml_configs()
    
    sample_img = "sample_data/FACTORY_SCENE.png"
    if os.path.exists(sample_img):
        # Place into train and valid
        for split in ["train", "valid"]:
            dst = os.path.join(DATASET_ROOT, "construction_ppe", "images", split, "sample_factory.png")
            shutil.copyfile(sample_img, dst)
            # Create a mock YOLO label file: class x_center y_center width height
            lbl_dst = os.path.join(DATASET_ROOT, "construction_ppe", "labels", split, "sample_factory.txt")
            with open(lbl_dst, "w", encoding="utf-8") as f:
                # 5: Person, 0: Hardhat, 7: Safety Vest
                f.write("5 0.50 0.50 0.20 0.50\n")
                f.write("0 0.50 0.30 0.08 0.08\n")
                f.write("7 0.50 0.55 0.15 0.25\n")
        print("Sample verification splits created for CI/local verification.")

if __name__ == "__main__":
    generate_sample_verification_splits()
