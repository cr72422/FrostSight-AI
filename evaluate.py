"""
Optional evaluation script.

Use this only after preparing a correctly labeled dataset in YOLO format
and editing data.yaml to match that dataset's actual paths and class names.
A pretrained general-purpose model may not support your dataset's classes.
For package-specific metrics, fine-tune a compatible detector first.
"""
from pathlib import Path

from ultralytics import YOLO

DATA_CONFIG = Path("data.yaml")
MODEL_PATH = "yolo26n.pt"


def main():
    if not DATA_CONFIG.exists():
        raise SystemExit(
            "data.yaml not found. Read DATASET_GUIDE.md before running "
            "this optional evaluation script."
        )

    model = YOLO(MODEL_PATH)
    metrics = model.val(data=str(DATA_CONFIG), split="val", imgsz=640, batch=1)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Mean precision: {metrics.box.mp:.4f}")
    print(f"Mean recall: {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
