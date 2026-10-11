from pathlib import Path

from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"
IMAGE_PATH = Path("test.jpg")


def main():
    print("Loading pretrained YOLO model...")
    model = YOLO(MODEL_NAME)
    print(f"Model loaded. Number of known classes: {len(model.names)}")

    if not IMAGE_PATH.exists():
        print(
            "\nNo test.jpg found.\n"
            "Add a JPG image named test.jpg to this project folder, then rerun.\n"
            "You can also start the web app with: python -m streamlit run app.py"
        )
        return

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        save=True,
        verbose=False
    )
    result = results[0]
    print(f"Detections: {len(result.boxes)}")
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        print(f"- {model.names[class_id]}: confidence={confidence:.3f}")
    print("Annotated output is saved by Ultralytics under runs/detect/.")


if __name__ == "__main__":
    main()
