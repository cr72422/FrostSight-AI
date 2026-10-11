# FrostSight — AI-Powered Object Detection

FrostSight is a beginner-friendly computer vision prototype for exploring object detection on warehouse-style images. It uses a pretrained Ultralytics YOLO model, OpenCV, and Streamlit.

## Features

- Upload JPG, JPEG, or PNG images
- Predict objects and display bounding boxes
- Adjust confidence threshold and inference image size
- Show class names, confidence scores, and object counts
- Display measured inference time
- Export detections to CSV

## Requirements

- Python 3.10 or 3.11 recommended
- Internet connection for installing dependencies and the first model-weight download
- A laptop that can run CPU inference; performance varies by hardware

## Quick start

### Windows PowerShell

```powershell
cd FrostSight
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks virtual-environment activation, run these instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
cd FrostSight
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal if your browser does not open automatically.

## Test the model from the terminal

Put a suitable image named `test.jpg` in the project folder, then run:

```bash
python test_model.py
```

## Dataset and evaluation

Read [DATASET_GUIDE.md](DATASET_GUIDE.md) before running the optional evaluation script. The included model is pretrained for general object classes; it is not guaranteed to identify shipping packages as a dedicated class. Meaningful precision, recall, and mAP require labeled ground-truth data and a model compatible with the dataset classes.

## Limitations

This is a learning prototype, not a validated warehouse system. Predictions can be wrong or miss objects. It does not estimate food freshness, temperature, or food safety.

## Project status

Starter project. Add your own test images, record actual results, and update this README with evidence from your run. Do not claim accuracy or training results that you have not measured.

## Licensing

Review the licenses of the model weights, datasets, and third-party libraries before using or redistributing them. Ultralytics licensing terms may differ by use case.
