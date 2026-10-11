import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

st.set_page_config(page_title="FrostSight", page_icon="📦", layout="wide")

MODEL_NAME = "yolo26n.pt"

st.title("📦 FrostSight")
st.subheader("AI-powered object detection for warehouse-style images")
st.write(
    "Upload an image to detect objects, view bounding boxes, inspect confidence "
    "scores, and export predictions. This is a learning prototype, not a "
    "validated warehouse or food-safety system."
)


@st.cache_resource
def load_model():
    """Load the pretrained model once per Streamlit session."""
    return YOLO(MODEL_NAME)


with st.sidebar:
    st.header("Settings")
    confidence = st.slider(
        "Minimum confidence", min_value=0.10, max_value=0.90,
        value=0.25, step=0.05
    )
    image_size = st.select_slider(
        "Inference image size", options=[320, 480, 640, 800], value=640
    )
    st.caption("The first run may download model weights and requires internet.")

uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        st.image(image, caption="Input image", width="stretch")

        if st.button("Detect objects", type="primary"):
            try:
                with st.spinner("Loading model and running detection..."):
                    model = load_model()
                    start = time.perf_counter()
                    result = model.predict(
                        source=image_array,
                        conf=confidence,
                        imgsz=image_size,
                        verbose=False
                    )[0]
                    elapsed = time.perf_counter() - start

                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                rows = []
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    score = float(box.conf.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    rows.append({
                        "class_id": class_id,
                        "class_name": str(model.names[class_id]),
                        "confidence": round(score, 4),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })

                col1, col2, col3 = st.columns(3)
                col1.metric("Objects detected", len(rows))
                col2.metric("Inference time", f"{elapsed:.2f} s")
                col3.metric(
                    "Unique classes",
                    len({row["class_name"] for row in rows})
                )

                st.subheader("Annotated image")
                st.image(
                    annotated_rgb,
                    caption="Predicted bounding boxes",
                    width="stretch"
                )

                if rows:
                    df = pd.DataFrame(rows)
                    st.subheader("Detection details")
                    st.dataframe(df, use_container_width=True)
                    st.subheader("Count by predicted class")
                    st.bar_chart(df["class_name"].value_counts())
                    st.download_button(
                        "Download predictions as CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="frostsight_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(
                        "No objects passed this confidence threshold. "
                        "Try a different image or lower the threshold."
                    )

                st.caption(
                    "Inference time varies with hardware, image size, model "
                    "initialization and whether the model is running on CPU/GPU."
                )
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                st.info(
                    "Check your internet connection for the first model download, "
                    "then review the terminal output for more details."
                )

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not open this image: {exc}")

st.divider()
st.caption(
    "Prototype only. A general pretrained model may not recognize packages "
    "as a distinct class. Do not use its predictions for operational, "
    "temperature-monitoring or food-safety decisions."
)
