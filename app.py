import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

st.set_page_config(
    page_title="YOLO License Plate Detection",
    page_icon="🚘",
    layout="centered"
)
st.title("🚘 License Plate Detection using YOLO")

with st.sidebar:
    st.header("Model Configuration")
    model_name = st.selectbox(
        "Select YOLO Model:",
        ("YOLOv9t", "YOLOv10n", "YOLOv11n")
    )
    confidence_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

MODEL_PATHS = {
    "YOLOv9t": "yolov9.pt",
    "YOLOv10n": "yolov10.pt",
    "YOLOv11n": "yolov11.pt"
}
selected_model_path = MODEL_PATHS[model_name]


@st.cache_resource
def load_yolo_model(path):
    """Loads a YOLO model from the given path."""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(
            f"Failed to load model from {path}. Please ensure the file exists and is correct. Error: {e}")
        return None


# Model
model = load_yolo_model(selected_model_path)

# File Upload
uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

# Detection
if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("🖼️ Original Image")
    st.image(image_np, use_container_width=True)

    results = model.predict(image_np, conf=confidence_threshold)

    result_image = image_np.copy()
    detections_count = 0

    st.subheader("📊 Detection Results")

    for result in results:
        boxes = result.boxes
        detections_count += len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = model.names[int(box.cls[0])]

            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(result_image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    st.image(result_image, use_container_width=True)

    if detections_count == 0:
        st.warning("No license plate detected in this image.", icon="⚠️")
    else:
        st.success(
            f"Successfully detected **{detections_count}** license plate(s).", icon="✅")

else:
    st.info("Please upload an image and select a model to start detection.", icon="👆")
