import streamlit as st
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
import torch
import tempfile
import cv2

st.set_page_config(page_title="Coral Detector", layout="centered")
st.title("Coral Detector App")
st.caption(
    "This application is developed as a Computer Vision final project using deep learning–based object detection."
)


# ===== SESSION STATE INIT =====
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# ===== DEVICE CHECK =====
USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else "cpu"
USE_HALF = USE_GPU
DEVICE_NAME = "GPU (CUDA)" if USE_GPU else "CPU"

st.warning(f"Running on device: {DEVICE_NAME}")


# ===== LOAD MODEL =====


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)
st.caption(
    "The confidence threshold controls how confident the model must be before a detection is shown. "
    "Increasing this value makes the model more selective."
)


# ===== FILE UPLOAD LOGIC =====
if st.session_state.uploaded_file is None:

    uploaded_file = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "mp4"]
    )

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()

else:
    uploaded_file = st.session_state.uploaded_file

    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("🗑 Remove uploaded file"):
        st.session_state.uploaded_file = None
        st.rerun()

# ================= IMAGE =================
if uploaded_file and uploaded_file.type.startswith("image"):
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    with st.spinner("Running detection..."):
        start_time = time.time()

        results = model(
            img,
            conf=conf,
            imgsz=640,
            device=DEVICE,
            half=USE_HALF,
            verbose=False
        )

        inference_time = time.time() - start_time

    annotated_img = results[0].plot()
    total = len(results[0].boxes)
    fps = 1 / inference_time if inference_time > 0 else 0

    st.image(annotated_img, caption="Detection Result",
             width='stretch')
    st.success(f"Total corals detected: {total}")

    col1, col2 = st.columns(2)
    col1.metric("FPS", f"{fps:.2f}")
    col2.metric("Inference Time (s)", f"{inference_time:.3f}")

# ================= VIDEO =================
elif uploaded_file and uploaded_file.type == "video/mp4":

    import cv2
    import tempfile
    import time

    # ===== SAVE UPLOADED VIDEO =====
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    # ===== OUTPUT VIDEO =====
    output_path = "output_inference.mp4"
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_video,
        (width, height)
    )

    # ===== UI ELEMENTS =====
    status_text = st.empty()
    progress_bar = st.progress(0)
    frame_counter = st.empty()

    status_text.info("Processing video... Please wait")

    processed = 0
    start_time = time.time()

    # ===== PROCESS VIDEO =====
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time_frame = time.time()

        results = model(
            frame,
            conf=conf,
            imgsz=640,
            device=DEVICE,
            half=USE_HALF,
            verbose=False
        )

        total_coral = len(results[0].boxes)
        fps = 1 / (time.time() - start_time_frame)

        annotated = results[0].plot()

        cv2.rectangle(annotated, (10, 10), (320, 90), (0, 0, 0), -1)

        cv2.putText(
            annotated,
            f"Total Coral: {total_coral}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        out.write(annotated)

        # ===== UPDATE UI =====
        processed += 1
        progress = int((processed / total_frames) * 100)

        progress_bar.progress(min(progress, 100))
        frame_counter.write(
            f"Processed frames: {processed}/{total_frames}"
        )

    cap.release()
    out.release()

    elapsed = time.time() - start_time

    # ===== FINAL UI =====
    status_text.success("Video processing completed")
    progress_bar.empty()
    frame_counter.empty()

    st.info(
        "The processed video is ready. "
        "Please download the file using the button below."
    )

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download",
            data=f,
            file_name="coral_detection_result.mp4",
            mime="video/mp4"
        )

    st.caption(f"Processing time: {elapsed:.1f} seconds")
