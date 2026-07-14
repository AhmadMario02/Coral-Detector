import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
from ultralytics import YOLO

st.set_page_config(
    page_title="Coral Reef Health Monitoring System",
    layout="centered"
)
st.sidebar.header("Navigation")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold (YOLO)",
    min_value=0.0, max_value=1.0, value=0.596, step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.caption("**Detection Model :** YOLOv11s (best.pt)")
st.sidebar.caption("**Classification Model :** ResNet18 (fine-tuned)")

st.title("Coral Reef Health Monitoring System")

YOLO_WEIGHTS_PATH = "models/best.pt"
CNN_WEIGHTS_PATH = "models/cnn_best.pth"

CLASS_NAMES = ["bleached_corals", "healthy_corals"]

COLOR_MAP = {
    "bleached_corals": (255, 0, 0),   # merah
    "healthy_corals": (0, 200, 0),    # hijau
}

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_yolo_model(weights_path):
    return YOLO(weights_path)


@st.cache_resource
def load_cnn_model(weights_path):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def classify_crop(crop_bgr, cnn_model):
    """Klasifikasikan satu crop (format BGR dari cv2) menjadi healthy/bleached."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    input_tensor = CNN_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = cnn_model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item()) * 100

    return CLASS_NAMES[pred_idx], confidence


def run_pipeline(image_bgr, yolo_model, cnn_model, conf_threshold):
    """Jalankan pipeline lengkap: YOLO deteksi -> crop -> CNN klasifikasi -> gambar ulang."""
    results = yolo_model(image_bgr, conf=conf_threshold, verbose=False)
    result = results[0]

    num_healthy = 0
    num_bleached = 0
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image_bgr.shape[1], x2)
        y2 = min(image_bgr.shape[0], y2)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        label, confidence = classify_crop(crop, cnn_model)

        if label == "bleached_corals":
            num_bleached += 1
        else:
            num_healthy += 1

        detections.append({"box": (x1, y1, x2, y2), "label": label, "confidence": confidence})

        color_rgb = COLOR_MAP[label]
        color_bgr = color_rgb[::-1]

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color_bgr, 2)

        text = f"{label} {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(
            image_bgr, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, num_healthy, num_bleached, detections

st.write("Upload underwater images to view the results of coral health detection and classification.")

uploaded_file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with st.spinner("Running detection and classification pipeline..."):
        try:
            yolo_model = load_yolo_model(YOLO_WEIGHTS_PATH)
            cnn_model = load_cnn_model(CNN_WEIGHTS_PATH)
        except FileNotFoundError as e:
            st.error(
                f"Model not found: {e}\n\n"
            )
            st.stop()

        result_image, num_healthy, num_bleached, detections = run_pipeline(
            image_bgr.copy(), yolo_model, cnn_model, conf_threshold
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real Image")
        st.image(pil_image, use_container_width=True)
    with col2:
        st.subheader("Multi-Stage Result")
        st.image(result_image, use_container_width=True)

    st.divider()

    total_detected = num_healthy + num_bleached
    m1, m2, m3 = st.columns(3)
    m1.metric("Detected", total_detected)
    m2.metric("Healthy", num_healthy)
    m3.metric("Bleached", num_bleached)

    if total_detected > 0:
        bleached_pct = (num_bleached / total_detected) * 100
        if bleached_pct >= 50:
            st.warning(
                f"⚠️ {bleached_pct:.1f}% The detected coral shows signs of bleaching."
            )
        else:
            st.success(
                f"✅ The majority of the coral was detected ({100 - bleached_pct:.1f}%) to be in good condition."
            )

        with st.expander("Details"):
            for i, det in enumerate(detections, start=1):
                st.write(f"**Object {i}:** {det['label']} — confidence {det['confidence']:.1f}%")
    else:
        st.info("No coral features were detected in this image. Try lowering the confidence threshold in the sidebar.")
else:
    st.info("Please upload an image to start the process.")