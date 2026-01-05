# Coral Detector App 🪸

https://coraldetector.streamlit.app/

Coral Detector App is a Computer Vision–based application developed as a final project.  
This application utilizes deep learning–based object detection to detect and count coral objects from images and videos.

---

## 📌 Project Overview

Coral reefs are an important marine ecosystem that requires continuous monitoring.  
This project aims to assist coral reef analysis by providing an automated coral detection system using a YOLO-based object detection model.

The application allows users to:

- Upload images or videos
- Perform coral detection
- Display bounding boxes on detected corals
- Count the number of detected corals
- Measure inference speed (FPS)
- Download processed video results

---

## 🧠 Methodology

The system is built using:

- **YOLO (You Only Look Once)** for object detection
- **Deep Learning** for feature extraction and classification
- **Streamlit** for interactive web-based deployment

### Detection Pipeline:

1. Input image or video is uploaded
2. YOLO model performs object detection
3. Bounding boxes are drawn on detected corals
4. The number of detected corals is counted per frame
5. Inference speed (FPS) is calculated
6. Results are displayed (image) or exported (video)

---

## ⚙️ Features

### ✅ Image Inference

- Coral detection with bounding boxes
- Total coral count
- Inference time and FPS measurement

### ✅ Video Inference

- Frame-by-frame coral detection
- Bounding boxes overlay
- Total coral count per frame
- FPS displayed on each frame
- Progress bar during video processing
- Downloadable processed video

### ✅ User Interface

- Confidence Threshold slider
- Clear device status (CPU / GPU)
- Informative processing status
- Clean and user-friendly layout

---

## 🎛 Confidence Threshold Explanation

The confidence threshold defines the minimum confidence score required for a detection to be considered valid.

- **Higher values** reduce false positives but may miss some detections
- **Lower values** allow more detections but increase the risk of false positives

This parameter allows users to control the sensitivity of the detection model.

---

## 🖥 Device Support

- Automatically detects available hardware
- Runs on **CPU** by default
- Supports **GPU (CUDA)** if available

The current device status is displayed on the application interface.

---

## 📦 Tech Stack

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Streamlit
- NumPy
- PIL (Python Imaging Library)

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Notes

This application is designed to run on Streamlit Cloud (CPU only).
OpenCV is installed using the headless version for compatibility.

---

## 📊 Output Description

- Images: Detection results are displayed directly in the application
- Videos: Detection results are exported as a downloadable video file to ensure playback stability

Each video frame includes:

- Bounding boxes
- Total detected coral count
- Inference FPS

---

## 📘 Academic Context

This application is developed as part of a final undergraduate project in Computer Vision Course.
It demonstrates the application of deep learning–based object detection for marine ecosystem analysis.

---

## ✨ Acknowledgements

- Ultralytics YOLO
- PyTorch Community
- Streamlit Team
