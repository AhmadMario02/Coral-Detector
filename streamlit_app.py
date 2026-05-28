import streamlit as st

st.set_page_config(
    page_title="Coral Reef Health Monitoring System",
    layout="centered"
)
st.sidebar.header("Navigation")

st.title("Coral Reef Health Monitoring System")

st.caption(
    "Deep Learning-Based Multi-Stage System for Coral Detection and Health Classification"
)

st.markdown("""
This application was developed as part of a research project focusing on 
automatic coral reef monitoring using Computer Vision and Deep Learning.

The proposed system integrates:
- YOLO for coral object detection
- CNN for coral health classification

The system aims to assist in identifying healthy and bleached coral reefs 
from underwater imagery automatically and efficiently.
""")