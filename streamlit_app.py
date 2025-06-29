# streamlit_app.py
import streamlit as st
import tempfile
from object_detection_tracking import run_vehicle_detection

st.title("🚗 Vehicle Detection & Counting with YOLO + DeepSort")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    st.info("Processing video, this may take a moment...")

    run_vehicle_detection(
        input_video=temp_input.name,
        output_video=temp_output.name,
        yolo_weights="yolov8s.pt",
        mars_model="config/mars-small128.pb",
        class_file="config/coco.names"
    )

    st.success("Detection complete. See results below:")
    st.video(temp_output.name)