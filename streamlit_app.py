# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Vehicle Detection Demo", layout="centered")
st.title("🚗 Real-Time Vehicle Detection and Counting")
st.write("Upload a video to see vehicle detection and counting in action!")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Basic vehicle-like color detection (placeholder)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = cv2.Canny(gray, 100, 200)
        colored_cars = cv2.cvtColor(cars, cv2.COLOR_GRAY2BGR)
        
        combined = cv2.addWeighted(frame, 0.8, colored_cars, 0.2, 0)

        stframe.image(combined, channels="BGR", use_column_width=True)

    cap.release()
    st.success("Video Processing Complete!")