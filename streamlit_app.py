import streamlit as st
import tempfile
import os
import cv2
from ultralytics import YOLO
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection

st.title("🚗 Vehicle Detection & Tracking Demo")

uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_video_path = tmp_input.name

    st.video(input_video_path)
    st.write("Processing, please wait...")

    # Define output path
    output_path = input_video_path.replace(".mp4", "_output.mp4")

    # Load YOLO & DeepSort
    model = YOLO("yolov8s.pt")  # Adjust to your model
    encoder = gdet.create_box_encoder("config/mars-small128.pb", batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
    tracker = Tracker(metric)

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]

        bboxes = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        features = encoder(frame, bboxes)
        detections = [Detection(bbox, 1.0, "car", feat) for bbox, feat in zip(bboxes, features)]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    st.success("Processing complete!")
    st.video(output_path)
    with open(output_path, "rb") as f:
        st.download_button("Download Result", f, "result.mp4")
