import streamlit as st
import tempfile
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet
from helper import create_video_writer

st.title("Vehicle Detection, Tracking & Counting")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    st.video(input_path)

    if st.button("Run Detection"):

        conf_threshold = 0.5
        max_cosine_distance = 0.4
        nn_budget = None
        points = [deque(maxlen=32) for _ in range(1000)]
        counter_A, counter_B, counter_C = 0, 0, 0

        video_cap = cv2.VideoCapture(input_path)
        output_path = input_path.replace(".mp4", "_output.mp4")
        writer = create_video_writer(video_cap, output_path)

        model = YOLO("yolov8s.pt")
        encoder = gdet.create_box_encoder("config/mars-small128.pb", batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        with open("config/coco.names", "r") as f:
            class_names = f.read().strip().split("\n")

        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3))

        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_line_A = (0, int(0.8 * frame_height))
        end_line_A = (int(0.4 * frame_width), int(0.8 * frame_height))
        start_line_B = (int(0.45 * frame_width), int(0.8 * frame_height))
        end_line_B = (int(0.7 * frame_width), int(0.8 * frame_height))
        start_line_C = (int(0.75 * frame_width), int(0.8 * frame_height))
        end_line_C = (frame_width, int(0.8 * frame_height))

        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            overlay = frame.copy()
            cv2.line(frame, start_line_A, end_line_A, (0, 255, 0), 12)
            cv2.line(frame, start_line_B, end_line_B, (255, 0, 0), 12)
            cv2.line(frame, start_line_C, end_line_C, (0, 0, 255), 12)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            results = model(frame)

            for result in results:
                bboxes, confidences, class_ids = [], [], []
                for data in result.boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = data
                    if conf > conf_threshold:
                        bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                        confidences.append(conf)
                        class_ids.append(int(class_id))

            names = [class_names[i] for i in class_ids]
            features = encoder(frame, bboxes)
            dets = [Detection(bbox, conf, name, feature) for bbox, conf, name, feature in zip(bboxes, confidences, names, features)]

            tracker.predict()
            tracker.update(dets)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                color = colors[class_names.index(track.get_class())]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)

            writer.write(frame)

        video_cap.release()
        writer.release()

        st.success("Processing Complete")
        st.video(output_path)
