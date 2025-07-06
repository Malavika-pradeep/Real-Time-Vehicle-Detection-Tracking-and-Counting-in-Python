import vehicle_demo as gr
import cv2
import os
from ultralytics import YOLO
import tempfile

# Load YOLOv8 model (uses yolov8s.pt in your folder)
model = YOLO("yolov8s.pt")

def detect_vehicles(video_path):
    output_path = "output_detected.mp4"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return output_path

iface = gr.Interface(
    fn=detect_vehicles,
    inputs=gr.Video(label="Upload Traffic Video"),
    outputs=gr.Video(label="Detected Video Output"),
    title="Vehicle Detection with YOLOv8",
    description="Upload a traffic video to get vehicle detection results using YOLOv8."
)

if __name__ == "__main__":
    iface.launch(share=True)  # 'share=True' for public link
