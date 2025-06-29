def run_vehicle_detection(input_video, output_video, yolo_weights, mars_model, class_file):
    import numpy as np
    import datetime
    import cv2
    from ultralytics import YOLO
    from collections import deque
    from deep_sort.deep_sort.tracker import Tracker
    from deep_sort.deep_sort import nn_matching
    from deep_sort.deep_sort.detection import Detection
    from deep_sort.tools import generate_detections as gdet
    from helper import create_video_writer

    conf_threshold = 0.5
    max_cosine_distance = 0.4
    nn_budget = None
    points = [deque(maxlen=32) for _ in range(1000)]
    counter_A, counter_B, counter_C = 0, 0, 0

    video_cap = cv2.VideoCapture(input_video)
    writer = create_video_writer(video_cap, output_video)

    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_line_A = (0, int(0.8 * frame_height))
    end_line_A = (int(0.4 * frame_width), int(0.8 * frame_height))
    start_line_B = (int(0.45 * frame_width), int(0.8 * frame_height))
    end_line_B = (int(0.7 * frame_width), int(0.8 * frame_height))
    start_line_C = (int(0.75 * frame_width), int(0.8 * frame_height))
    end_line_C = (frame_width, int(0.8 * frame_height))

    model = YOLO(yolo_weights)
    encoder = gdet.create_box_encoder(mars_model, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    with open(class_file, "r") as f:
        class_names = f.read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

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
                x1, y1, x2, y2, confidence, class_id = data
                if confidence > conf_threshold:
                    bboxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                    confidences.append(confidence)
                    class_ids.append(int(class_id))

        names = [class_names[cid] for cid in class_ids]
        features = encoder(frame, bboxes)
        dets = [Detection(b, c, n, f) for b, c, n, f in zip(bboxes, confidences, names, features)]

        tracker.predict()
        tracker.update(dets)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            track_id, class_name = track.track_id, track.get_class()
            color = colors[class_names.index(class_name)]

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            points[track_id].append((center_x, center_y))

            if len(points[track_id]) > 1:
                last_point = points[track_id][0]
                if center_y > start_line_A[1] and start_line_A[0] < center_x < end_line_A[0] and last_point[1] < start_line_A[1]:
                    counter_A += 1
                    points[track_id].clear()
                elif center_y > start_line_B[1] and start_line_B[0] < center_x < end_line_B[0] and last_point[1] < start_line_B[1]:
                    counter_B += 1
                    points[track_id].clear()
                elif center_y > start_line_C[1] and start_line_C[0] < center_x < end_line_C[0] and last_point[1] < start_line_C[1]:
                    counter_C += 1
                    points[track_id].clear()

        writer.write(frame)

    video_cap.release()
    writer.release()
