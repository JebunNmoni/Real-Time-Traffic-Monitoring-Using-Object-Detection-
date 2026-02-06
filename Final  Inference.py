
import cv2
import time
import psutil
from ultralytics import YOLO
from supervision import ByteTrack, Detections

model = YOLO("yolov8_traffic.trt")

CLASS_NAMES = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Motorbike",
    4: "Bicycle",
    5: "Rickshaw"
}
model.names = CLASS_NAMES

tracker = ByteTrack()

cap = cv2.VideoCapture("Supporting video for Dataset-1")
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(
    "processed_tracking_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (w, h)
)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)[0]
    detections = Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    for box, cls, conf, tid in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence,
        detections.tracker_id
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"ID:{tid} {model.names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cpu = psutil.cpu_percent()
    count = len(detections.tracker_id)

    overlay = f"FPS:{fps:.1f} | CPU:{cpu}% | Count:{count}"
    cv2.putText(frame, overlay, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    out.write(frame)

cap.release()
out.release()
