
import cv2
import time
import psutil
from ultralytics import YOLO
from supervision import ByteTrack, Detections

# =========================
# Load model (TensorRT preferred)
# =========================
model = YOLO("yolov8_traffic.trt")  # fallback: yolov8_traffic.pt

# EXACT class names from PDF
model.names = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Motorbike",
    4: "Bicycle",
    5: "Rickshaw"
}

tracker = ByteTrack()

# =========================
# Video paths
# =========================
input_video = "/mnt/data/Supporting video for Dataset-3.mp4"
output_video = "processed_tracking_output.mp4"

cap = cv2.VideoCapture(input_video)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

prev_time = time.time()

# =========================
# Processing loop
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, iou=0.5)[0]
    detections = Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Draw detections + IDs
    for box, cls, conf, tid in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence,
        detections.tracker_id
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"ID:{tid} {model.names[int(cls)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # FPS + metrics overlay
    now = time.time()
    fps_now = 1 / (now - prev_time)
    prev_time = now

    cpu = psutil.cpu_percent()
    count = len(detections.tracker_id)

    overlay = f"FPS: {fps_now:.1f} | CPU: {cpu}% | Vehicles: {count}"
    cv2.rectangle(frame, (10, 10), (460, 45), (0, 0, 0), -1)
    cv2.putText(
        frame, overlay,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    writer.write(frame)

cap.release()
writer.release()

print("✅ Processed video generated:", output_video)
