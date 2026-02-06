import cv2
from ultralytics import YOLO
from supervision import ByteTrack, Detections

# Load model
model = YOLO("yolov8_traffic.pt")

# Tracker
tracker = ByteTrack()

# Video IO
cap = cv2.VideoCapture("Supporting video for Dataset-1.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "processed_tracking_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, iou=0.5)[0]

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
        cv2.putText(
            frame, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    out.write(frame)

cap.release()
out.release()
