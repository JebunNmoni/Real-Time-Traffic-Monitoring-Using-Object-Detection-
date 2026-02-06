model = YOLO("yolov8_traffic.trt")

import time
import psutil

prev_time = time.time()

def draw_metrics(frame, fps, count):
    cpu = psutil.cpu_percent()
    text = f"FPS: {fps:.1f} | CPU: {cpu}% | Vehicles: {count}"

    cv2.rectangle(frame, (10, 10), (420, 45), (0, 0, 0), -1)
    cv2.putText(
        frame, text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
