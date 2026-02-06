pip install ultralytics onnx onnxruntime-gpu

from ultralytics import YOLO

model = YOLO("yolov8_traffic.pt")
model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=True
)
