"""
Leak Detection Module
Combines preprocessing and YOLO inference
"""
import torch
import cv2
from preprocessing.leak_preprocessing import preprocess_leak

# Load model once at module level
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/leak_detection.pt', force_reload=False)
model.conf = 0.1 # our model recommendation

def detect_leak(image):
    # Apply preprocessing
    processed = preprocess_leak(image)

    # Run detection on preprocessed image
    results = model(processed)

    detections = []
    pred = results.xyxy[0].cpu().numpy()

    for x1, y1, x2, y2, conf, cls in pred:
        detections.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'confidence': float(conf)
        })

    return detections
