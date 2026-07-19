# Basic model tests

import os
from ultralytics import YOLO
import pytest

def test_model_loads():
    weights_path = "weights/yolov8n.pt" if os.path.exists("weights/yolov8n.pt") else "yolov8n.pt"
    model = YOLO(weights_path)
    assert model is not None

def test_class_mapping():
    from src.utils.class_mapping import UNIFIED_CLASSES, CLASS_TO_IDX
    assert len(UNIFIED_CLASSES) == 6
    assert CLASS_TO_IDX["pothole"] == 0