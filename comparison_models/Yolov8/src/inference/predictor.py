"""Inference predictor module for PatchFinders."""

from src.models.yolov8_model import YOLOv8Model
from typing import Any, Dict

class Predictor:
    """Predictor handler for image and video inference."""

    def __init__(self, weights: str = "runs/train/yolov8m_baseline/weights/best.pt"):
        self.model = YOLOv8Model(weights)

    def predict(self, source: str, conf: float = 0.25, save: bool = True, **kwargs):
        """Run prediction on input source."""
        return self.model.predict(
            source=source,
            conf=conf,
            save=save,
            project="outputs/predictions",
            name="run",
            line_width=2,
            **kwargs,
        )
