"""YOLOv8 Model Wrapper for PatchFinders."""

from ultralytics import YOLO
from typing import Any, Dict

class YOLOv8Model:
    """Wrapper around Ultralytics YOLO model for training, evaluation, and inference."""
    
    def __init__(self, weights: str = "weights/yolov8m.pt"):
        self.weights = weights
        self.model = YOLO(weights)

    def train(self, train_cfg: Dict[str, Any]):
        """Train the model with specified configuration parameters."""
        return self.model.train(**train_cfg)

    def evaluate(self, data: str = "configs/dataset.yaml", split: str = "test", **kwargs):
        """Evaluate the model on a dataset split."""
        return self.model.val(data=data, split=split, **kwargs)

    def predict(self, source: str, conf: float = 0.25, **kwargs):
        """Run prediction/inference on a target source."""
        return self.model.predict(source=source, conf=conf, **kwargs)
