# In-distribution evaluation

from src.models.yolov8_model import YOLOv8Model
from src.utils.logger import get_logger
import pandas as pd
import os

logger = get_logger("evaluator")

class Evaluator:
    def __init__(self, weights: str):
        self.model = YOLOv8Model(weights)

    def evaluate(self, data: str = "configs/dataset.yaml"):
        logger.info("Running in-distribution evaluation...")
        metrics = self.model.evaluate(data=data, split="test")

        results = {
            "mAP50":     round(metrics.box.map50, 4),
            "mAP50-95":  round(metrics.box.map,   4),
            "Precision": round(float(metrics.box.p.mean()), 4),
            "Recall":    round(float(metrics.box.r.mean()), 4),
        }

        logger.info(f"Results: {results}")
        return results