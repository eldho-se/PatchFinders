# Main training logic

from src.models.yolov8_model import YOLOv8Model
from src.utils.logger import get_logger
import os

logger = get_logger("trainer")

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = YOLOv8Model(
            weights=cfg.get("weights", "yolov8m.pt")
        )

    def run(self):
        logger.info("=" * 40)
        logger.info("Starting PatchFinders Training")
        logger.info("=" * 40)

        train_cfg = {
            "data":         self.cfg.get("data",    "configs/dataset.yaml"),
            "epochs":       self.cfg.get("epochs",   100),
            "imgsz":        self.cfg.get("imgsz",    640),
            "batch":        self.cfg.get("batch",    16),
            "device":       self.cfg.get("device",   0),
            "project":      "runs/train",
            "name":         self.cfg.get("name",     "yolov8m_baseline"),
            "degrees":      15.0,
            "perspective":  0.001,
            "mosaic":       1.0,
            "mixup":        0.1,
            "save":         True,
            "plots":        True,
        }

        results = self.model.train(train_cfg)
        logger.info("Training complete!")
        return results