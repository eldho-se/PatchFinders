# Out-of-distribution evaluation across all test sets

from src.models.yolov8_model import YOLOv8Model
from src.utils.logger import get_logger
import pandas as pd
import os

logger = get_logger("ood_evaluator")

OOD_SETS = {
    "in_distribution":      "data/processed/test/in_distribution/images",
    "pedestrian_viewpoint": "data/processed/test/out_of_distribution/pedestrian_viewpoint/images",
    "cobblestone":          "data/processed/test/out_of_distribution/cobblestone/images",
    "dirt_path":            "data/processed/test/out_of_distribution/dirt_path/images",
    "urban_surface":        "data/processed/test/out_of_distribution/urban_surface/images",
}

class OODEvaluator:
    def __init__(self, weights: str):
        self.model = YOLOv8Model(weights)

    def evaluate_all(self):
        all_results = {}

        for name, path in OOD_SETS.items():
            if not os.path.exists(path):
                logger.warning(f"Skipping {name} — path not found")
                continue

            logger.info(f"Evaluating: {name}")
            metrics = self.model.evaluate(
                data="configs/dataset.yaml", split="test"
            )

            all_results[name] = {
                "mAP50":     round(metrics.box.map50, 4),
                "mAP50-95":  round(metrics.box.map,   4),
                "Precision": round(float(metrics.box.p.mean()), 4),
                "Recall":    round(float(metrics.box.r.mean()), 4),
            }

        df = pd.DataFrame(all_results).T
        os.makedirs("outputs/metrics", exist_ok=True)
        df.to_csv("outputs/metrics/ood_summary.csv")
        logger.info("\n" + df.to_string())
        return all_results