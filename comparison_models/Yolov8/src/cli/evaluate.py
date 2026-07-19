# evaluate.py  ← Evaluation entry point

from ultralytics import YOLO
import os
import json
import pandas as pd

OOD_SETS = {
    "in_distribution":      "data/processed/val/images",
    "urban_surface":        "data/processed/test/out_of_distribution/urban_surface/test/images",
    "pedestrian_viewpoint": "data/processed/test/out_of_distribution/pedestrian_viewpoint/images",
    "cobblestone":          "data/processed/test/out_of_distribution/cobblestone/images",
    "dirt_path":            "data/processed/test/out_of_distribution/dirt_path/images",
}

def evaluate(weights="runs/detect/runs/train/yolov8s_baseline-4/weights/best.pt"):
    print("=" * 50)
    print("   PatchFinders - OOD Evaluation")
    print("=" * 50)

    # Fall back if specified weights file doesn't exist locally
    if not os.path.exists(weights):
        fallback_candidates = [
            "runs/detect/runs/train/yolov8s_baseline/weights/best.pt",
            "runs/detect/runs/train/yolov8n_baseline-4/weights/best.pt",
            "weights/yolov8s.pt",
            "weights/yolov8n.pt"
        ]
        for candidate in fallback_candidates:
            if os.path.exists(candidate):
                weights = candidate
                break

    print(f"[INFO] Evaluating model weights: {weights}")
    model = YOLO(weights)
    all_results = {}

    # Primary in-distribution / validation evaluation
    print("\n[EVAL] Running In-Distribution Validation...")
    metrics = model.val(
        data="configs/dataset.yaml",
        split="val",
        project="outputs/metrics",
        name="in_distribution",
        save_json=True,
    )

    all_results["in_distribution"] = {
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50-95":  round(float(metrics.box.map),   4),
        "Precision": round(float(metrics.box.mp),      4),
        "Recall":    round(float(metrics.box.mr),      4),
    }
    print(f"   mAP50: {all_results['in_distribution']['mAP50']} | mAP50-95: {all_results['in_distribution']['mAP50-95']}")

    # Save summary
    os.makedirs("outputs/metrics", exist_ok=True)
    df = pd.DataFrame(all_results).T
    df.to_csv("outputs/metrics/ood_summary.csv")
    print("\n[INFO] Results saved to outputs/metrics/ood_summary.csv")
    print("\n" + df.to_string())
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="runs/detect/runs/train/yolov8s_baseline-4/weights/best.pt")
    args = parser.parse_args()
    evaluate(args.weights)