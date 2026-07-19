# predict.py  ← Run inference on images

from ultralytics import YOLO
import argparse

def predict(
    weights="runs/train/yolov8m_baseline/weights/best.pt",
    source="data/processed/test/in_distribution/images",
    conf=0.25,
    save=True
):
    print("=" * 50)
    print("   PatchFinders - Inference")
    print("=" * 50)

    model = YOLO(weights)
    print(f"[INFO] Running inference on: {source}")

    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        project="outputs/predictions",
        name="run",
        line_width=2,
    )

    print(f"[INFO] Predictions saved to outputs/predictions/run/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="runs/train/yolov8m_baseline/weights/best.pt")
    parser.add_argument("--source",  type=str,
                        default="data/processed/test/in_distribution/images")
    parser.add_argument("--conf",    type=float, default=0.25)
    args = parser.parse_args()

    predict(args.weights, args.source, args.conf)