"""Main entry point for PatchFinders repository."""

import argparse
import sys
from src.cli.train import train
from src.cli.evaluate import evaluate
from src.cli.predict import predict

def main():
    parser = argparse.ArgumentParser(
        description="PatchFinders — Cross-Domain Road Damage Detection CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model", type=str, default="m", help="Model size (n/s/m/l/x)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--device", type=int, default=0, help="GPU device (0) or CPU (-1)")
    train_parser.add_argument("--resume", action="store_true", help="Resume training")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on OOD test sets")
    eval_parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/yolov8m_baseline/weights/best.pt",
        help="Path to model weights",
    )

    # Predict subcommand
    pred_parser = subparsers.add_parser("predict", help="Run inference on images")
    pred_parser.add_argument(
        "--weights",
        type=str,
        default="runs/train/yolov8m_baseline/weights/best.pt",
        help="Path to model weights",
    )
    pred_parser.add_argument(
        "--source",
        type=str,
        default="data/processed/test/in_distribution/images",
        help="Source directory or file",
    )
    pred_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        evaluate(weights=args.weights)
    elif args.command == "predict":
        predict(weights=args.weights, source=args.source, conf=args.conf)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
