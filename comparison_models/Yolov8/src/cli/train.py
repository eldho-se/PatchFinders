# train.py  ← Main training entry point

from ultralytics import YOLO
import argparse
import os

def train(
    model_size="m",
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    resume=False
):
    print("=" * 50)
    print("   PatchFinders - Road Damage Detection")
    print("=" * 50)

    # Load pretrained model
    weights_path = f"weights/yolov8{model_size}.pt"
    if os.path.exists(weights_path):
        model_path = weights_path
    else:
        model_path = f"yolov8{model_size}.pt"

    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # Start training
    print(f"[INFO] Starting training for {epochs} epochs...")
    results = model.train(
        data="configs/dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        resume=resume,
        workers=0,  

        # Augmentation for OOD robustness
        degrees=15.0,           # Rotation
        perspective=0.001,      # Perspective shift
        flipud=0.1,             # Vertical flip
        fliplr=0.5,             # Horizontal flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # MixUp augmentation
        hsv_h=0.015,            # Hue
        hsv_s=0.7,              # Saturation
        hsv_v=0.4,              # Value/brightness

        # Saving
        project="runs/train",
        name=f"yolov8{model_size}_baseline",
        save=True,
        plots=True,
    )

    print("\n[INFO] Training complete!")
    print(f"[INFO] Best weights saved at: runs/train/yolov8{model_size}_baseline/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train road damage detection model")
    parser.add_argument("--model",   type=str,  default="m",    help="Model size: n/s/m/l/x")
    parser.add_argument("--epochs",  type=int,  default=100,    help="Number of epochs")
    parser.add_argument("--batch",   type=int,  default=16,     help="Batch size")
    parser.add_argument("--imgsz",   type=int,  default=640,    help="Image size")
    parser.add_argument("--device",  type=int,  default=0,      help="GPU device (0) or CPU (-1)")
    parser.add_argument("--resume",  action="store_true",       help="Resume training")
    args = parser.parse_args()

    train(
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume
    )