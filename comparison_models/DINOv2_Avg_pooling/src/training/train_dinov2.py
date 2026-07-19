import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from src.training.dinov2_trainer import DINOv2Trainer
def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 road damage detector")
    parser.add_argument("--model",   type=str,   default="facebook/dinov2-small")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--queries", type=int,   default=100)
    parser.add_argument("--resume",  action="store_true", default=False)
    parser.add_argument("--unfreeze", action="store_true", default=False)
    args = parser.parse_args()

    config = {
        "model_name":      args.model,
        "epochs":          args.epochs,
        "batch":           args.batch,
        "lr":              args.lr,
        "num_classes":     6,
        "num_queries":     args.queries,
        "freeze_backbone": not args.unfreeze,
        "resume":          args.resume,
        "train_images":    "data/processed/train/images",
        "train_labels":    "data/processed/train/labels",
        "val_images":      "data/processed/val/images",
        "val_labels":      "data/processed/val/labels",
    }

    trainer = DINOv2Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
