import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoImageProcessor
from tqdm import tqdm
import os

from src.data.dinov2_dataset import RoadDamageDataset, collate_fn
from src.models.dinov2_model import DINOv2Detector


class DetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        logits     = outputs["logits"]
        pred_boxes = outputs["pred_boxes"]

        total_cls  = torch.tensor(0.0, device=logits.device)
        total_box  = torch.tensor(0.0, device=logits.device)
        valid      = 0

        for b in range(len(targets)):
            gt_labels = targets[b].get("class_labels")
            gt_boxes  = targets[b].get("boxes")

            if gt_labels is None or len(gt_labels) == 0:
                continue

            gt_labels = gt_labels.to(logits.device)
            gt_boxes  = gt_boxes.to(logits.device)
            n_gt      = min(len(gt_labels), logits.shape[1])

            cls_pred = logits[b, :n_gt]
            box_pred = pred_boxes[b, :n_gt]

            gt_boxes_norm = gt_boxes[:n_gt].clone()
            if gt_boxes_norm.max() > 1.0:
                gt_boxes_norm = gt_boxes_norm / 640.0

            total_cls += self.cls_loss(cls_pred, gt_labels[:n_gt])
            total_box += self.box_loss(box_pred, gt_boxes_norm)
            valid     += 1

        if valid > 0:
            total_cls /= valid
            total_box /= valid

        total = total_cls + 5.0 * total_box
        return total, {"cls": total_cls.item(), "box": total_box.item()}


class DINOv2Trainer:
    def __init__(self, config):
        self.config    = config
        self.device    = self._get_device()
        self.processor = AutoImageProcessor.from_pretrained(
            config.get("model_name", "facebook/dinov2-base")
        )

    def _get_device(self):
        if torch.backends.mps.is_available():
            print("[INFO] Using Apple MPS (Mac GPU)")
            return torch.device("mps")
        if torch.cuda.is_available():
            print("[INFO] Using CUDA GPU")
            return torch.device("cuda")
        print("[INFO] Using CPU")
        return torch.device("cpu")

    def train(self):
        cfg = self.config

        train_ds = RoadDamageDataset(
            img_dir   = cfg.get("train_images", "data/processed/train/images"),
            label_dir = cfg.get("train_labels", "data/processed/train/labels"),
            processor = self.processor,
        )
        val_ds = RoadDamageDataset(
            img_dir   = cfg.get("val_images", "data/processed/val/images"),
            label_dir = cfg.get("val_labels", "data/processed/val/labels"),
            processor = self.processor,
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.get("batch", 4),
                                  shuffle=True,  collate_fn=collate_fn, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.get("batch", 4),
                                  shuffle=False, collate_fn=collate_fn, num_workers=0)

        model = DINOv2Detector(
            model_name      = cfg.get("model_name", "facebook/dinov2-base"),
            num_classes     = cfg.get("num_classes", 6),
            num_queries     = cfg.get("num_queries", 100),
            freeze_backbone = cfg.get("freeze_backbone", True),
        ).to(self.device)

        criterion = DetectionLoss(num_classes=cfg.get("num_classes", 6))
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.get("lr", 1e-4), weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 30))

        best_loss = float("inf")
        os.makedirs("runs/dinov2", exist_ok=True)

        print("\n" + "=" * 50)
        print("  DINOv2 Road Damage Detector")
        print("=" * 50)
        print("  Device     : " + str(self.device))
        print("  Train size : " + str(len(train_ds)))
        print("  Val size   : " + str(len(val_ds)))
        print("  Epochs     : " + str(cfg.get("epochs", 30)))
        print("  Batch      : " + str(cfg.get("batch", 4)))
        print("=" * 50 + "\n")

        for epoch in range(1, cfg.get("epochs", 30) + 1):
            model.train()
            train_losses = []

            pbar = tqdm(train_loader,
                        desc="Epoch " + str(epoch) + "/" + str(cfg.get("epochs", 30)) + " [Train]")
            for pixel_values, targets in pbar:
                pixel_values = pixel_values.to(self.device)
                optimizer.zero_grad()
                outputs      = model(pixel_values)
                loss, ld     = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                train_losses.append(loss.item())
                pbar.set_postfix({"loss": round(loss.item(), 4),
                                  "cls": round(ld["cls"], 4),
                                  "box": round(ld["box"], 4)})

            scheduler.step()
            avg_train = sum(train_losses) / len(train_losses)

            model.eval()
            val_losses = []
            with torch.no_grad():
                for pixel_values, targets in tqdm(val_loader,
                    desc="Epoch " + str(epoch) + "/" + str(cfg.get("epochs", 30)) + " [Val]"):
                    pixel_values = pixel_values.to(self.device)
                    outputs      = model(pixel_values)
                    loss, _      = criterion(outputs, targets)
                    val_losses.append(loss.item())

            avg_val = sum(val_losses) / len(val_losses)
            print("\n[Epoch " + str(epoch) + "] Train: " + str(round(avg_train, 4)) +
                  " | Val: " + str(round(avg_val, 4)) +
                  " | LR: " + str(round(scheduler.get_last_lr()[0], 6)))

            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), "runs/dinov2/best.pt")
                print("  Best model saved (val_loss: " + str(round(best_loss, 4)) + ")")

            torch.save(model.state_dict(), "runs/dinov2/last.pt")

        print("\n[INFO] Training complete!")
        print("[INFO] Best val loss: " + str(round(best_loss, 4)))
        print("[INFO] Weights saved to runs/dinov2/")