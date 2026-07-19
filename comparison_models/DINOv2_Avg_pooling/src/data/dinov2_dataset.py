import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class RoadDamageDataset(Dataset):
    def __init__(self, img_dir, label_dir, processor, imgsz=518):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        self.processor = processor
        self.imgsz     = imgsz
        self.samples   = []

        for img_path in Path(img_dir).glob("*.jpg"):
            lbl_path = Path(label_dir) / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((str(img_path), str(lbl_path)))

        for img_path in Path(img_dir).glob("*.png"):
            lbl_path = Path(label_dir) / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((str(img_path), str(lbl_path)))

        print("[INFO] Loaded " + str(len(self.samples)) + " samples from " + img_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h  = image.size

        # Load labels
        boxes  = []
        labels = []
        areas  = []

        with open(lbl_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts    = line.split()
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h

                boxes.append([x1, y1, x2 - x1, y2 - y1])
                labels.append(class_id)
                areas.append((x2 - x1) * (y2 - y1))

        if not boxes:
            boxes  = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
            areas  = [1.0]

        # Process image only — no annotations passed to processor
        encoding     = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()

        # Build target separately
        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes":        torch.tensor(boxes,  dtype=torch.float32),
            "area":         torch.tensor(areas,  dtype=torch.float32),
            "image_id":     torch.tensor([idx]),
        }

        return pixel_values, target


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    targets      = [item[1] for item in batch]
    return pixel_values, targets