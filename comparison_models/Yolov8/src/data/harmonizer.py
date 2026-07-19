# Unifies all datasets into one consistent format

import os
import shutil
from pathlib import Path
from src.data.converter import voc_to_yolo, save_yolo_label
from src.utils.file_utils import get_image_paths, ensure_dir
from src.utils.logger import get_logger
import cv2

logger = get_logger("harmonizer")

def harmonize_rdd2022(raw_dir: str, out_dir: str):
    """Process RDD2022 dataset into unified YOLO format."""
    logger.info(f"Harmonizing RDD2022 from {raw_dir}")
    images = get_image_paths(raw_dir)

    for img_path in images:
        xml_path = str(img_path).replace("images", "annotations").replace(img_path.suffix, ".xml")
        if not os.path.exists(xml_path):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        labels = voc_to_yolo(xml_path, w, h)
        if not labels:
            continue

        # Save image
        dst_img = os.path.join(out_dir, "images", img_path.name)
        ensure_dir(os.path.dirname(dst_img))
        shutil.copy2(str(img_path), dst_img)

        # Save label
        dst_lbl = os.path.join(out_dir, "labels", img_path.stem + ".txt")
        ensure_dir(os.path.dirname(dst_lbl))
        save_yolo_label(labels, dst_lbl)

    logger.info(f"RDD2022 harmonization complete → {out_dir}")

def run_full_harmonization():
    datasets = {
        "data/raw/RDD2022":   "data/processed/train",
        "data/raw/CRACK500":  "data/processed/train",
        "data/raw/Pothole600": "data/processed/train",
    }
    for raw, out in datasets.items():
        if os.path.exists(raw):
            harmonize_rdd2022(raw, out)
        else:
            logger.warning(f"Dataset not found: {raw}")

if __name__ == "__main__":
    run_full_harmonization()