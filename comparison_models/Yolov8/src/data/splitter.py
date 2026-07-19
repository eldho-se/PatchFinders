# Splits processed data into train / val / test sets

import os
import shutil
import random
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("splitter")

def split_dataset(
    src_images: str,
    src_labels: str,
    out_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    random.seed(seed)
    images = sorted(Path(src_images).glob("*.jpg"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, imgs in splits.items():
        img_out = os.path.join(out_dir, split, "images")
        lbl_out = os.path.join(out_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img in imgs:
            shutil.copy2(img, os.path.join(img_out, img.name))
            lbl = Path(src_labels) / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, os.path.join(lbl_out, lbl.name))

        logger.info(f"{split}: {len(imgs)} images")

    logger.info("Split complete!")