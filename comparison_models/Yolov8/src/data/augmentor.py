# Custom augmentation pipeline for OOD robustness

import albumentations as A
import cv2
import os
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("augmentor")

def get_train_transforms(imgsz: int = 640) -> A.Compose:
    """Augmentations focused on viewpoint and surface variation."""
    return A.Compose([
        A.RandomPerspective(scale=0.3, p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.3, p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),
        A.CLAHE(p=0.3),
        A.Sharpen(p=0.2),
        A.Resize(imgsz, imgsz),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    ))

def get_val_transforms(imgsz: int = 640) -> A.Compose:
    """Minimal transforms for validation."""
    return A.Compose([
        A.Resize(imgsz, imgsz),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"]
    ))