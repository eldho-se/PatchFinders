# Converts VOC (XML) and COCO (JSON) labels to YOLO (.txt) format

import os
import json
import xml.etree.ElementTree as ET
from src.utils.class_mapping import CLASS_TO_IDX, LABEL_MAPPING
from src.utils.logger import get_logger

logger = get_logger("converter")

def voc_to_yolo(xml_path: str, img_w: int, img_h: int) -> list:
    """Convert Pascal VOC XML annotation to YOLO format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_labels = []

    for obj in root.findall("object"):
        raw_name = obj.find("name").text
        name = LABEL_MAPPING.get(raw_name, raw_name)
        if name not in CLASS_TO_IDX:
            logger.warning(f"Unknown class: {name}, skipping")
            continue

        class_id = CLASS_TO_IDX[name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (normalized)
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h

        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_labels

def save_yolo_label(labels: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(labels))