import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import argparse
from PIL import Image, ImageDraw
from pathlib import Path
from transformers import AutoImageProcessor
from src.models.dinov2_model import DINOv2Detector

CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

COLORS = {
    0: (255, 0,   0),
    1: (0,   255, 0),
    2: (0,   0,   255),
    3: (255, 255, 0),
    4: (255, 0,   255),
    5: (0,   255, 255),
}

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def predict_image(model, processor, img_path, device, conf=0.5):
    image    = Image.open(img_path).convert("RGB")
    w, h     = image.size
    encoding = processor(images=image, return_tensors="pt")
    pixels   = encoding["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixels)

    logits     = outputs["logits"][0]
    pred_boxes = outputs["pred_boxes"][0]
    probs      = torch.softmax(logits, dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)

    # Filter by confidence
    mask   = scores > conf
    scores = scores[mask]
    labels = labels[mask]
    boxes  = pred_boxes[mask]

    # Keep only top 5 predictions
    if len(scores) > 5:
        top_k  = torch.topk(scores, 5)
        scores = scores[top_k.indices]
        labels = labels[top_k.indices]
        boxes  = boxes[top_k.indices]

    detections = []
    for score, label, box in zip(scores, labels, boxes):
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Skip boxes covering whole image
        if (x2 - x1) > w * 0.95 and (y2 - y1) > h * 0.95:
            continue

        detections.append({
            "class_id":   label.item(),
            "class_name": CLASSES.get(label.item(), "unknown"),
            "score":      round(score.item(), 3),
            "box":        [x1, y1, x2, y2]
        })
    return image, detections

def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = COLORS.get(det["class_id"], (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = det["class_name"] + " " + str(det["score"])
        draw.rectangle([x1, y1, x1 + len(label)*7, y1+15], fill=color)
        draw.text((x1+2, y1), label, fill=(0, 0, 0))
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,   default="runs/dinov2/best.pt")
    parser.add_argument("--source",  type=str,   default="data/processed/test/out_of_distribution/urban_surface/images")
    parser.add_argument("--conf",    type=float, default=0.3)
    args = parser.parse_args()

    device = get_device()
    print("[INFO] Using device: " + str(device))
    print("[INFO] Loading weights: " + args.weights)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model     = DINOv2Detector(
        model_name      = "facebook/dinov2-small",
        num_classes     = 6,
        freeze_backbone = True
    ).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print("[INFO] Model loaded!")

    source  = Path(args.source)
    images  = list(source.glob("*.jpg"))
    images += list(source.glob("*.JPG"))
    images += list(source.glob("*.jpeg"))
    images += list(source.glob("*.png"))
    print("[INFO] Found " + str(len(images)) + " images")

    save_dir = Path("outputs/predictions/dinov2_urban_surface")
    save_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []
    for img_path in images:
        image, detections = predict_image(
            model, processor, img_path, device, args.conf
        )
        all_detections.append({
            "image":      img_path.name,
            "detections": len(detections),
            "classes":    [d["class_name"] for d in detections]
        })
        if detections:
            image = draw_detections(image, detections)
            image.save(str(save_dir / img_path.name))
        print("  " + img_path.name + ": " + str(len(detections)) + " detections")

    print("\n" + "=" * 50)
    print("  DINOv2 Inference Summary")
    print("=" * 50)
    total = sum(d["detections"] for d in all_detections)
    print("  Images processed : " + str(len(images)))
    print("  Total detections : " + str(total))
    if images:
        print("  Avg per image    : " + str(round(total/len(images), 1)))
    print("  Results saved to : " + str(save_dir))

    from collections import Counter
    all_classes = []
    for d in all_detections:
        all_classes.extend(d["classes"])
    class_counts = Counter(all_classes)
    print("\n  Class breakdown:")
    for cls, cnt in class_counts.most_common():
        print("    " + cls.ljust(25) + str(cnt))
    print("=" * 50)

if __name__ == "__main__":
    main()
