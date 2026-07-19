import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from PIL import Image
from transformers import AutoImageProcessor
from src.models.dinov2_model import DINOv2Detector

# ── Config ───────────────────────────────────────────────
IMAGE_FOLDER = "data/processed/test/out_of_distribution/images"
WEIGHTS      = "runs/dinov2/best.pt"
MODEL_NAME   = "facebook/dinov2-small"
NUM_CLASSES  = 6
NUM_QUERIES  = 100
CONF_THRESH  = 0.3

CLASS_NAMES = [
    'pothole', 'longitudinal_crack', 'transverse_crack',
    'alligator_crack', 'rutting', 'surface_deterioration'
]

# ── Device ───────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[INFO] Using Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("[INFO] Using CUDA")
else:
    device = torch.device("cpu")
    print("[INFO] Using CPU")

# ── Load model ───────────────────────────────────────────
print("[INFO] Loading model...")
model = DINOv2Detector(
    model_name      = MODEL_NAME,
    num_classes     = NUM_CLASSES,
    num_queries     = NUM_QUERIES,
    freeze_backbone = True,
).to(device)

model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
print("[INFO] Model loaded successfully")

# ── Processor ────────────────────────────────────────────
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# ── Run on images ────────────────────────────────────────
print(f"\n[INFO] Running on images in: {IMAGE_FOLDER}\n")
print("=" * 50)

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if len(image_files) == 0:
    print("[ERROR] No images found in folder!")
    sys.exit(1)

print(f"[INFO] Found {len(image_files)} images\n")

for img_name in sorted(image_files):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    image    = Image.open(img_path).convert("RGB")

    inputs       = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    logits     = outputs["logits"][0]       # (num_queries, num_classes)
    pred_boxes = outputs["pred_boxes"][0]   # (num_queries, 4)

    probs          = torch.softmax(logits, dim=-1)
    scores, labels = probs.max(dim=-1)

    # ── Remove duplicate boxes ───────────────────────────
    seen_boxes   = []
    kept_indices = []
    for i in range(len(labels)):
        if scores[i] < CONF_THRESH:
            continue
        box = pred_boxes[i].cpu().numpy().round(3).tolist()
        if box not in seen_boxes:
            seen_boxes.append(box)
            kept_indices.append(i)

    print(f"Image: {img_name}")
    if len(kept_indices) == 0:
        print("  → Nothing detected above threshold")
    else:
        for i in kept_indices:
            cls  = CLASS_NAMES[labels[i].item()]
            conf = scores[i].item()
            box  = pred_boxes[i].cpu().numpy()
            print(f"  → {cls} ({conf:.0%} confidence) | box: {box.round(3)}")
    print()

print("=" * 50)
print("[INFO] Done!")

import cv2

output_folder = "data/processed/test/out_of_distribution/predictions"
os.makedirs(output_folder, exist_ok=True)

for img_name in sorted(image_files):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    image    = Image.open(img_path).convert("RGB")
    img_cv   = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w     = img_cv.shape[:2]

    inputs       = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    logits     = outputs["logits"][0]
    pred_boxes = outputs["pred_boxes"][0]
    probs          = torch.softmax(logits, dim=-1)
    scores, labels = probs.max(dim=-1)

    seen_boxes   = []
    kept_indices = []
    for i in range(len(labels)):
        if scores[i] < CONF_THRESH:
            continue
        box = pred_boxes[i].cpu().numpy().round(3).tolist()
        if box not in seen_boxes:
            seen_boxes.append(box)
            kept_indices.append(i)

    img_draw = cv2.imread(img_path)
    for i in kept_indices:
        cls  = CLASS_NAMES[labels[i].item()]
        conf = scores[i].item()
        cx, cy, bw, bh = pred_boxes[i].cpu().numpy()
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, f"{cls} {conf:.0%}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = os.path.join(output_folder, img_name)
    cv2.imwrite(out_path, img_draw)

print(f"\n[INFO] Visualisations saved to {output_folder}")
