import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.dinov2_model import DINOv2Detector

CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_ground_truth(label_dir):
    gt = {}
    label_path = Path(label_dir)
    if not label_path.exists():
        print("[WARN] Label dir not found: " + str(label_dir))
        return gt
    for lbl_file in label_path.glob("*.txt"):
        classes = []
        with open(lbl_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    cls = int(line.split()[0])
                    classes.append(cls)
        gt[lbl_file.stem] = classes
    return gt

def predict_scores(model, processor, img_path, device):
    image    = Image.open(img_path).convert("RGB")
    encoding = processor(images=image, return_tensors="pt")
    pixels   = encoding["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixels)

    logits = outputs["logits"][0]
    probs  = torch.softmax(logits, dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)

    class_scores = defaultdict(float)
    for score, label in zip(scores.tolist(), labels.tolist()):
        cls = int(label)
        class_scores[cls] = max(class_scores[cls], score)

    max_score = max(scores.tolist()) if len(scores) > 0 else 0.0
    return max_score, class_scores

def main():
    img_dir   = "data/processed/test/out_of_distribution/road_surface_manual/test/images"
    label_dir = "data/processed/test/out_of_distribution/road_surface_manual/test/labels"
    weights   = "runs/dinov2/best.pt"

    print("=" * 60)
    print("  PatchFinders - DINOv2 AUROC Evaluation")
    print("=" * 60)

    device    = get_device()
    print("[INFO] Device: " + str(device))

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model     = DINOv2Detector(
        model_name      = "facebook/dinov2-small",
        num_classes     = 6,
        freeze_backbone = True
    ).to(device)

    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print("[INFO] Model loaded!")

    gt_labels = load_ground_truth(label_dir)
    print("[INFO] GT files: " + str(len(gt_labels)))

    images  = list(Path(img_dir).glob("*.jpg"))
    images += list(Path(img_dir).glob("*.JPG"))
    images += list(Path(img_dir).glob("*.jpeg"))
    images += list(Path(img_dir).glob("*.png"))
    print("[INFO] Images: " + str(len(images)))

    per_class_gt     = defaultdict(list)
    per_class_scores = defaultdict(list)

    for i, img_path in enumerate(images):
        img_name  = img_path.stem
        _, class_scores = predict_scores(
            model, processor, img_path, device
        )

        gt = gt_labels.get(img_name, [])

        for cls_id in range(6):
            per_class_gt[cls_id].append(1 if cls_id in gt else 0)
            per_class_scores[cls_id].append(class_scores.get(cls_id, 0.0))

        if (i + 1) % 50 == 0:
            print("  Processed: " + str(i+1) + "/" + str(len(images)))

    print("\n" + "=" * 60)
    print("  DINOv2 AUROC Results")
    print("=" * 60)

    class_aurocs = {}
    print("\n  Per-Class AUROC:")
    print("  " + "-" * 50)
    for cls_id in range(6):
        if cls_id == 4:
            continue
        gt_arr    = np.array(per_class_gt[cls_id])
        score_arr = np.array(per_class_scores[cls_id])
        if len(np.unique(gt_arr)) > 1:
            auroc = roc_auc_score(gt_arr, score_arr)
            ap    = average_precision_score(gt_arr, score_arr)
            class_aurocs[cls_id] = auroc
            bar = "#" * int(auroc * 20)
            print("  " + CLASSES[cls_id].ljust(25) +
                  " AUROC: " + str(round(auroc, 4)) +
                  "  AP: " + str(round(ap, 4)) +
                  "  " + bar)
        else:
            class_aurocs[cls_id] = None
            print("  " + CLASSES[cls_id].ljust(25) + " AUROC: N/A")

    # Plot
    os.makedirs("outputs/metrics", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("DINOv2 ROC Curves - Urban Surface OOD", fontsize=13)

    colors = ["#4B55E3", "#7A44B9", "#B23A70", "#E63737", "#F15B24", "#FF9E22"]
    for cls_id in range(6):
        if cls_id == 4:
            continue
        gt_arr    = np.array(per_class_gt[cls_id])
        score_arr = np.array(per_class_scores[cls_id])
        if len(np.unique(gt_arr)) > 1 and class_aurocs.get(cls_id):
            fpr, tpr, _ = roc_curve(gt_arr, score_arr)
            auroc = class_aurocs[cls_id]
            ax.plot(fpr, tpr, color=colors[cls_id], linewidth=2,
                    label=CLASSES[cls_id] + " (" + str(round(auroc, 3)) + ")")

    ax.plot([0,1],[0,1],"k--", label="Random")
    ax.set_title("Per-Class ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    save_path = "outputs/metrics/auroc_dinov2_urban_surface.png"
    plt.savefig(save_path, dpi=150)
    print("\n[INFO] Saved: " + save_path)
    plt.show()

    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    valid = [v for v in class_aurocs.values() if v]
    if valid:
        print("  Mean AUROC    : " + str(round(float(np.mean(valid)), 4)))
        print("  Best AUROC    : " + str(round(max(valid), 4)))
        print("  Worst AUROC   : " + str(round(min(valid), 4)))
    print("=" * 60)

if __name__ == "__main__":
    main()