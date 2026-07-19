import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from collections import defaultdict

CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

def load_ground_truth(label_dir):
    gt = {}
    for lbl_file in Path(label_dir).glob("*.txt"):
        classes = []
        with open(lbl_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    cls = int(line.split()[0])
                    classes.append(cls)
        gt[lbl_file.stem] = classes
    return gt

def run_auroc(weights, img_dir, label_dir, name, save_dir="outputs/metrics"):
    print(f"\n[INFO] Running AUROC for: {name}")
    model     = YOLO(weights)
    gt_labels = load_ground_truth(label_dir)

    results = model.predict(
        source  = img_dir,
        conf    = 0.001,
        workers = 0,
        stream  = True,
        imgsz   = 640,
        batch   = 1,
        verbose = False,
    )

    all_gt    = []
    all_scores = []
    per_class_gt     = defaultdict(list)
    per_class_scores = defaultdict(list)

    image_results = {}
    for r in results:
        img_name = Path(r.path).stem
        boxes    = r.boxes
        if boxes is not None and len(boxes) > 0:
            scores  = boxes.conf.tolist()
            classes = boxes.cls.tolist()
            max_score    = max(scores)
            class_scores = defaultdict(float)
            for s, c in zip(scores, classes):
                c = int(c)
                class_scores[c] = max(class_scores[c], s)
        else:
            max_score    = 0.0
            class_scores = {}
        image_results[img_name] = {
            "max_score":    max_score,
            "class_scores": class_scores
        }

    for img_name, pred in image_results.items():
        gt         = gt_labels.get(img_name, [])
        has_damage = 1 if len(gt) > 0 else 0
        all_gt.append(has_damage)
        all_scores.append(pred["max_score"])
        for cls_id in range(6):
            per_class_gt[cls_id].append(1 if cls_id in gt else 0)
            per_class_scores[cls_id].append(pred["class_scores"].get(cls_id, 0.0))

    all_gt    = np.array(all_gt)
    all_scores = np.array(all_scores)

    overall_auroc = 0.0
    if len(np.unique(all_gt)) > 1:
        overall_auroc = roc_auc_score(all_gt, all_scores)

    class_aurocs = {}
    for cls_id in range(6):
        gt_arr    = np.array(per_class_gt[cls_id])
        score_arr = np.array(per_class_scores[cls_id])
        if len(np.unique(gt_arr)) > 1:
            class_aurocs[cls_id] = roc_auc_score(gt_arr, score_arr)
        else:
            class_aurocs[cls_id] = None

    return overall_auroc, class_aurocs

if __name__ == "__main__":
    weights = "runs/detect/runs/train/yolov8s_baseline-4/weights/best.pt"

    # In-distribution
    id_auroc, id_class = run_auroc(
        weights   = weights,
        img_dir   = "data/processed/val/images",
        label_dir = "data/processed/val/labels",
        name      = "In-Distribution (Val Set)"
    )

    # OOD
    ood_auroc, ood_class = run_auroc(
        weights   = weights,
        img_dir   = "data/processed/test/out_of_distribution/urban_surface/test/images",
        label_dir = "data/processed/test/out_of_distribution/urban_surface/test/labels",
        name      = "OOD (Urban Surface)"
    )

    # Comparison
    print("\n" + "=" * 60)
    print("  AUROC Comparison: In-Distribution vs OOD")
    print("=" * 60)
    print(f"  {'Metric':<30} {'In-Dist':>10} {'OOD':>10} {'Drop':>10}")
    print("  " + "-" * 55)
    print(f"  {'Overall AUROC':<30} {id_auroc:>10.4f} {ood_auroc:>10.4f} {(ood_auroc-id_auroc):>+10.4f}")

    print("\n  Per-Class:")
    for cls_id in range(6):
        id_val  = id_class.get(cls_id)
        ood_val = ood_class.get(cls_id)
        name    = CLASSES[cls_id]
        if id_val and ood_val:
            drop = ood_val - id_val
            print(f"  {name:<30} {id_val:>10.4f} {ood_val:>10.4f} {drop:>+10.4f}")
        else:
            print(f"  {name:<30} {'N/A':>10} {'N/A':>10}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    classes_list = [CLASSES[i] for i in range(6)]
    id_vals  = [id_class.get(i, 0) or 0 for i in range(6)]
    ood_vals = [ood_class.get(i, 0) or 0 for i in range(6)]

    x = np.arange(len(classes_list))
    w = 0.35
    ax.bar(x - w/2, id_vals,  w, label="In-Distribution", color="steelblue")
    ax.bar(x + w/2, ood_vals, w, label="OOD Urban Surface", color="tomato")
    ax.axhline(y=0.5, color="black", linestyle="--", label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes_list, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC: In-Distribution vs OOD (Urban Surface)")
    ax.legend()
    ax.grid(axis="y")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("outputs/metrics/auroc_comparison.png", dpi=150)
    print("\n[INFO] Saved to outputs/metrics/auroc_comparison.png")
    plt.show()
    print("=" * 60)