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
    """Load ground truth labels from YOLO format."""
    gt = {}
    label_path = Path(label_dir)
    if not label_path.exists():
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

def run_auroc_evaluation(
    weights,
    img_dir,
    label_dir,
    conf_threshold=0.001,
    save_dir="outputs/metrics"
):
    print("=" * 60)
    print("  PatchFinders — AUROC Evaluation")
    print("=" * 60)
    print(f"  Weights  : {weights}")
    print(f"  Images   : {img_dir}")
    print(f"  Labels   : {label_dir}")
    print("=" * 60)

    # Load model
    model = YOLO(weights)

    # Load ground truth
    gt_labels = load_ground_truth(label_dir)
    print(f"\n[INFO] Loaded {len(gt_labels)} ground truth files")

    # Get all images
    img_path = Path(img_dir)
    images   = list(img_path.glob("*.jpg"))
    images  += list(img_path.glob("*.JPG"))
    images  += list(img_path.glob("*.jpeg"))
    images  += list(img_path.glob("*.png"))
    print(f"[INFO] Found {len(images)} images")

    # Run inference with very low confidence to get all predictions
    print("[INFO] Running inference...")
    results = model.predict(
        source     = img_dir,
        conf       = conf_threshold,
        workers    = 0,
        stream     = True,
        imgsz      = 640,
        batch      = 1,
        verbose    = False,
    )

    # Collect per-image results
    all_gt_binary     = []  # 1 if image has any damage, 0 if not
    all_pred_scores   = []  # max confidence score per image
    per_class_gt      = defaultdict(list)
    per_class_scores  = defaultdict(list)

    image_results = {}
    for r in results:
        img_name = Path(r.path).stem
        boxes    = r.boxes

        # Get max confidence score for this image
        if boxes is not None and len(boxes) > 0:
            scores = boxes.conf.tolist()
            classes = boxes.cls.tolist()
            max_score = max(scores)

            # Per class scores
            class_scores = defaultdict(float)
            for score, cls in zip(scores, classes):
                cls = int(cls)
                class_scores[cls] = max(class_scores[cls], score)
        else:
            max_score    = 0.0
            class_scores = {}

        image_results[img_name] = {
            "max_score":    max_score,
            "class_scores": class_scores
        }

    # Build binary labels and scores
    for img_name, pred in image_results.items():
        # Ground truth: does this image have any damage?
        gt = gt_labels.get(img_name, [])
        has_damage = 1 if len(gt) > 0 else 0
        all_gt_binary.append(has_damage)
        all_pred_scores.append(pred["max_score"])

        # Per-class
        for cls_id in range(6):
            gt_has_class = 1 if cls_id in gt else 0
            pred_score   = pred["class_scores"].get(cls_id, 0.0)
            per_class_gt[cls_id].append(gt_has_class)
            per_class_scores[cls_id].append(pred_score)

    # Calculate overall AUROC
    all_gt_binary   = np.array(all_gt_binary)
    all_pred_scores = np.array(all_pred_scores)

    print("\n" + "=" * 60)
    print("  AUROC Results")
    print("=" * 60)

    # Overall AUROC
    if len(np.unique(all_gt_binary)) > 1:
        overall_auroc = roc_auc_score(all_gt_binary, all_pred_scores)
        overall_ap    = average_precision_score(all_gt_binary, all_pred_scores)
        print(f"\n  Overall AUROC : {overall_auroc:.4f}")
        print(f"  Average Prec  : {overall_ap:.4f}")
    else:
        print("\n  [WARN] Not enough GT variety for overall AUROC")
        overall_auroc = 0.0

    # Per-class AUROC
    print("\n  Per-Class AUROC:")
    print("  " + "-" * 45)
    class_aurocs = {}
    for cls_id in range(6):
        gt_arr    = np.array(per_class_gt[cls_id])
        score_arr = np.array(per_class_scores[cls_id])

        if len(np.unique(gt_arr)) > 1:
            auroc = roc_auc_score(gt_arr, score_arr)
            ap    = average_precision_score(gt_arr, score_arr)
            class_aurocs[cls_id] = auroc
            bar = "█" * int(auroc * 20)
            print(f"  {CLASSES[cls_id]:<25} AUROC: {auroc:.4f}  AP: {ap:.4f}  {bar}")
        else:
            class_aurocs[cls_id] = 0.0
            print(f"  {CLASSES[cls_id]:<25} AUROC: N/A (no GT samples)")

    # Plot ROC Curves
    print("\n[INFO] Plotting ROC curves...")
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PatchFinders — ROC Curves (Urban Surface OOD Test)", fontsize=13)

    # Overall ROC
    if len(np.unique(all_gt_binary)) > 1:
        fpr, tpr, _ = roc_curve(all_gt_binary, all_pred_scores)
        axes[0].plot(fpr, tpr, color="blue", linewidth=2,
                     label=f"Overall (AUROC={overall_auroc:.3f})")
        axes[0].plot([0,1],[0,1], "k--", linewidth=1, label="Random")
        axes[0].set_title("Overall ROC Curve")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend()
        axes[0].grid(True)

    # Per-class ROC
    colors = ["red", "green", "blue", "orange", "purple", "cyan"]
    for cls_id in range(6):
        gt_arr    = np.array(per_class_gt[cls_id])
        score_arr = np.array(per_class_scores[cls_id])
        if len(np.unique(gt_arr)) > 1:
            fpr, tpr, _ = roc_curve(gt_arr, score_arr)
            auroc = class_aurocs[cls_id]
            axes[1].plot(fpr, tpr, color=colors[cls_id], linewidth=2,
                         label=f"{CLASSES[cls_id]} ({auroc:.3f})")

    axes[1].plot([0,1],[0,1], "k--", linewidth=1, label="Random")
    axes[1].set_title("Per-Class ROC Curves")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "auroc_urban_surface.png")
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Saved ROC plot to: {save_path}")
    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("  AUROC Summary")
    print("=" * 60)
    print(f"  Overall AUROC    : {overall_auroc:.4f}")
    valid_aurocs = [v for v in class_aurocs.values() if v > 0]
    if valid_aurocs:
        print(f"  Mean Class AUROC : {np.mean(valid_aurocs):.4f}")
        print(f"  Best Class AUROC : {max(valid_aurocs):.4f}")
        print(f"  Worst Class AUROC: {min(valid_aurocs):.4f}")
    print("=" * 60)

    return overall_auroc, class_aurocs

if __name__ == "__main__":
    # YOLOv8s on Urban Surface
    run_auroc_evaluation(
        weights   = "runs/detect/runs/train/yolov8s_baseline-4/weights/best.pt",
        img_dir   = "data/processed/test/out_of_distribution/urban_surface/test/images",
        label_dir = "data/processed/test/out_of_distribution/urban_surface/test/labels",
        save_dir  = "outputs/metrics"
    )