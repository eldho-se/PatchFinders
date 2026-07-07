import os
import sys
import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.models.dino_detector import DinoMultiBoxDetector

def yolo_to_pixels(box_yolo, w_orig, h_orig):
    x_c, y_c, w, h = box_yolo
    x1 = int((x_c - w / 2) * w_orig)
    y1 = int((y_c - h / 2) * h_orig)
    x2 = int((x_c + w / 2) * w_orig)
    y2 = int((y_c + h / 2) * h_orig)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_orig - 1, x2), min(h_orig - 1, y2)
    return x1, y1, x2, y2

def aggregate_boxes(boxes, scores, classes, margin=20):
    n = len(boxes)
    if n == 0:
        return [], [], []
        
    def check_overlap(b1, b2):
        return not (b1[2] + margin < b2[0] or b1[0] - margin > b2[2] or 
                    b1[3] + margin < b2[1] or b1[1] - margin > b2[3])
        
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if classes[i] == classes[j]:
                if check_overlap(boxes[i], boxes[j]):
                    adj[i].append(j)
                    adj[j].append(i)
                    
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
            
    merged_boxes = []
    merged_scores = []
    merged_classes = []
    
    for comp in components:
        comp_boxes = [boxes[idx] for idx in comp]
        comp_scores = [scores[idx] for idx in comp]
        comp_classes = [classes[idx] for idx in comp]
        
        x1 = min(b[0] for b in comp_boxes)
        y1 = min(b[1] for b in comp_boxes)
        x2 = max(b[2] for b in comp_boxes)
        y2 = max(b[3] for b in comp_boxes)
        
        score = max(comp_scores)
        cls_id = comp_classes[0]
        
        merged_boxes.append([x1, y1, x2, y2])
        merged_scores.append(score)
        merged_classes.append(cls_id)
        
    return merged_boxes, merged_scores, merged_classes

def predict_and_compare(target_image_path, threshold=0.50, proximity_margin=20, save_output_path="prediction_comparison.png"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Target Device Backend: {device.type.upper()}")
    
    manifest_path = os.path.join(project_root, "data/processed/urbansurface_harmonized.json")
    checkpoint_path = os.path.join(project_root, "models/checkpoints/dino_probing_heads.pt")
    
    entry = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            dataset_meta = json.load(f)
        target_norm = os.path.normpath(target_image_path)
        for item in dataset_meta:
            if os.path.normpath(item["image_path"]) == target_norm:
                entry = item
                break
                
    orig_img = cv2.imread(target_image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Error: Could not read image file at: {target_image_path}")
        
    h_orig, w_orig = orig_img.shape[:2]
    gt_img = orig_img.copy()
    pred_img = orig_img.copy()
    
    class_names = {0: "Crack", 1: "Pothole"}
    gt_color = (0, 255, 0)      
    pred_color = (255, 0, 0)    
    
    if entry is not None:
        print(f"Found ground truth metadata for {Path(target_image_path).name}")
        for ann in entry["annotations"]:
            cls_id = ann["category_id"]
            if cls_id >= 2:
                continue
            bbox_yolo = ann["bbox_yolo"]
            x1, y1, x2, y2 = yolo_to_pixels(bbox_yolo, w_orig, h_orig)
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), gt_color, 4)
            label_text = f"GT: {class_names[cls_id]}"
            cv2.putText(gt_img, label_text, (x1, max(30, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, gt_color, 2)
    else:
        print(f"Warning: No ground truth metadata found for {target_image_path}")
        
    model = DinoMultiBoxDetector(num_classes=2, grid_size=42).to(device)
    if os.path.exists(checkpoint_path):
        trainable_weights = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(trainable_weights, strict=False)
        print("Checkpoint weights loaded.")
    else:
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Running with randomized weights.")
    model.eval()
    
    img_resized = cv2.resize(orig_img, (588, 588), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_tensor = (img_tensor / 255.0 - mean) / std
    
    with torch.no_grad():
        class_grid, box_grid = model(img_tensor)
        probs_grid = F.softmax(class_grid, dim=-1).squeeze(0) 
        box_grid = box_grid.squeeze(0)                      
        
    candidate_boxes = []
    candidate_scores = []
    candidate_classes = []
    
    for y in range(42):
        for x in range(42):
            probs = probs_grid[y, x].tolist()
            fg_probs = probs[:2]
            cls_id = int(np.argmax(fg_probs))
            max_prob = fg_probs[cls_id]
            
            if max_prob > threshold:
                box_yolo = box_grid[y, x].tolist()
                x1, y1, x2, y2 = yolo_to_pixels(box_yolo, w_orig, h_orig)
                candidate_boxes.append([x1, y1, x2, y2])
                candidate_scores.append(max_prob)
                candidate_classes.append(cls_id)
                
    merged_boxes, merged_scores, merged_classes = aggregate_boxes(
        candidate_boxes, candidate_scores, candidate_classes, margin=proximity_margin
    )
    
    for idx in range(len(merged_boxes)):
        x1, y1, x2, y2 = merged_boxes[idx]
        cls_id = merged_classes[idx]
        score = merged_scores[idx]
        
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), pred_color, 4)
        label_text = f"Pred: {class_names[cls_id]} {score:.2f}"
        cv2.putText(pred_img, label_text, (x1, max(30, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
                    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Ground Truth - ({Path(target_image_path).name})")
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Aggregated Predictions (Proximity < {proximity_margin}px) - Found {len(merged_boxes)}")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_output_path, dpi=150)
    print(f"Comparison plot saved to: {save_output_path}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_aggregated.py <image_path> [threshold] [proximity_margin] [save_output_path]")
        sys.exit(1)
        
    image_path = sys.argv[1]
    conf_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
    margin = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    output_path = sys.argv[4] if len(sys.argv) > 4 else "prediction_comparison.png"
    
    predict_and_compare(image_path, conf_thresh, margin, output_path)