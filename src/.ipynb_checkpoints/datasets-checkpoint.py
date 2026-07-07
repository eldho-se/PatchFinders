import os
import glob
import json
import cv2
import numpy as np
import scipy.io as sio

# Core taxonomy matching Section 3.1 of Section2_3_Foundation_Pitch.pptx
UNIFIED_CLASSES = {
    0: "Longitudinal",
    1: "Transverse",
    2: "Alligator",
    3: "Pothole"
}

def get_crack_type_by_geometry(x, y, w, h):
    """Heuristically converts a generic crack mask bounding box into our specific taxonomy[cite: 1]."""
    aspect_ratio = w / float(h) if h > 0 else 1.0
    if aspect_ratio > 2.0:
        return 1  # Transverse (wide/horizontal)[cite: 1]
    elif aspect_ratio < 0.5:
        return 0  # Longitudinal (tall/vertical)[cite: 1]
    else:
        return 2  # Alligator (complex/block patterns)[cite: 1]

def parse_rdd2022(base_dir):
    """Parses native YOLO format from RDD2022 paths[cite: 1]."""
    manifest = []
    labels_path = os.path.join(base_dir, "labels")
    images_path = os.path.join(base_dir, "images")
    
    for txt_path in glob.glob(os.path.join(labels_path, "*.txt")):
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        img_path = os.path.join(images_path, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            continue
            
        annotations = []
        with open(txt_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    if cls_id in UNIFIED_CLASSES:
                        annotations.append({
                            "category_id": cls_id,
                            "bbox_yolo": [float(x) for x in parts[1:]]
                        })
        if annotations:
            manifest.append({"image_path": img_path, "annotations": annotations, "domain": "RDD2022_dashcam"})
    return manifest

def parse_crack500(data_dir):
    """Converts Crack500 image masks into bounding boxes and infers category[cite: 1]."""
    manifest = []
    # Search for raw source images
    img_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    
    for img_path in img_paths:
        base_name = os.path.splitext(img_path)[0]
        mask_path = f"{base_name}_mask.png"
        
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        img_h, img_w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotations = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100: # Filter out small pixel noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Normalize to YOLO format: [x_center, y_center, width, height]
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            norm_w = w / float(img_w)
            norm_h = h / float(img_h)
            
            cat_id = get_crack_type_by_geometry(x, y, w, h)
            annotations.append({
                "category_id": cat_id,
                "bbox_yolo": [x_center, y_center, norm_w, norm_h]
            })
            
        if annotations:
            manifest.append({"image_path": img_path, "annotations": annotations, "domain": "Crack500_OOD"})
    return manifest

def parse_crackforest(base_dir):
    """Extracts ground-truth pixel matrix data from MATLAB structures[cite: 1]."""
    manifest = []
    img_dir = os.path.join(base_dir, "image")
    gt_dir = os.path.join(base_dir, "groundTruth")
    
    for img_path in glob.glob(os.path.join(img_dir, "*.jpg")):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mat_path = os.path.join(gt_dir, f"{base_name}.mat")
        
        if not os.path.exists(mat_path):
            continue
            
        try:
            mat_contents = sio.loadmat(mat_path)
            # CrackForest wraps segmentation arrays inside a nested groundTruth reference cell
            mask = mat_contents['groundTruth'][0, 0]['Segmentation']
            
            img_h, img_w = mask.shape[:2]
            # Find coordinates where crack pixel value == 1
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            annotations = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 50:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                norm_w = w / float(img_w)
                norm_h = h / float(img_h)
                
                cat_id = get_crack_type_by_geometry(x, y, w, h)
                annotations.append({
                    "category_id": cat_id,
                    "bbox_yolo": [x_center, y_center, norm_w, norm_h]
                })
                
            if annotations:
                manifest.append({"image_path": img_path, "annotations": annotations, "domain": "CrackForest_OOD"})
        except Exception as e:
            print(f"⚠️ Warning: Could not parse mat file {mat_path}. Error: {e}")
            
    return manifest

def run_harmonization_pipeline():
    print("🚀 Running master domain harmonization pipeline...")
    
    # Process Train Source Data[cite: 1]
    train_data = parse_rdd2022("data/raw/RDD2022/train")
    
    # Process OOD Target Evaluation Sets[cite: 1]
    cf_data = parse_crackforest("data/raw/CrackForest/CrackForest-dataset-master")
    c500_data = parse_crack500("data/raw/Crack500/traindata/traindata") # Using primary sample cluster
    
    ood_data = cf_data + c500_data
    
    # Export Manifest files
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/train_harmonized.json", "w") as f:
        json.dump(train_data, f, indent=4)
    with open("data/processed/ood_harmonized.json", "w") as f:
        json.dump(ood_data, f, indent=4)
        
    print(f"🎉 Harmonization complete!")
    print(f"   - Source Train Pool: {len(train_data)} images mapped.")
    print(f"   - Target OOD Evaluation Pool: {len(ood_data)} images mapped.")

if __name__ == "__main__":
    run_harmonization_pipeline()