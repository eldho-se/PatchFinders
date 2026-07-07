import os
import json
from pathlib import Path

def harmonize_urban_surface(project_root_str="."):
    project_root = Path(project_root_str).resolve()
    
    raw_dir = project_root / "data" / "raw" / "UrbanSurface"
    images_dir = raw_dir / "Images"
    labels_dir = raw_dir / "Labels"
    output_dir = project_root / "data" / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "urbansurface_harmonized.json"
    
    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    
    harmonized_data = []
    
    if not labels_dir.exists():
        print(f"Error: Labels directory not found at {labels_dir}")
        return
        
    print(f"Scanning labels in {labels_dir}...")
    
    label_files = sorted(list(labels_dir.glob("*.txt")))
    for label_path in label_files:
        base_name = label_path.stem  
        
        image_path = None
        for ext in image_extensions:
            candidate = images_dir / f"{base_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break
                
        if image_path is None:
            print(f"Warning: Image file not found for label {label_path.name}")
            continue
            
        rel_image_path = os.path.relpath(image_path, project_root)
        
        annotations = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        x_c = float(parts[1])
                        y_c = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        
                        annotations.append({
                            "category_id": class_id,
                            "bbox_yolo": [x_c, y_c, w, h]
                        })
                    except ValueError:
                        print(f"Warning: Could not parse line in {label_path.name}: '{line.strip()}'")
                elif parts:
                    print(f"Warning: Unexpected number of values in {label_path.name}: '{line.strip()}'")
                    
        entry = {
            "image_path": rel_image_path,
            "annotations": annotations,
            "domain": "UrbanSurface"
        }
        harmonized_data.append(entry)
        
    with open(output_file, "w") as f:
        json.dump(harmonized_data, f, indent=4)
        
    print(f"Successfully harmonized {len(harmonized_data)} images.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    harmonize_urban_surface(project_root)