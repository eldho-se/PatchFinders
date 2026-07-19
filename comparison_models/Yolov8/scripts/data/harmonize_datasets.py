import os
import shutil
import yaml
from pathlib import Path

# ── Unified class mapping ──────────────────────────────────────────
UNIFIED_CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

# Maps each dataset's class index → our unified class index
DATASET_MAPPINGS = {
    "RDD2022": {
        0: 1,   # D00 → longitudinal_crack
        1: 1,   # D01 → longitudinal_crack
        2: 5,   # D0w0 → surface_deterioration
        3: 2,   # D10 → transverse_crack
        4: 2,   # D11 → transverse_crack
        5: 3,   # D20 → alligator_crack
        6: 0,   # D40 → pothole
        7: 0,   # D43 → pothole
        8: 0,   # D44 → pothole
        9: 5,   # D50 → surface_deterioration
    },
    "CRACK500": {
        0: 1,   # D00 → longitudinal_crack
        1: 2,   # D10 → transverse_crack
        2: 3,   # D20 → alligator_crack
        3: 0,   # D40 → pothole
    },
    "Pothole600": {
        0: 0,   # Lubang → pothole
        1: 5,   # Pelepasan Agregat → surface_deterioration
        2: 1,   # Retak Lain → longitudinal_crack
    }
}

OUTPUT_DIR = "data/processed"

def remap_label_file(src_path, dst_path, class_mapping):
    """Read a YOLO label file, remap class IDs, save to new location."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(src_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_class = int(parts[0])
        if old_class not in class_mapping:
            continue  # Skip unknown classes
        new_class = class_mapping[old_class]
        new_line = f"{new_class} {' '.join(parts[1:])}"
        new_lines.append(new_line)

    if new_lines:
        with open(dst_path, 'w') as f:
            f.write('\n'.join(new_lines))
        return True
    return False

def harmonize_dataset(dataset_name, dataset_dir, split_mapping):
    """Process one dataset into unified format."""
    print(f"\n[INFO] Harmonizing {dataset_name}...")
    class_mapping = DATASET_MAPPINGS[dataset_name]
    counts = {"train": 0, "val": 0, "test": 0}

    for src_split, dst_split in split_mapping.items():
        src_img_dir = os.path.join(dataset_dir, src_split, "images")
        src_lbl_dir = os.path.join(dataset_dir, src_split, "labels")

        if not os.path.exists(src_img_dir):
            print(f"  [SKIP] {src_split} not found")
            continue

        dst_img_dir = os.path.join(OUTPUT_DIR, dst_split, "images")
        dst_lbl_dir = os.path.join(OUTPUT_DIR, dst_split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        images = list(Path(src_img_dir).glob("*.jpg")) + \
                 list(Path(src_img_dir).glob("*.png"))

        saved = 0
        for img_path in images:
            # Copy image with dataset prefix to avoid name conflicts
            new_name = f"{dataset_name}_{img_path.name}"
            dst_img = os.path.join(dst_img_dir, new_name)
            shutil.copy2(str(img_path), dst_img)

            # Remap and copy label
            src_lbl = os.path.join(src_lbl_dir, img_path.stem + ".txt")
            dst_lbl = os.path.join(dst_lbl_dir, 
                                   f"{dataset_name}_{img_path.stem}.txt")

            if os.path.exists(src_lbl):
                ok = remap_label_file(src_lbl, dst_lbl, class_mapping)
                if ok:
                    saved += 1
            else:
                # Create empty label for background images
                open(dst_lbl, 'w').close()
                saved += 1

        counts[dst_split] += saved
        print(f"  {src_split} → {dst_split}: {saved} images")

    return counts

def save_unified_yaml():
    """Save the unified dataset.yaml."""
    config = {
        "path": os.path.abspath(OUTPUT_DIR).replace("\\", "/"),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/in_distribution/images",
        "nc":    6,
        "names": list(UNIFIED_CLASSES.values())
    }
    yaml_path = "configs/dataset.yaml"
    os.makedirs("configs", exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n[INFO] Saved unified config to {yaml_path}")

def main():
    print("=" * 50)
    print("  PatchFinders — Dataset Harmonization")
    print("=" * 50)

    # Clean output directories
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)

    # Harmonize each dataset
    split_map = {"train": "train", "valid": "val", "test": "val"}

    total = {"train": 0, "val": 0, "test": 0}
    datasets = {
        "RDD2022":    "data/raw/RDD2022",
        "CRACK500":   "data/raw/CRACK500",
        "Pothole600": "data/raw/Pothole600",
    }

    for name, path in datasets.items():
        if os.path.exists(path):
            counts = harmonize_dataset(name, path, split_map)
            for k, v in counts.items():
                total[k] += v
        else:
            print(f"[SKIP] {name} not found at {path}")

    # Save unified config
    save_unified_yaml()

    # Summary
    print("\n" + "=" * 50)
    print("  Harmonization Complete!")
    print("=" * 50)
    print(f"  Train images : {total['train']}")
    print(f"  Val images   : {total['val']}")
    print(f"  Total        : {total['train'] + total['val']}")
    print("\n  Class mapping:")
    for idx, name in UNIFIED_CLASSES.items():
        print(f"    {idx}: {name}")

if __name__ == "__main__":
    main()