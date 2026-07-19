import os
import yaml
from pathlib import Path

UNIFIED_CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

def check_harmonization():
    print("=" * 50)
    print("  PatchFinders — Harmonization Verification")
    print("=" * 50)

    # 1. Check folder structure
    print("\n[1] Checking folder structure...")
    folders = [
        "data/processed/train/images",
        "data/processed/train/labels",
        "data/processed/val/images",
        "data/processed/val/labels",
    ]
    for folder in folders:
        count = len(list(Path(folder).glob("*"))) if os.path.exists(folder) else 0
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {folder}: {count} files")

    # 2. Check dataset.yaml
    print("\n[2] Checking configs/dataset.yaml...")
    yaml_path = "configs/dataset.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  ✅ Classes : {config.get('nc')}")
        print(f"  ✅ Names   : {config.get('names')}")
        print(f"  ✅ Train   : {config.get('train')}")
        print(f"  ✅ Val     : {config.get('val')}")
    else:
        print("  ❌ dataset.yaml not found!")

    # 3. Check class distribution
    print("\n[3] Checking class distribution in train labels...")
    label_dir = "data/processed/train/labels"
    class_counts = {i: 0 for i in range(6)}
    empty_files = 0
    total_files = 0

    if os.path.exists(label_dir):
        for label_file in Path(label_dir).glob("*.txt"):
            total_files += 1
            with open(label_file, 'r') as f:
                lines = f.readlines()
            if not lines:
                empty_files += 1
                continue
            for line in lines:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    print(f"  Total label files : {total_files}")
    print(f"  Empty label files : {empty_files}")
    print(f"\n  Class distribution:")
    for class_id, count in class_counts.items():
        bar = "█" * (count // 500)
        print(f"    {class_id} {UNIFIED_CLASSES[class_id]:<25} {count:>6} {bar}")

    # 4. Check image-label pairs match
    print("\n[4] Checking image-label pairs match...")
    for split in ["train", "val"]:
        img_dir = f"data/processed/{split}/images"
        lbl_dir = f"data/processed/{split}/labels"
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            imgs = set(Path(img_dir).glob("*.jpg"))
            imgs.update(Path(img_dir).glob("*.png"))
            lbls = set(Path(lbl_dir).glob("*.txt"))
            img_stems = {p.stem for p in imgs}
            lbl_stems = {p.stem for p in lbls}
            matched = len(img_stems & lbl_stems)
            unmatched = len(img_stems - lbl_stems)
            status = "✅" if unmatched == 0 else "⚠️"
            print(f"  {status} {split}: {matched} matched pairs, {unmatched} unmatched")

    # 5. Sample label check
    print("\n[5] Sample label content check...")
    label_dir = "data/processed/train/labels"
    if os.path.exists(label_dir):
        samples = [f for f in Path(label_dir).glob("*.txt")
                   if os.path.getsize(f) > 0][:3]
        for sample in samples:
            with open(sample, 'r') as f:
                content = f.read().strip()
            first_line = content.split('\n')[0]
            class_id = int(first_line.split()[0])
            class_name = UNIFIED_CLASSES.get(class_id, "unknown")
            print(f"  📄 {sample.name}")
            print(f"     {first_line} → class: {class_name}")

    print("\n" + "=" * 50)
    print("  Verification Complete!")
    print("=" * 50)

if __name__ == "__main__":
    check_harmonization()