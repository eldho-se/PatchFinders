import os
import shutil
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

DATASET_MAPPINGS = {
    "RDD2022": {0:1, 1:1, 2:5, 3:2, 4:2, 5:3, 6:0, 7:0, 8:0, 9:5},
    "CRACK500": {0:1, 1:2, 2:3, 3:0},
    "Pothole600": {0:0, 1:5, 2:1}
}

OUTPUT_DIR = "data/processed"


def remap_label_file(src_path, dst_path, class_mapping):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_class = int(parts[0])
        if old_class not in class_mapping:
            continue
        new_class = class_mapping[old_class]
        new_lines.append(str(new_class) + " " + " ".join(parts[1:]))
    if new_lines:
        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines))
        return True
    return False


def harmonize_dataset(dataset_name, dataset_dir, split_mapping):
    print("[INFO] Harmonizing " + dataset_name)
    class_mapping = DATASET_MAPPINGS[dataset_name]
    counts = {"train": 0, "val": 0}

    for src_split, dst_split in split_mapping.items():
        src_img_dir = os.path.join(dataset_dir, src_split, "images")
        src_lbl_dir = os.path.join(dataset_dir, src_split, "labels")

        if not os.path.exists(src_img_dir):
            print("  [SKIP] " + src_split + " not found")
            continue

        dst_img_dir = os.path.join(OUTPUT_DIR, dst_split, "images")
        dst_lbl_dir = os.path.join(OUTPUT_DIR, dst_split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        images = list(Path(src_img_dir).glob("*.jpg"))
        images += list(Path(src_img_dir).glob("*.png"))

        saved = 0
        for img_path in images:
            new_name = dataset_name + "_" + img_path.name
            shutil.copy2(str(img_path), os.path.join(dst_img_dir, new_name))

            src_lbl = os.path.join(src_lbl_dir, img_path.stem + ".txt")
            dst_lbl = os.path.join(dst_lbl_dir, dataset_name + "_" + img_path.stem + ".txt")

            if os.path.exists(src_lbl):
                remap_label_file(src_lbl, dst_lbl, class_mapping)
            else:
                open(dst_lbl, "w").close()
            saved += 1

        counts[dst_split] += saved
        print("  " + src_split + " -> " + dst_split + ": " + str(saved) + " images")

    return counts


def main():
    print("=" * 50)
    print("  PatchFinders - Dataset Harmonization")
    print("=" * 50)

    split_map = {"train": "train", "valid": "val", "test": "val"}
    datasets = {
        "RDD2022": "data/raw/RDD2022",
        "CRACK500": "data/raw/CRACK500",
        "Pothole600": "data/raw/Pothole600"
    }

    total = {"train": 0, "val": 0}
    for name, path in datasets.items():
        if os.path.exists(path):
            counts = harmonize_dataset(name, path, split_map)
            for k, v in counts.items():
                total[k] += v
        else:
            print("[SKIP] " + name + " not found at " + path)

    os.makedirs("configs", exist_ok=True)
    config = {
        "path": OUTPUT_DIR,
        "train": "train/images",
        "val": "val/images",
        "test": "test/in_distribution/images",
        "nc": 6,
        "names": list(UNIFIED_CLASSES.values())
    }
    with open("configs/dataset.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("[INFO] Saved configs/dataset.yaml")
    print("=" * 50)
    print("  Harmonization Complete!")
    print("  Train : " + str(total["train"]) + " images")
    print("  Val   : " + str(total["val"]) + " images")
    print("  Total : " + str(total["train"] + total["val"]) + " images")
    print("=" * 50)


if __name__ == "__main__":
    main()