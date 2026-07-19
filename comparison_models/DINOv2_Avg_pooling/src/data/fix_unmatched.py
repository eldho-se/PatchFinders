from pathlib import Path
import os

def fix_unmatched(split):
    img_dir = f"data/processed/{split}/images"
    lbl_dir = f"data/processed/{split}/labels"
    images  = list(Path(img_dir).glob("*.jpg"))
    images += list(Path(img_dir).glob("*.png"))
    fixed   = 0
    for img_path in images:
        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            open(lbl_path, 'w').close()
            fixed += 1
    print(f"  {split}: created {fixed} empty label files")

print("Fixing unmatched pairs...")
fix_unmatched("train")
fix_unmatched("val")
print("Done!")