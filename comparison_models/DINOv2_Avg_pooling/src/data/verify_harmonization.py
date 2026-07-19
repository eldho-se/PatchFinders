import os, yaml
from pathlib import Path

CLASSES = {0:"pothole", 1:"longitudinal_crack", 2:"transverse_crack",
           3:"alligator_crack", 4:"rutting", 5:"surface_deterioration"}

print("=" * 50)
print("  Harmonization Verification")
print("=" * 50)

# Folder check
print("\n[1] Folder structure...")
for folder in ["data/processed/train/images","data/processed/train/labels",
               "data/processed/val/images","data/processed/val/labels"]:
    count  = len(list(Path(folder).glob("*"))) if os.path.exists(folder) else 0
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {folder}: {count} files")

# Class distribution
print("\n[2] Class distribution...")
counts = {i:0 for i in range(6)}
total  = 0
for lf in Path("data/processed/train/labels").glob("*.txt"):
    total += 1
    for line in open(lf).readlines():
        line = line.strip()
        if line:
            cid = int(line.split()[0])
            if cid in counts:
                counts[cid] += 1
print(f"  Total label files: {total}")
for cid, cnt in counts.items():
    bar = "█" * (cnt // 500)
    print(f"  {cid} {CLASSES[cid]:<25} {cnt:>6} {bar}")

# Pair matching
print("\n[3] Image-label matching...")
for split in ["train", "val"]:
    imgs = {p.stem for p in Path(f"data/processed/{split}/images").glob("*.jpg")}
    lbls = {p.stem for p in Path(f"data/processed/{split}/labels").glob("*.txt")}
    matched   = len(imgs & lbls)
    unmatched = len(imgs - lbls)
    status    = "✅" if unmatched == 0 else "⚠️"
    print(f"  {status} {split}: {matched} matched, {unmatched} unmatched")

print("\n✅ Verification Complete!")