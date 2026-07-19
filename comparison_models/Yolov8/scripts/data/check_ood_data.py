import os

OOD_SETS = {
    "in_distribution":      "data/processed/test/in_distribution/images",
    "pedestrian_viewpoint": "data/processed/test/out_of_distribution/pedestrian_viewpoint/images",
    "cobblestone":          "data/processed/test/out_of_distribution/cobblestone/images",
    "dirt_path":            "data/processed/test/out_of_distribution/dirt_path/images",
    "urban_surface":        "data/processed/test/out_of_distribution/urban_surface/images",
}

def check_ood_data():
    print("=" * 55)
    print("  PatchFinders — OOD Dataset Status")
    print("=" * 55)

    ready   = []
    missing = []

    for name, path in OOD_SETS.items():
        if os.path.exists(path):
            images = [f for f in os.listdir(path)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
            count  = len(images)
            if count > 0:
                print(f"  ✅ {name:<30} {count} images")
                ready.append(name)
            else:
                print(f"  ❌ {name:<30} 0 images (empty)")
                missing.append(name)
        else:
            print(f"  ❌ {name:<30} folder not found")
            missing.append(name)

    print("\n" + "=" * 55)
    print(f"  Ready   : {len(ready)} sets")
    print(f"  Missing : {len(missing)} sets")
    print("=" * 55)

    if missing:
        print("\n  [!] Missing OOD sets:")
        for name in missing:
            print(f"      - {name}")
        print("\n  You need to collect images for missing sets.")
        print("  Options:")
        print("  1. Download from Roboflow Universe")
        print("  2. Collect your own photos")
        print("  3. Use val set as temporary test set")

    if ready:
        print("\n  [✅] Ready to evaluate on:")
        for name in ready:
            print(f"      - {name}")

if __name__ == "__main__":
    check_ood_data()