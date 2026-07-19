from ultralytics import YOLO
from pathlib import Path
from collections import Counter

CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
    4: "rutting",
    5: "surface_deterioration"
}

if __name__ == "__main__":
    weights = "runs/detect/runs/train/yolov8s_baseline-4/weights/best.pt"
    source  = "data/processed/test/out_of_distribution/urban_surface/test/images"
    conf    = 0.25

    print("=" * 55)
    print("  PatchFinders - YOLOv8s Urban Surface Test")
    print("=" * 55)
    print("[INFO] Loading model: " + weights)

    model = YOLO(weights)

    print("[INFO] Running inference on: " + source)
    results = model.predict(
        source     = source,
        conf       = conf,
        save       = True,
        project    = "outputs/predictions",
        name       = "yolov8s_urban_surface_v2",
        line_width = 2,
        workers    = 0,
        stream     = True,
        imgsz      = 640,
        batch      = 1,
    )

    all_classes    = []
    total_det      = 0
    images_with    = 0
    images_without = 0
    count          = 0

    for r in results:
        count += 1
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            images_with += 1
            for cls in boxes.cls.tolist():
                all_classes.append(CLASSES.get(int(cls), "unknown"))
                total_det += 1
        else:
            images_without += 1
        if count % 20 == 0:
            print("  Processed: " + str(count) + " images...")

    print("\n" + "=" * 55)
    print("  YOLOv8s Urban Surface v2 — Results")
    print("=" * 55)
    print("  Images processed     : " + str(count))
    print("  Images with damage   : " + str(images_with))
    print("  Images no damage     : " + str(images_without))
    print("  Total detections     : " + str(total_det))
    if count > 0:
        print("  Avg per image        : " + str(round(total_det/count, 2)))

    print("\n  Class breakdown:")
    class_counts = Counter(all_classes)
    for cls, cnt in class_counts.most_common():
        bar = "#" * (cnt // 3)
        print("    " + cls.ljust(25) + str(cnt).rjust(5) + "  " + bar)

    print("\n  Saved to: outputs/predictions/yolov8s_urban_surface_v2/")
    print("=" * 55)