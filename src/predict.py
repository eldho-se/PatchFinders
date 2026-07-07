import importlib.util
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def predict_on_image(
    image_path, output_path="prediction_output.jpg", threshold=0.65
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    spec_md = importlib.util.spec_from_file_location(
        "dino_detector", "src/models/dino_detector.py"
    )
    md_mod = importlib.util.module_from_spec(spec_md)
    spec_md.loader.exec_module(md_mod)

    model = md_mod.DinoMultiBoxDetector(num_classes=2, grid_size=42).to(device)

    checkpoint_path = "models/checkpoints/dino_probing_heads.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=False
        )
        print(f"Loaded trained weights from {checkpoint_path}")
    else:
        print(
            "Warning: No checkpoint found! Running with randomized probing heads."
        )

    model.eval()

    orig_img = cv2.imread(image_path)
    h_orig, w_orig = orig_img.shape[:2]

    img_resized = cv2.resize(orig_img, (588, 588), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_tensor = (
        torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )
    img_tensor.div_(255.0)

    with torch.no_grad():
        class_grid, box_grid = model(img_tensor)
        probs_grid = F.softmax(class_grid, dim=-1).squeeze(0)
        box_grid = box_grid.squeeze(0)

    class_names = {0: "Crack", 1: "Pothole"}
    colors = {0: (0, 255, 255), 1: (0, 0, 255)}

    detections_found = 0
    for y in range(42):
        for x in range(42):
            probs = probs_grid[y, x].tolist()
            fg_probs = probs[:2]
            cls_id = int(np.argmax(fg_probs))
            max_prob = fg_probs[cls_id]

            if max_prob > threshold and cls_id < 2:
                detections_found += 1
                x_c, y_c, w, h = box_grid[y, x].tolist()

                x1 = int((x_c - w / 2) * w_orig)
                y1 = int((y_c - h / 2) * h_orig)
                x2 = int((x_c + w / 2) * w_orig)
                y2 = int((y_c + h / 2) * h_orig)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_orig - 1, x2), min(h_orig - 1, y2)

                cv2.rectangle(orig_img, (x1, y1), (x2, y2), colors[cls_id], 2)
                label_text = f"{class_names[cls_id]}: {max_prob:.2f}"
                cv2.putText(
                    orig_img,
                    label_text,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors[cls_id],
                    2,
                )

    cv2.imwrite(output_path, orig_img)
    print(
        f"Processed image! Found {detections_found} anomalies. Saved visualization to: {output_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_on_image(sys.argv[1])
    else:
        print(
            "Please provide an image path. Example: python src/predict.py path/to/image.jpg"
        )