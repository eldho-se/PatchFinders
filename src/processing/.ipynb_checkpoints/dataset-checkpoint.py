import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class RoadDamageDataset(Dataset):
    def __init__(self, manifest_path, apply_augmentations=False, target_size=(588, 588)):
        self.target_size = target_size
        self.apply_augmentations = apply_augmentations

        # Base folder structure extraction
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(manifest_path))))

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest missing at: {manifest_path}")

        with open(manifest_path, 'r') as f:
            self.data = json.load(f)

        # Standardize transform pipeline
        self.transform = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Quick path sanity check validation check
        if len(self.data) > 0:
            sample_rel_path = self.data[0].get('image_path', '')
            # Try absolute path or join with project root directory
            test_path = sample_rel_path if os.path.isabs(sample_rel_path) else os.path.join(self.base_dir, sample_rel_path)
            if not os.path.exists(test_path):
                print(f"⚠️ PATH WARNING: A sample image wasn't found at: {test_path}")
                print(f"👉 The script will attempt alternative path resolving during execution.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        rel_path = entry.get('image_path', '')

        # Build alternative paths to account for different execution roots
        possible_paths = [
            rel_path,
            os.path.join(self.base_dir, rel_path),
            os.path.abspath(rel_path),
            os.path.join(os.getcwd(), rel_path)
        ]

        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break

        if img_path is None:
            # Explicitly warn the user instead of failing silently!
            if idx % 1000 == 0: 
                print(f"❌ File Not Found: {rel_path} could not be resolved in project workspace.")
            return None

        try:
            # Load and force RGB conversion
            image = Image.open(img_path).convert("RGB")

            # Extract spatial coordinate scaling metrics
            orig_w, orig_h = image.size
            image_tensor = self.transform(image)

            # Normalize target box parameters [0.0 - 1.0 relative bounds]
            boxes = []
            labels = []
            for item in entry.get('annotations', []):
                bbox = item.get('bbox', [0, 0, 0, 0]) # [x_min, y_min, w, h]
                lbl = item.get('category_id', 0)

                # Convert to normalized coordinates relative to center [x_c, y_c, normalized_w, normalized_h]
                x_c = (bbox[0] + (bbox[2] / 2.0)) / orig_w
                y_c = (bbox[1] + (bbox[3] / 2.0)) / orig_h
                w_norm = bbox[2] / orig_w
                h_norm = bbox[3] / orig_h

                boxes.append([x_c, y_c, w_norm, h_norm])
                labels.append(lbl)

            if len(boxes) == 0:
                # Add dummy background box if annotations are empty
                boxes.append([0.5, 0.5, 1.0, 1.0])
                labels.append(2)

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

            return image_tensor, target

        except Exception as e:
            print(f"⚠️ Processing Error on item index {idx}: {str(e)}")
            return None

def collate_fn(batch):
    # Strip away all occurrences of failed None components
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return [], []
    images, targets = zip(*batch)
    return list(images), list(targets)
