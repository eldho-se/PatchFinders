import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import importlib.util
import numpy as np

def train_dino_pipeline():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training Engine Initialized on Device Backend: {device.type.upper()}")

    spec_ds = importlib.util.spec_from_file_location("dataset", "src/processing/dataset.py")
    ds_mod = importlib.util.module_from_spec(spec_ds)
    spec_ds.loader.exec_module(ds_mod)

    spec_md = importlib.util.spec_from_file_location("dino_detector", "src/models/dino_detector.py")
    md_mod = importlib.util.module_from_spec(spec_md)
    spec_md.loader.exec_module(md_mod)

    train_dataset = ds_mod.RoadDamageDataset(
        manifest_path="data/processed/train_harmonized.json",
        apply_augmentations=True,
        target_size=(588, 588)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=ds_mod.collate_fn,
        num_workers=2
    )

    model = md_mod.DinoMultiBoxDetector(num_classes=2, grid_size=42).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-4)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_box = nn.MSELoss()

    print("Beginning Training Run Validation Loop...")
    model.train()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images_tensor = torch.stack(images).to(device)
        batch_size = images_tensor.size(0)

        tgt_cls = torch.full((batch_size, 42, 42), 2, dtype=torch.long, device=device)
        tgt_box = torch.zeros((batch_size, 42, 42, 4), dtype=torch.float32, device=device)

        for b in range(batch_size):
            boxes = targets[b]["boxes"]
            labels = targets[b]["labels"]

            for box, label in zip(boxes, labels):
                x_c, y_c, w, h = box.tolist()

                grid_x = int(np.clip(x_c * 42, 0, 41))
                grid_y = int(np.clip(y_c * 42, 0, 41))

                tgt_cls[b, grid_y, grid_x] = label
                tgt_box[b, grid_y, grid_x] = torch.tensor([x_c, y_c, w, h], device=device)

        optimizer.zero_grad(set_to_none=True)

        pred_cls, pred_box = model(images_tensor)

        loss_cls = criterion_cls(pred_cls.view(-1, 3), tgt_cls.view(-1))

        fg_mask = tgt_cls < 2
        if fg_mask.any():
            loss_box = criterion_box(pred_box[fg_mask], tgt_box[fg_mask])
            total_loss = loss_cls + (2.0 * loss_box)
        else:
            total_loss = loss_cls

        total_loss.backward()
        optimizer.step()

        print(f" • [Batch {batch_idx+1}] Cross-Entropy Cls Loss: {loss_cls.item():.4f} | Total Step Loss: {total_loss.item():.4f}")

        if batch_idx >= 2:
            break

    print("\nPhase complete! The spatial target mapping and optimization loops are completely functional.")

if __name__ == "__main__":
    train_dino_pipeline()