import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import importlib.util
import numpy as np

def train_dino_pipeline():
    # 1. Hardware Targeting (Unified Memory Core Assignment)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training Engine Initialized on Device Backend: {device.type.upper()}")

    # 2. Dynamic Component Loading
    spec_ds = importlib.util.spec_from_file_location("dataset", "src/processing/dataset.py")
    ds_mod = importlib.util.module_from_spec(spec_ds)
    spec_ds.loader.exec_module(ds_mod)

    spec_md = importlib.util.spec_from_file_location("dino_detector", "src/models/dino_detector.py")
    md_mod = importlib.util.module_from_spec(spec_md)
    spec_md.loader.exec_module(md_mod)

    # 3. Memory-Isolated Data Loader Setup (588x588 Patch Fix)
    train_dataset = ds_mod.RoadDamageDataset(
        manifest_path="data/processed/train_harmonized.json",
        apply_augmentations=True,
        target_size=(588, 588)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, # Kept low at 2 to preserve the M1's 8GB RAM headroom
        shuffle=True, 
        collate_fn=ds_mod.collate_fn,
        num_workers=0
    )

    # 4. Initialize Multi-Box Network Architecture
    model = md_mod.DinoMultiBoxDetector(num_classes=2, grid_size=42).to(device)

    # Only pass parameters that require gradients (the linear probing heads)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-4)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_box = nn.MSELoss()

    print("Beginning Training Run Validation Loop...")
    model.train()

    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move image tensors to M1 GPU memory bank
        images_tensor = torch.stack(images).to(device)
        batch_size = images_tensor.size(0)

        # Initialize blank target matrices matching our 42x42 spatial layout
        # Default class index is 2 (Background)
        tgt_cls = torch.full((batch_size, 42, 42), 2, dtype=torch.long, device=device)
        tgt_box = torch.zeros((batch_size, 42, 42, 4), dtype=torch.float32, device=device)

        # 5. Spatial Label Assigner: Map YOLO coordinates into the 42x42 Patch Grid
        for b in range(batch_size):
            boxes = targets[b]["boxes"]
            labels = targets[b]["labels"]

            for box, label in zip(boxes, labels):
                x_c, y_c, w, h = box.tolist()

                # Determine which exact patch column and row houses the center point
                grid_x = int(np.clip(x_c * 42, 0, 41))
                grid_y = int(np.clip(y_c * 42, 0, 41))

                # Assign the labels directly to that coordinate anchor point
                tgt_cls[b, grid_y, grid_x] = label
                tgt_box[b, grid_y, grid_x] = torch.tensor([x_c, y_c, w, h], device=device)

        # 6. Optimization Steps Pass
        optimizer.zero_grad(set_to_none=True) # Drops gradients completely rather than zeroing out RAM buffers

        pred_cls, pred_box = model(images_tensor)

        # Flatten outputs and targets over the grid space to compute total loss matrices
        loss_cls = criterion_cls(pred_cls.view(-1, 3), tgt_cls.view(-1))

        # Only calculate box localization penalties on patches containing actual cracks/potholes
        fg_mask = tgt_cls < 2
        if fg_mask.any():
            loss_box = criterion_box(pred_box[fg_mask], tgt_box[fg_mask])
            total_loss = loss_cls + (2.0 * loss_box) # Heavy upweighting for box bounds
        else:
            total_loss = loss_cls

        total_loss.backward()
        optimizer.step()

        print(f" • [Batch {batch_idx+1}] Cross-Entropy Cls Loss: {loss_cls.item():.4f} | Total Step Loss: {total_loss.item():.4f}")

        # Safe checkpoint exit to confirm working mechanics without exploding runtime memory
        if batch_idx >= 2:
            break

    print("\n🎉 Phase complete! The spatial target mapping and optimization loops are completely functional.")

if __name__ == "__main__":
    train_dino_pipeline()
