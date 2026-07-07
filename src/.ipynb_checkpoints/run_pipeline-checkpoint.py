
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import importlib.util
import numpy as np

def run_main_workflow(epochs=3, sandbox_mode=False):
    # 1. Hardware Targeting
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training Engine Initialized on Device Backend: {device.type.upper()}")
    if sandbox_mode:
        print("⚡ OPERATIONAL MODE: SANDBOX (Subsampling dataset to protect 8GB RAM headroom)")
    else:
        print("🔥 OPERATIONAL MODE: FULL DATASET FINE-TUNING")

    # 2. Dynamic Component Modules Loading
    spec_ds = importlib.util.spec_from_file_location("dataset", "src/processing/dataset.py")
    ds_mod = importlib.util.module_from_spec(spec_ds)
    spec_ds.loader.exec_module(ds_mod)

    spec_md = importlib.util.spec_from_file_location("dino_detector", "src/models/dino_detector.py")
    md_mod = importlib.util.module_from_spec(spec_md)
    spec_md.loader.exec_module(md_mod)

    spec_eval = importlib.util.spec_from_file_location("metrics", "src/evaluation/metrics.py")
    eval_mod = importlib.util.module_from_spec(spec_eval)
    spec_eval.loader.exec_module(eval_mod)

    manifest_path = "data/processed/train_harmonized.json"
    if not os.path.exists(manifest_path):
        print(f"❌ Error: Manifest file missing at {manifest_path}")
        return

    # 3. Complete Dataset Instantiation
    full_dataset = ds_mod.RoadDamageDataset(
        manifest_path=manifest_path,
        apply_augmentations=True,
        target_size=(588, 588)
    )
    
    total_available = len(full_dataset)
    print(f"📊 Total entries discovered in manifest metadata: {total_available} images")

    # 4. Stratified Index Splitting
    np.random.seed(42)
    shuffled_indices = np.random.permutation(total_available)
    
    if sandbox_mode:
        train_indices = shuffled_indices[:500]    
        val_indices = shuffled_indices[500:650]   
    else:
        split_idx = int(total_available * 0.85)
        train_indices = shuffled_indices[:split_idx]
        val_indices = shuffled_indices[split_idx:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # 5. Core Data Streams
    train_loader = DataLoader(
        train_subset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=ds_mod.collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=2,
        shuffle=False,
        collate_fn=ds_mod.collate_fn,
        num_workers=0
    )

    print(f"📦 Data Streams Bound -> Training: {len(train_subset)} images | Validation: {len(val_subset)} images")
    print("🤖 Loading Multi-Box Linear Probing Network Architecture...")
    
    model = md_mod.DinoMultiBoxDetector(num_classes=2, grid_size=42).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-4)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_box = nn.MSELoss()

    total_train_batches = len(train_loader)

    # 6. Deep Learning Execution Loop
    for epoch in range(epochs):
        print(f"\n🏋️ Starting Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images_tensor = torch.stack(images).to(device)
            batch_size = images_tensor.size(0)

            tgt_cls = torch.full((batch_size, 42, 42), 2, dtype=torch.long, device=device)
            tgt_box = torch.zeros((batch_size, 42, 42, 4), dtype=torch.float32, device=device)

            for b in range(batch_size):
                for box, label in zip(targets[b]["boxes"], targets[b]["labels"]):
                    grid_x = int(np.clip(box[0].item() * 42, 0, 41))
                    grid_y = int(np.clip(box[1].item() * 42, 0, 41))
                    tgt_cls[b, grid_y, grid_x] = label
                    tgt_box[b, grid_y, grid_x] = torch.tensor([box[0], box[1], box[2], box[3]], device=device)

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
            epoch_loss += total_loss.item()

            # Dynamic inline progress step counter for the training batch loop
            print(f"\r   → Training: [{batch_idx + 1}/{total_train_batches}] batches | Loss: {total_loss.item():.4f}", end="", flush=True)

        print(f"\n✅ Epoch {epoch + 1} Complete. Average Train Loss: {epoch_loss / total_train_batches:.4f}")

        # Persistent Checkpoint Sync
        weights_dir = "models/checkpoints"
        os.makedirs(weights_dir, exist_ok=True)
        save_path = os.path.join(weights_dir, "dino_probing_heads.pt")
        trainable_weights = {k: v for k, v in model.state_dict().items() if "backbone" not in k}
        torch.save(trainable_weights, save_path)
        print(f"💾 Checkpoint weights synchronized to disk: {save_path}")

    # 7. GPU-Accelerated Metric Scorecard Delivery
    print(f"\n📊 Computing GPU-accelerated AUROC metrics over evaluation stream...")
    results = eval_mod.calculate_grid_auroc(model, val_loader, device)
    
    print("\n--- Pipeline Scoreboard Summary ---")
    for key, val in results.items():
        print(f" • {key}: {val:.4f}" if isinstance(val, float) else f" • {key}: {val}")
    print("-----------------------------------")
    print("🎉 Pipeline optimization run completed successfully!")

if __name__ == "__main__":
    run_main_workflow(epochs=3, sandbox_mode=True)
