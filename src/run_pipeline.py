import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import importlib.util
import numpy as np

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if run directly in an environment without __file__
    script_dir = os.path.abspath("src")

project_root = os.path.dirname(script_dir)

spec_ds = importlib.util.spec_from_file_location("dataset", os.path.join(script_dir, "processing/dataset.py"))
ds_mod = importlib.util.module_from_spec(spec_ds)
spec_ds.loader.exec_module(ds_mod)

spec_md = importlib.util.spec_from_file_location("dino_detector", os.path.join(script_dir, "models/dino_detector.py"))
md_mod = importlib.util.module_from_spec(spec_md)
spec_md.loader.exec_module(md_mod)

spec_eval = importlib.util.spec_from_file_location("metrics", os.path.join(script_dir, "evaluation/metrics.py"))
eval_mod = importlib.util.module_from_spec(spec_eval)
spec_eval.loader.exec_module(eval_mod)


def get_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device Specification: {device.type.upper()}")
    return device

def load_data(sandbox_mode=True, manifest_path="data/processed/train_harmonized.json"):

    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(project_root, manifest_path)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Error: Hromnized file missing at {manifest_path}")
    
    full_dataset = ds_mod.RoadDamageDataset(
        manifest_path=manifest_path,
        apply_augmentations=True,
        target_size=(588, 588)
    )
    
    total_available = len(full_dataset)
    print(f"Total entries discovered in manifest metadata: {total_available} images")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(total_available)
    
    if sandbox_mode:
        print("Sample dataset mode enabled: training will be limited to 500 samples with 150 validation samples.")
        train_indices = shuffled_indices[:500]    
        val_indices = shuffled_indices[500:650]   
    else:
        print("Full dataset mode enabled: training will use 85% of the dataset with 15% for validation.")
        split_idx = int(total_available * 0.85)
        train_indices = shuffled_indices[:split_idx]
        val_indices = shuffled_indices[split_idx:]

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

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

    print(f"Data Streams Bound -> Training: {len(train_subset)} images | Validation: {len(val_subset)} images")
    return train_loader, val_loader


def initialize_model_and_optimizer(device, num_classes=2, grid_size=42, lr=2e-4, weight_decay=1e-4):
    print("Loading Multi-Box Linear Probing Network Architecture...")
    model = md_mod.DinoMultiBoxDetector(num_classes=num_classes, grid_size=grid_size).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    # Class weights: [Crack, Pothole, Background]
    # Background (class 2) is downweighted to 0.02 to prioritize Crack (0) and Pothole (1) detections
    class_weights = torch.tensor([1.0, 1.0, 0.02], dtype=torch.float32, device=device)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_box = nn.MSELoss()
    
    return model, optimizer, criterion_cls, criterion_box


def load_checkpoint(model, checkpoint_path="models/checkpoints/dino_probing_heads.pt", device="cpu"):

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        trainable_weights = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(trainable_weights, strict=False)
        print("Checkpoint loaded successfully (strict=False for backbone).")
    else:
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
    return model


def train_pipeline(model, train_loader, optimizer, criterion_cls, criterion_box, epochs, device, checkpoint_path="models/checkpoints/dino_probing_heads.pt"):
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)

    total_train_batches = len(train_loader)

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images_tensor = torch.stack(images).to(device)
            batch_size = images_tensor.size(0)

            # Background is class index 2; foreground classes are Crack=0 and Pothole=1
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

            print(f"\r   → Training: [{batch_idx + 1}/{total_train_batches}] batches | Loss: {total_loss.item():.4f}", end="", flush=True)

        print(f"\nEpoch {epoch + 1} Complete. Average Train Loss: {epoch_loss / total_train_batches:.4f}")

        # Persistent Checkpoint Sync
        weights_dir = os.path.dirname(checkpoint_path)
        os.makedirs(weights_dir, exist_ok=True)
        trainable_weights = {k: v for k, v in model.state_dict().items() if "backbone" not in k}
        torch.save(trainable_weights, checkpoint_path)
        print(f"Checkpoint weights synchronized to disk: {checkpoint_path}")


def evaluate_pipeline(model, val_loader, device, max_batches=None):
    print(f"\nComputing GPU-accelerated AUROC metrics over evaluation stream...")
    results = eval_mod.calculate_grid_auroc(model, val_loader, device, max_batches=max_batches)
    
    print("\n--- Pipeline Scoreboard Summary ---")
    for key, val in results.items():
        print(f" • {key}: {val:.4f}" if isinstance(val, float) else f" • {key}: {val}")
    print("-----------------------------------")
    print("Pipeline optimization run completed successfully!")
    return results


def load_ood_data(sandbox_mode=True, manifest_path="data/processed/ood_harmonized.json"):
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(project_root, manifest_path)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Error: OOD Manifest file missing at {manifest_path}")

    full_dataset = ds_mod.RoadDamageDataset(
        manifest_path=manifest_path,
        apply_augmentations=False,
        target_size=(588, 588)
    )
    
    total_available = len(full_dataset)
    print(f"Total OOD entries discovered: {total_available} images")

    # Sample subset in sandbox mode
    if sandbox_mode:
        print("Sample OOD mode enabled: OOD evaluation will be limited to 100 samples.")
        np.random.seed(42)
        shuffled_indices = np.random.permutation(total_available)
        subset_indices = shuffled_indices[:100]
        dataset_subset = Subset(full_dataset, subset_indices)
    else:
        dataset_subset = full_dataset

    # Core Data Stream
    ood_loader = DataLoader(
        dataset_subset,
        batch_size=2,
        shuffle=False,
        collate_fn=ds_mod.collate_fn,
        num_workers=0
    )

    print(f"📦 OOD Data Stream Bound -> {len(dataset_subset)} images")
    return ood_loader


def evaluate_ood(model, val_loader, ood_loader, device):
    """Runs OOD evaluation and prints metrics."""
    print(f"\n📊 Evaluating Out-of-Distribution (OOD) detection performance...")
    if hasattr(eval_mod, "evaluate_ood_detection"):
        results = eval_mod.evaluate_ood_detection(model, val_loader, ood_loader, device)
        
        print("\n--- OOD Detection Scoreboard Summary ---")
        for key, val in results.items():
            print(f" • {key}: {val:.4f}" if isinstance(val, float) else f" • {key}: {val}")
        print("----------------------------------------")
        return results
    else:
        print("⚠️ metrics.py does not contain evaluate_ood_detection function.")
        return None


def run_main_workflow(epochs=3, sandbox_mode=True):
    """Runs the entire pipeline (setup, loading data, training, evaluation, and OOD detection)."""
    device = get_device()
    train_loader, val_loader = load_data(sandbox_mode=sandbox_mode)
    
    try:
        ood_loader = load_ood_data(sandbox_mode=sandbox_mode)
    except Exception as e:
        print(f"⚠️ Skipping OOD Loader setup: {e}")
        ood_loader = None

    model, optimizer, criterion_cls, criterion_box = initialize_model_and_optimizer(device)
    
    train_pipeline(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion_cls=criterion_cls,
        criterion_box=criterion_box,
        epochs=epochs,
        device=device
    )
    
    evaluate_pipeline(model=model, val_loader=val_loader, device=device)
    
    if ood_loader is not None:
        evaluate_ood(model=model, val_loader=val_loader, ood_loader=ood_loader, device=device)


if __name__ == "__main__":
    run_main_workflow(epochs=2, sandbox_mode=False)

