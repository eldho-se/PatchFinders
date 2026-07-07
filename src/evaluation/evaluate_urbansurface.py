import os
import sys
import torch
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from processing.dataset import RoadDamageDataset, collate_fn
from models.dino_detector import DinoMultiBoxDetector
from evaluation.metrics import calculate_grid_auroc

def evaluate_urbansurface():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware device target: {device.type.upper()}")
    
    manifest_path = os.path.join(project_root, "data/processed/urbansurface_harmonized.json")
    checkpoint_path = os.path.join(project_root, "models/checkpoints/dino_probing_heads.pt")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Error: Harmonized manifest missing at: {manifest_path}")
        
    print(f"Loading UrbanSurface dataset from: {manifest_path}")
    
    dataset = RoadDamageDataset(
        manifest_path=manifest_path,
        apply_augmentations=False,
        target_size=(588, 588)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Loaded {len(dataset)} images in the stream.")
    
    print("Loading Multi-Box Linear Probing Network...")
    model = DinoMultiBoxDetector(num_classes=2, grid_size=42).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from: {checkpoint_path}")
        trainable_weights = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(trainable_weights, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Running with randomized weights.")
        
    model.eval()
    
    results = calculate_grid_auroc(model, loader, device)
    
    print("\n--- UrbanSurface Performance Scoreboard ---")
    for key, val in results.items():
        print(f" • {key}: {val:.4f}" if isinstance(val, float) else f" • {key}: {val}")
    print("-------------------------------------------")

if __name__ == "__main__":
    evaluate_urbansurface()