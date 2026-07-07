
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import MulticlassAUROC

def calculate_grid_auroc(model, data_loader, device):
    """
    Runs inference fast on the M1 GPU, but transfers final accumulated tensors 
    to the CPU for metric evaluation to prevent Metal performance kernel corruption errors.
    """
    model.eval()
    
    # Initialize the metric engines strictly on the CPU
    auroc_metric_macro = MulticlassAUROC(num_classes=3, average="macro").to("cpu")
    auroc_metric_none = MulticlassAUROC(num_classes=3, average="none").to("cpu")
    
    all_targets = []
    all_probabilities = []
    
    MAX_EVAL_BATCHES = 75
    total_batches = min(len(data_loader), MAX_EVAL_BATCHES)
    
    print(f"⏳ Processing evaluation stream (Capped at {total_batches} safety batches)...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx >= MAX_EVAL_BATCHES:
                break
                
            # Keep the heavy model forward execution on M1 GPU (mps)
            images_tensor = torch.stack(images).to(device)
            batch_size = images_tensor.size(0)
            
            # Background is class index 2; foreground classes are Crack=0 and Pothole=1
            tgt_cls = torch.full((batch_size, 42, 42), 2, dtype=torch.long, device=device)
            
            for b in range(batch_size):
                for box, label in zip(targets[b]["boxes"], targets[b]["labels"]):
                    grid_x = int(np.clip(box[0].item() * 42, 0, 41))
                    grid_y = int(np.clip(box[1].item() * 42, 0, 41))
                    clean_label = int(np.clip(label.item(), 0, 1))
                    tgt_cls[b, grid_y, grid_x] = clean_label
            
            pred_cls, _ = model(images_tensor)
            probs = F.softmax(pred_cls, dim=-1)
            
            # Immediately shift the flattened batch components to CPU memory
            all_targets.append(tgt_cls.view(-1).to("cpu"))
            all_probabilities.append(probs.view(-1, 3).to("cpu"))
            
            print(f"\r   → Evaluating: [{batch_idx + 1}/{total_batches}] batches processed", end="", flush=True)
            
    print("\n🖥️ Calculating stable AUROC curve metrics on CPU...")
    
    # Combine everything safely in RAM
    y_true = torch.cat(all_targets, dim=0)
    y_prob = torch.cat(all_probabilities, dim=0)
    
    y_true = torch.clamp(y_true, 0, 2)
    
    # Compute using standard x86/ARM instructions instead of Metal kernels
    macro_auroc = auroc_metric_macro(y_prob, y_true).item()
    per_class_auroc = auroc_metric_none(y_prob, y_true).numpy()
    
    metrics = {
        "Macro AUROC": macro_auroc,
        "Crack AUROC (Class 0)": float(per_class_auroc[0]),
        "Pothole AUROC (Class 1)": float(per_class_auroc[1]),
        "Background AUROC (Class 2)": float(per_class_auroc[2])
    }
    
    return metrics


def evaluate_ood_detection(model, id_loader, ood_loader, device):
    """
    Computes Out-of-Distribution (OOD) detection performance metrics using different scores:
    1. Maximum Softmax Probability (MSP)
    2. Energy Score
    3. Shannon Entropy
    
    ID is positive (1), OOD is negative (0).
    Returns ROC-AUC and FPR95 scores for each method.
    """
    model.eval()
    
    id_msp_scores = []
    id_energy_scores = []
    id_entropy_scores = []
    
    ood_msp_scores = []
    ood_energy_scores = []
    ood_entropy_scores = []
    
    # Process In-Distribution (ID) Loader
    print("⏳ Processing ID validation stream for OOD baseline...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(id_loader):
            if len(images) == 0:
                continue
            images_tensor = torch.stack(images).to(device)
            pred_cls, _ = model(images_tensor) # [B, 42, 42, 3]
            
            # Compute scores per image
            for b in range(pred_cls.size(0)):
                logits = pred_cls[b] # [42, 42, 3]
                probs = F.softmax(logits, dim=-1)
                
                # MSP: Mean of max probability over all patches
                max_probs = probs.max(dim=-1)[0] # [42, 42]
                id_msp_scores.append(max_probs.mean().item())
                
                # Energy Score: Mean logsumexp over all patches
                energy = torch.logsumexp(logits, dim=-1) # [42, 42]
                id_energy_scores.append(energy.mean().item())
                
                # Entropy: Mean negative Shannon entropy (negated so higher is more ID-like)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1) # [42, 42]
                id_entropy_scores.append(-entropy.mean().item())
                
    # Process Out-of-Distribution (OOD) Loader
    print("⏳ Processing OOD stream...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(ood_loader):
            if len(images) == 0:
                continue
            images_tensor = torch.stack(images).to(device)
            pred_cls, _ = model(images_tensor)
            
            for b in range(pred_cls.size(0)):
                logits = pred_cls[b]
                probs = F.softmax(logits, dim=-1)
                
                max_probs = probs.max(dim=-1)[0]
                ood_msp_scores.append(max_probs.mean().item())
                
                energy = torch.logsumexp(logits, dim=-1)
                ood_energy_scores.append(energy.mean().item())
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                ood_entropy_scores.append(-entropy.mean().item())
                
    # Combine scores and create targets (ID = 1, OOD = 0)
    from sklearn.metrics import roc_auc_score
    
    y_true = np.concatenate([np.ones(len(id_msp_scores)), np.zeros(len(ood_msp_scores))])
    
    msp_scores = np.concatenate([id_msp_scores, ood_msp_scores])
    energy_scores = np.concatenate([id_energy_scores, ood_energy_scores])
    entropy_scores = np.concatenate([id_entropy_scores, ood_entropy_scores])
    
    msp_auroc = roc_auc_score(y_true, msp_scores)
    energy_auroc = roc_auc_score(y_true, energy_scores)
    entropy_auroc = roc_auc_score(y_true, entropy_scores)
    
    # Calculate False Positive Rate at 95% TPR (FPR95)
    def calculate_fpr95(scores, labels):
        id_scores = scores[labels == 1]
        ood_scores = scores[labels == 0]
        
        # Sort ID scores descending (higher scores mean more ID-like)
        id_scores_sorted = np.sort(id_scores)[::-1]
        # Find threshold where 95% of ID are correctly identified
        threshold_idx = int(np.floor(0.95 * len(id_scores)))
        threshold = id_scores_sorted[min(threshold_idx, len(id_scores) - 1)]
        
        # Calculate fraction of OOD samples that exceed this threshold (false positives)
        fpr = np.mean(ood_scores >= threshold)
        return fpr
        
    msp_fpr95 = calculate_fpr95(msp_scores, y_true)
    energy_fpr95 = calculate_fpr95(energy_scores, y_true)
    entropy_fpr95 = calculate_fpr95(entropy_scores, y_true)
    
    results = {
        "MSP AUROC": float(msp_auroc),
        "MSP FPR95": float(msp_fpr95),
        "Energy AUROC": float(energy_auroc),
        "Energy FPR95": float(energy_fpr95),
        "Entropy AUROC": float(entropy_auroc),
        "Entropy FPR95": float(entropy_fpr95)
    }
    
    return results

