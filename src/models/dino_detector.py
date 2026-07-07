
import torch
import torch.nn as nn

class DinoMultiBoxDetector(nn.Module):
    def __init__(self, num_classes=2, grid_size=42):
        super().__init__()
        # 1. Load and freeze the core DINOv2 ViT-Small backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embedding_dim = 384
        self.num_classes = num_classes
        self.grid_size = grid_size  # 588 / 14 = 42

        # 2. Dense Patch Heads (Processes each patch token individually)
        self.patch_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes + 1) # +1 for Background class
        )
        
        self.patch_regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4), # [x_offset, y_offset, local_w, local_h]
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract the dense intermediate layer tokens from the transformer
        features = self.backbone.get_intermediate_layers(x, n=1)[0]
        
        # Apply the linear probing heads across the spatial token sequences
        class_logits = self.patch_classifier(features)  
        pred_boxes = self.patch_regressor(features)      
        
        # Reshape into a structured 2D spatial feature grid [Batch_Size, 42, 42, ...]
        class_grid = class_logits.view(batch_size, self.grid_size, self.grid_size, -1)
        box_grid = pred_boxes.view(batch_size, self.grid_size, self.grid_size, 4)
        
        return class_grid, box_grid
