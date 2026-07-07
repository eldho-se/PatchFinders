import torch
import torch.nn as nn

class DinoRoadDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 1. Load the lightweight, frozen DINOv2 ViT-Small backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # Freeze backbone parameters completely to prevent M1 RAM spikes
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embedding_dim = 384 # ViT-Small feature token dimension
        self.num_classes = num_classes

        # 2. Lightweight linear projection heads for bounding boxes and classes
        # Assuming 42x42 grid tokens from a 588x588 image input
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.box_regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4), # Predicts [x_center, y_center, width, height]
            nn.Sigmoid()       # Keeps normalized coordinates between 0 and 1
        )

    def forward(self, x):
        # x shape: [Batch_Size, 3, 588, 588]
        batch_size = x.size(0)

        # Extract patch tokens directly bypassing the global [CLS] token
        # DINOv2 returns a patch feature map of shape [Batch_Size, 42*42, 384]
        patch_features = self.backbone.get_intermediate_layers(x, n=1)[0]

        # Aggregate features across spatial map or take max/mean projection
        # To keep it extremely light on your 8GB M1, we mean-pool the tokens for a global frame prediction
        global_features = torch.mean(patch_features, dim=1) # Shape: [Batch_Size, 384]

        class_logits = self.classifier(global_features)
        pred_boxes = self.box_regressor(global_features)

        return class_logits, pred_boxes
