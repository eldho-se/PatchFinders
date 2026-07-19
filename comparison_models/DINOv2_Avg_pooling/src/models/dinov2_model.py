import torch
import torch.nn as nn
from transformers import AutoModel


class DetectionHead(nn.Module):
    def __init__(self, hidden_size, num_classes, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.query_embed = nn.Embedding(num_queries, hidden_size)

        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes + 1)
        )

        self.box_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 4),
            nn.Sigmoid()
        )

    def forward(self, features):
        pooled          = features.mean(dim=1)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, self.num_queries, -1)
        class_logits    = self.class_head(pooled_expanded)
        pred_boxes      = self.box_head(pooled_expanded)
        return {"logits": class_logits, "pred_boxes": pred_boxes}


class DINOv2Detector(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", num_classes=6,
                 num_queries=100, freeze_backbone=True):
        super().__init__()
        print("[INFO] Loading DINOv2: " + model_name)
        self.backbone   = AutoModel.from_pretrained(model_name)
        hidden_size     = self.backbone.config.hidden_size
        print("[INFO] Hidden size: " + str(hidden_size))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[INFO] Backbone frozen — only training detection head")

        self.head = DetectionHead(hidden_size, num_classes, num_queries)

    def forward(self, pixel_values):
        outputs  = self.backbone(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        return self.head(features)