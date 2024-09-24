import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class DINO_ViT(nn.Module):
    def __init__(self, config):
        super(DINO_ViT, self).__init__()
        self.backbone = vit_b_16(weights="IMAGENET1K_V1")  # Pretrained weights 설정
        self.backbone.heads = nn.Identity()  # Remove the default classification head
        
        # ViT의 feature 크기는 768이므로, hidden_dim을 768로 고정
        self.head = nn.Linear(768, config['model']['num_classes'])  # 768 -> 100 classes
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out
