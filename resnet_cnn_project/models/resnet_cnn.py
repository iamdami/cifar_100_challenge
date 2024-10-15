import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWithEfficientNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, dropout_rate=0.3):
        super(ResNetWithEfficientNet, self).__init__()
        
        # ResNet-18 backbone
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # ResNet fc 레이어의 in_features 저장
        resnet_output_dim = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Identity()  # ResNet의 fully connected 레이어 제거
        
        # EfficientNet-B0 모델 결합
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # EfficientNet의 classifier를 제거하고 마지막 풀링 층의 출력 크기 가져오기
        efficientnet_output_dim = self.efficientnet.classifier[1].in_features  # EfficientNet의 마지막 linear 층
        
        self.efficientnet.classifier = nn.Identity()  # EfficientNet의 분류 레이어 제거
        
        # Dropout 레이어 추가
        self.dropout = nn.Dropout(dropout_rate)
        
        # 최종 분류 레이어 (ResNet과 EfficientNet 출력 결합)
        self.classifier = nn.Linear(resnet_output_dim + efficientnet_output_dim, num_classes)
    
    def forward(self, x):
        # ResNet backbone 출력
        resnet_out = self.resnet(x)
        
        # EfficientNet 모델 출력
        efficientnet_out = self.efficientnet(x)
        
        # ResNet과 EfficientNet의 출력을 결합
        combined_out = torch.cat([resnet_out, efficientnet_out], dim=1)
        
        # Dropout 적용
        combined_out = self.dropout(combined_out)
        
        # 최종 분류
        out = self.classifier(combined_out)
        return out
