import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2
from efficientnet_pytorch import EfficientNet

class EfficientNetWideResNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, dropout_rate=0.3):
        super(EfficientNetWideResNet, self).__init__()
        
        # WideResNet 모델 로드
        self.wide_resnet = wide_resnet50_2(pretrained=pretrained)
        wide_resnet_output_dim = self.wide_resnet.fc.in_features  # WideResNet의 마지막 레이어 차원
        self.wide_resnet.fc = nn.Identity()  # 마지막 레이어 제거
        
        # EfficientNet 모델 로드
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        efficientnet_output_dim = self.efficientnet._fc.in_features  # EfficientNet의 마지막 레이어 차원
        self.efficientnet._fc = nn.Identity()  # 마지막 레이어 제거
        
        # 두 모델의 출력 차원 결합 (WideResNet과 EfficientNet의 차원 합)
        self.classifier = nn.Sequential(
            nn.Linear(wide_resnet_output_dim + efficientnet_output_dim, 512),
            nn.Dropout(dropout_rate),  # Dropout 레이어 추가
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # WideResNet의 특징 추출
        wide_resnet_features = self.wide_resnet(x)
        
        # EfficientNet의 특징 추출
        efficientnet_features = self.efficientnet(x)
        
        # 특징 결합
        combined_features = torch.cat((wide_resnet_features, efficientnet_features), dim=1)
        
        # 결합된 특징을 사용해 최종 분류
        output = self.classifier(combined_features)
        
        return output
