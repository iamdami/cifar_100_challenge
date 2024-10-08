import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWithEfficientNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super(ResNetWithEfficientNet, self).__init__()
        
        # ResNet-18 backbone
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Identity()  # ResNet의 fully connected 레이어 제거
        
        # EfficientNet-B0 모델 결합 (torchvision에서 가져옴)
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # EfficientNet의 분류 레이어에 접근하여 in_features 값을 얻음
        efficientnet_in_features = self.efficientnet.classifier[1].in_features
        
        # ResNet의 마지막 레이어의 출력 크기를 얻음
        resnet_in_features = self.resnet.layer4[1].conv1.in_channels
        
        # 최종 분류 레이어
        self.classifier = nn.Linear(resnet_in_features + efficientnet_in_features, num_classes)
    
    def forward(self, x):
        # ResNet backbone 출력
        resnet_out = self.resnet(x)
        
        # EfficientNet 모델 출력
        efficientnet_out = self.efficientnet.features(x)
        efficientnet_out = torch.flatten(efficientnet_out, 1)  # feature map을 평탄화
        
        # ResNet과 EfficientNet의 출력을 결합
        combined_out = torch.cat([resnet_out, efficientnet_out], dim=1)
        
        # 최종 분류
        out = self.classifier(combined_out)
        return out
