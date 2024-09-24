import torch
import torch.nn as nn
from dino_vit import DINO_ViT  # 학습에 사용된 모델 파일
from datasets import get_dataloaders  # 데이터셋 로드 함수
import os

def load_checkpoint(model, optimizer, filepath):
    """
    체크포인트를 불러오는 함수
    Args:
        model: 평가에 사용될 모델
        optimizer: 옵티마이저 (평가 시에는 불필요하지만, 로드됨)
        filepath: 체크포인트 파일 경로
    """
    if os.path.isfile(filepath):
        print(f"Loading checkpoint from '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"No checkpoint found at '{filepath}'")

def validate(model, val_loader, criterion, device):
    """
    평가 함수
    Args:
        model: 학습된 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: GPU 또는 CPU
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}, Loss: {val_loss / len(val_loader):.4f}')
    return accuracy

def evaluate_model(config, checkpoint_path):
    """
    모델을 평가하는 함수
    Args:
        config: 설정 값 (yaml 파일로부터 로드됨)
        checkpoint_path: 체크포인트 파일 경로
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    _, val_loader = get_dataloaders(config)
    
    # 모델 설정
    model = DINO_ViT(hidden_dim=config['model']['hidden_dim'],
                     num_heads=config['model']['num_heads'],
                     depth=config['model']['depth'],
                     patch_size=config['model']['patch_size'],
                     num_classes=config['model']['num_classes']).to(device)
    
    # 옵티마이저 (필요하지 않지만, 체크포인트 로드를 위해 사용)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'])
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 체크포인트 불러오기
    load_checkpoint(model, optimizer, checkpoint_path)
    
    # 검증 데이터로 모델 평가
    accuracy = validate(model, val_loader, criterion, device)
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    import yaml
    
    # 설정 파일 로드
    with open("config_dino_cifar100.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # 체크포인트 경로
    checkpoint_path = "./checkpoints/checkpoint_epoch_150.pth"  # 마지막 에포크 체크포인트
    
    # 모델 평가
    evaluate_model(config, checkpoint_path)
