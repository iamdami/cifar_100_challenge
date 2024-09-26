import torch
import torch.nn as nn
from dino_vit import DINO_ViT  # DINO 모델
from datasets import get_dataloaders  # 데이터 로드 함수
import os

def load_checkpoint(model, optimizer, filepath, device):
    """
    체크포인트를 불러오는 함수
    Args:
        model: 평가에 사용될 모델
        optimizer: 옵티마이저 (평가 시에는 불필요하지만, 로드됨)
        filepath: 체크포인트 파일 경로
        device: 사용 중인 장치 (cuda 또는 cpu)
    """
    if os.path.isfile(filepath):
        print(f"Loading checkpoint from '{filepath}'")
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"No checkpoint found at '{filepath}'")


# Add your superclass mapping here for CIFAR-100
superclass_mapping = {
    0: "aquatic mammals", 1: "aquatic mammals", 2: "aquatic mammals", 3: "aquatic mammals", 4: "aquatic mammals",
    5: "fish", 6: "fish", 7: "fish", 8: "fish", 9: "fish",
    10: "flowers", 11: "flowers", 12: "flowers", 13: "flowers", 14: "flowers",
    15: "food containers", 16: "food containers", 17: "food containers", 18: "food containers", 19: "food containers",
    20: "fruit and vegetables", 21: "fruit and vegetables", 22: "fruit and vegetables", 23: "fruit and vegetables", 24: "fruit and vegetables",
    25: "household electrical devices", 26: "household electrical devices", 27: "household electrical devices", 28: "household electrical devices", 29: "household electrical devices",
    30: "household furniture", 31: "household furniture", 32: "household furniture", 33: "household furniture", 34: "household furniture",
    35: "insects", 36: "insects", 37: "insects", 38: "insects", 39: "insects",
    40: "large carnivores", 41: "large carnivores", 42: "large carnivores", 43: "large carnivores", 44: "large carnivores",
    45: "large man-made outdoor things", 46: "large man-made outdoor things", 47: "large man-made outdoor things", 48: "large man-made outdoor things", 49: "large man-made outdoor things",
    50: "large natural outdoor scenes", 51: "large natural outdoor scenes", 52: "large natural outdoor scenes", 53: "large natural outdoor scenes", 54: "large natural outdoor scenes",
    55: "large omnivores and herbivores", 56: "large omnivores and herbivores", 57: "large omnivores and herbivores", 58: "large omnivores and herbivores", 59: "large omnivores and herbivores",
    60: "medium-sized mammals", 61: "medium-sized mammals", 62: "medium-sized mammals", 63: "medium-sized mammals", 64: "medium-sized mammals",
    65: "non-insect invertebrates", 66: "non-insect invertebrates", 67: "non-insect invertebrates", 68: "non-insect invertebrates", 69: "non-insect invertebrates",
    70: "people", 71: "people", 72: "people", 73: "people", 74: "people",
    75: "reptiles", 76: "reptiles", 77: "reptiles", 78: "reptiles", 79: "reptiles",
    80: "small mammals", 81: "small mammals", 82: "small mammals", 83: "small mammals", 84: "small mammals",
    85: "trees", 86: "trees", 87: "trees", 88: "trees", 89: "trees",
    90: "vehicles 1", 91: "vehicles 1", 92: "vehicles 1", 93: "vehicles 1", 94: "vehicles 1",
    95: "vehicles 2", 96: "vehicles 2", 97: "vehicles 2", 98: "vehicles 2", 99: "vehicles 2"
}

def top_k_accuracy(output, target, k=5):
    """Calculate top-k accuracy."""
    with torch.no_grad():
        max_k = min(k, output.size(1))
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top_k_acc = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        return top_k_acc

def superclass_accuracy(output, target):
    """Calculate accuracy based on superclass."""
    with torch.no_grad():
        # Map each prediction and target to their superclass
        pred_super = [superclass_mapping[p.item()] for p in torch.argmax(output, dim=1)]
        target_super = [superclass_mapping[t.item()] for t in target]
        correct_super = sum(p == t for p, t in zip(pred_super, target_super))
        return correct_super

def validate(model, val_loader, criterion, device):
    """
    평가 함수
    """
    model.eval()
    val_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    correct_superclass = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            correct_top5 += top_k_accuracy(outputs, labels, k=5)
            
            # Superclass accuracy
            correct_superclass += superclass_accuracy(outputs, labels)
            
            total += labels.size(0)

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    superclass_acc = correct_superclass / total

    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
    print(f'Top 1 Accuracy: {top1_accuracy:.4f}')
    print(f'Top 5 Accuracy: {top5_accuracy:.4f}')
    print(f'Superclass Accuracy: {superclass_acc:.4f}')
    
    return top1_accuracy, top5_accuracy, superclass_acc

# Remaining functions like `load_checkpoint`, `evaluate_model`, etc. would remain the same.

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
    
    # 손실 함수 정의 (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    
    # 체크포인트 불러오기 (device를 전달)
    load_checkpoint(model, optimizer, checkpoint_path, device)
    
    # 검증 데이터로 모델 평가
    accuracy = validate(model, val_loader, criterion, device)
    print(f"Final Accuracy: {accuracy:.4f}")



if __name__ == "__main__":
    import yaml
    
    # 설정 파일 로드
    with open("config_dino_cifar100.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # 체크포인트 경로
    checkpoint_path = "./checkpoints/checkpoint_epoch_50.pth"  # 마지막 에포크 체크포인트
    
    # 모델 평가
    evaluate_model(config, checkpoint_path)
