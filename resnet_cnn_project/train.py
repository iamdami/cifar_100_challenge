import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from models.resnet_cnn import ResNetWithEfficientNet
from datasets import get_dataloaders
from sklearn.metrics import precision_recall_curve

# Loss 기록 리스트
train_loss_history = []
val_loss_history = []
top1_acc_history = []
top5_acc_history = []
superclass_acc_history = []

# PR Curve를 위한 리스트
precision_list = []
recall_list = []

# Top-k 정확도 함수
def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k = min(k, output.size(1))
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top_k_acc = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        return top_k_acc

# Superclass 매핑 정의
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

# Superclass 정확도 계산 함수
def superclass_accuracy(output, target):
    with torch.no_grad():
        pred_super = [superclass_mapping[p.item()] for p in torch.argmax(output, dim=1)]
        target_super = [superclass_mapping[t.item()] for t in target]
        correct_super = sum(p == t for p, t in zip(pred_super, target_super))
        return correct_super

# CutMix 함수 정의
def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    
    new_data = data.clone()
    new_data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
    
    return new_data, targets, shuffled_targets, lam

# 랜덤 박스 생성
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_one_epoch(model, train_loader, criterion, optimizer, device, use_cutmix=False, alpha=1.0):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if use_cutmix:
            inputs, labels_a, labels_b, lam = cutmix(inputs, labels, alpha)
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{i+1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def validate(model, val_loader, criterion, device):
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
            
            # Top-1 정확도
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top-5 정확도
            correct_top5 += top_k_accuracy(outputs, labels, k=5)
            
            # Superclass 정확도
            correct_superclass += superclass_accuracy(outputs, labels)
            
            total += labels.size(0)

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    superclass_acc = correct_superclass / total

    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
    print(f'Top 1 Accuracy: {top1_accuracy:.4f}')
    print(f'Top 5 Accuracy: {top5_accuracy:.4f}')
    print(f'Superclass Accuracy: {superclass_acc:.4f}')
    
    return val_loss, top1_accuracy, top5_accuracy, superclass_acc

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_acc(top1_acc, top5_acc, superclass_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(top1_acc, label="Top-1 Accuracy")
    plt.plot(top5_acc, label="Top-5 Accuracy")
    plt.plot(superclass_acc, label="Superclass Accuracy")
    plt.title("Accuracy Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    train_loader, val_loader = get_dataloaders(config)
    
    # 모델 설정
    model = ResNetWithEfficientNet(num_classes=config['model']['num_classes'], pretrained=False).to(device)

    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    
    # 손실 함수
    criterion = torch.nn.CrossEntropyLoss()

    # 학습 시작
    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["epochs"]}')
        train_one_epoch(model, train_loader, criterion, optimizer, device, use_cutmix=True, alpha=1.0)
        
        val_loss, top1_accuracy, top5_accuracy, superclass_acc = validate(model, val_loader, criterion, device)
        
        # Loss 및 정확도 기록
        train_loss_history.append(val_loss)
        val_loss_history.append(val_loss)
        top1_acc_history.append(top1_accuracy)
        top5_acc_history.append(top5_accuracy)
        superclass_acc_history.append(superclass_acc)

        if epoch % config['train']['save_every'] == 0:
            checkpoint_path = os.path.join("./checkpoints", f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # Loss와 정확도 시각화
    plot_loss(train_loss_history, val_loss_history)
    plot_acc(top1_acc_history, top5_acc_history, superclass_acc_history)
    print("Training Complete!")

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

if __name__ == "__main__":
    import yaml
    
    # 설정 파일 로드
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # 모델 학습 시작
    train_model(config)
