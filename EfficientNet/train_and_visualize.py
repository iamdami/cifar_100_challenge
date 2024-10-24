import torch
import torch.optim as optim
import torch.nn as nn
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.efficientnet import build_model


class EarlyStopping:
    def __init__(self, patience=50, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Superclass mapping
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

# Helper functions for accuracy
def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k = min(k, output.size(1))
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top_k_acc = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        return top_k_acc

def superclass_accuracy(output, target):
    with torch.no_grad():
        pred_super = [superclass_mapping[p.item()] for p in torch.argmax(output, dim=1)]
        target_super = [superclass_mapping[t.item()] for t in target]
        correct_super = sum(p == t for p, t in zip(pred_super, target_super))
        return correct_super

def get_dataloaders(batch_size, num_workers):
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),  # CIFAR-100을 EfficientNet에 맞게 크기 조정
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.Resize(224),  # 테스트 데이터셋 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device, log_file):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    correct_superclass = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, top1_pred = outputs.max(1)
        correct_top1 += (top1_pred == labels).sum().item()

        # Top-5 accuracy
        _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
        correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

        # Superclass accuracy
        correct_superclass += superclass_accuracy(outputs, labels)
        total += labels.size(0)

    top1_accuracy = 100. * correct_top1 / total
    top5_accuracy = 100. * correct_top5 / total
    superclass_acc = 100. * correct_superclass / total
    avg_loss = running_loss / total

    log_entry = f"Train Loss: {avg_loss:.4f}, Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}, Superclass Accuracy: {superclass_acc:.4f}\n"
    print(log_entry)
    log_file.write(log_entry)

    return avg_loss, top1_accuracy, top5_accuracy, superclass_acc

def validate(model, val_loader, criterion, device, log_file):
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
            val_loss += loss.item() * inputs.size(0)

            _, top1_pred = outputs.max(1)
            correct_top1 += (top1_pred == labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Superclass accuracy
            correct_superclass += superclass_accuracy(outputs, labels)
            total += labels.size(0)

    top1_accuracy = 100. * correct_top1 / total
    top5_accuracy = 100. * correct_top5 / total
    superclass_acc = 100. * correct_superclass / total
    avg_loss = val_loss / total

    log_entry = f"Validation Loss: {avg_loss:.4f}, Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}, Superclass Accuracy: {superclass_acc:.4f}\n"
    print(log_entry)
    log_file.write(log_entry)

    return avg_loss, top1_accuracy, top5_accuracy, superclass_acc

def train_model(config):
    device = torch.device(config['train']['device'])
    train_loader, val_loader = get_dataloaders(config['train']['batch_size'], config['data']['num_workers'])

    model = build_model(
    config['model']['backbone'], 
    config['model']['num_classes'], 
    config['model']['pretrained'], 
    config['model']['dropout_rate']
).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'], nesterov=config['optimizer']['nesterov'])
    criterion = torch.nn.CrossEntropyLoss()
    
    best_accuracy = 0
    train_losses, val_losses, top1_acc_list, top5_acc_list, superclass_acc_list = [], [], [], [], []

    log_file = open("./logs/train_log.txt", "a")  # 로그 파일 생성 또는 열기

    for epoch in range(1, config['train']['epochs'] + 1):
        print(f"Epoch {epoch}/{config['train']['epochs']}")
        log_file.write(f"Epoch {epoch}/{config['train']['epochs']}\n")
        
        train_loss, train_top1, train_top5, train_superclass = train_one_epoch(model, train_loader, criterion, optimizer, device, log_file)
        val_loss, val_top1, val_top5, val_superclass = validate(model, val_loader, criterion, device, log_file)

        # 기록 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        top1_acc_list.append(val_top1)
        top5_acc_list.append(val_top5)
        superclass_acc_list.append(val_superclass)

        # Early stopping 및 best model 저장
        if val_top1 > best_accuracy:
            best_accuracy = val_top1
            torch.save(model.state_dict(), "./checkpoints/best_model.pth")

    # 결과 저장
    np.savez('./logs/train_results.npz', train_loss=train_losses, val_loss=val_losses, top1_acc=top1_acc_list, top5_acc=top5_acc_list, superclass_acc=superclass_acc_list)

    log_file.write("Training Complete\n")
    log_file.close()  # 로그 파일 닫기

    print("Training Complete")

# Visualization function
def visualize_results(log_file_path):
    epochs, train_loss, val_loss, top1_acc, top5_acc, superclass_acc = [], [], [], [], [], []
    
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if "Epoch" in line:
                epoch = int(line.split('/')[0].split()[-1])
                epochs.append(epoch)
            elif "Train Loss" in line:
                train_loss.append(float(line.split(',')[0].split()[-1]))
                top1_acc.append(float(line.split(',')[1].split()[-1]))
                top5_acc.append(float(line.split(',')[2].split()[-1]))
                superclass_acc.append(float(line.split(',')[3].split()[-1]))
            elif "Validation Loss" in line:
                val_loss.append(float(line.split(',')[0].split()[-1]))

    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, top1_acc, label="Top-1 Accuracy")
    plt.plot(epochs, top5_acc, label="Top-5 Accuracy")
    plt.plot(epochs, superclass_acc, label="Superclass Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()

if __name__ == "__main__":
    config = {
        "model": {
            "backbone": "efficientnet_b5",
            "num_classes": 100,
            "pretrained": True,
            "dropout_rate": 0.4
        },
        "train": {
            "batch_size": 32,
            "epochs": 200,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_every": 50,
            "early_stopping": True,
            "patience": 50
        },
        "optimizer": {
            "type": "SGD",
            "lr": 0.01,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.0005
        },
        "scheduler": {
            "type": "CosineAnnealingWarmRestarts",
            "T_0": 10,
            "T_mult": 2,
            "eta_min": 1e-6
        },
        "logging": {
            "log_dir": "./logs"
        },
        "data": {
            "augmentations": ["RandomResizedCrop", "RandomHorizontalFlip", "CutMix", "Mixup"],
            "num_workers": 8,
            "root": "./data"
        }
    }

    train_model(config)
    visualize_results(config['logging']['log_dir'] + "/train_output2.log")
