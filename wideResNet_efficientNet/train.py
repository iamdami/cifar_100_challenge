import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
from models.efficientnet_wideresnet import EfficientNetWideResNet
from datasets import get_dataloaders

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=20, delta=0):
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

def cutmix_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    target_a = y
    target_b = y[index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    return x, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

train_loss_history = []
val_loss_history = []
top1_acc_history = []
top5_acc_history = []
superclass_acc_history = []

def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k = min(k, output.size(1))
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        top_k_acc = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        return top_k_acc

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


def superclass_accuracy(output, target):
    with torch.no_grad():
        pred_super = [superclass_mapping[p.item()] for p in torch.argmax(output, dim=1)]
        target_super = [superclass_mapping[t.item()] for t in target]
        correct_super = sum(p == t for p, t in zip(pred_super, target_super))
        return correct_super

def train_one_epoch(model, train_loader, criterion, optimizer, device, use_cutmix=True, alpha=1.0):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    total_samples = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        if i % 100 == 99:
            print(f'[{i+1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    return total_loss / total_samples

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
            val_loss += loss.item() * inputs.size(0)
            
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

    print(f'Validation Loss: {val_loss / total:.4f}')
    print(f'Top 1 Accuracy: {top1_accuracy:.4f}')
    print(f'Top 5 Accuracy: {top5_accuracy:.4f}')
    print(f'Superclass Accuracy: {superclass_acc:.4f}')
    
    return val_loss / total, top1_accuracy, top5_accuracy, superclass_acc

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(config)

    log_file = open(config['logging']['log_dir'] + "/wideResNet_efficientNet_train_output.log", "a")
    
    model = EfficientNetWideResNet(num_classes=config['model']['num_classes'], pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=10, delta=0.01)
    
    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["epochs"]}')
        log_file.write(f'Epoch {epoch}/{config["train"]["epochs"]}\n')
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, top1_accuracy, top5_accuracy, superclass_acc = validate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        top1_acc_history.append(top1_accuracy)
        top5_acc_history.append(top5_accuracy)
        superclass_acc_history.append(superclass_acc)

        print(f'Epoch {epoch}/{config["train"]["epochs"]} Results:')
        print(f'Top-1 Accuracy: {top1_accuracy:.4f}')
        print(f'Top-5 Accuracy: {top5_accuracy:.4f}')
        print(f'Superclass Accuracy: {superclass_acc:.4f}')
        
        log_file.write(f'Training Loss: {train_loss:.4f}\n')
        log_file.write(f'Validation Loss: {val_loss:.4f}\n')
        log_file.write(f'Top-1 Accuracy: {top1_accuracy:.4f}\n')
        log_file.write(f'Top-5 Accuracy: {top5_accuracy:.4f}\n')
        log_file.write(f'Superclass Accuracy: {superclass_acc:.4f}\n')

        if early_stopping(val_loss):
            print("Early stopping triggered")
            log_file.write("Early stopping triggered\n")
            break

        if epoch % config['train']['save_every'] == 0:
            checkpoint_path = os.path.join("./checkpoints", f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
    
    print("Training Complete!")
    log_file.write("Training Complete!\n")
    log_file.close()

def save_checkpoint(model, optimizer, epoch, filepath):
    print(f'Saving checkpoint at epoch {epoch}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    train_model(config)
