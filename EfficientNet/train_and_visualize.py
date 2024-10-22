import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet

# Global Variables
LOG_FILE = "./logs/efficientNet_train_output.log"
NUM_CLASSES = 100
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.1
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
MODEL_NAME = 'efficientnet-b0'

# CIFAR-100 Superclass mapping
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

# Helper functions
def log_message(message):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        max_k = min(k, output.size(1))
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct[:k].reshape(-1).float().sum(0, keepdim=True).item()

def superclass_accuracy(output, target):
    with torch.no_grad():
        pred_super = [superclass_mapping[p.item()] for p in torch.argmax(output, dim=1)]
        target_super = [superclass_mapping[t.item()] for t in target]
        correct_super = sum(p == t for p, t in zip(pred_super, target_super))
        return correct_super

# Dataset Preparation
def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader

# Model Definition
def get_model():
    model = EfficientNet.from_pretrained(MODEL_NAME, num_classes=NUM_CLASSES)
    return model

# Training Function
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_top1, correct_top5, correct_superclass, total = 0, 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct_top1 += (outputs.argmax(1) == labels).sum().item()
        correct_top5 += top_k_accuracy(outputs, labels, k=5)
        correct_superclass += superclass_accuracy(outputs, labels)
        total += labels.size(0)

    avg_loss = running_loss / total
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    superclass_acc = correct_superclass / total

    return avg_loss, top1_acc, top5_acc, superclass_acc

# Validation Function
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1, correct_top5, correct_superclass, total = 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct_top1 += (outputs.argmax(1) == labels).sum().item()
            correct_top5 += top_k_accuracy(outputs, labels, k=5)
            correct_superclass += superclass_accuracy(outputs, labels)
            total += labels.size(0)

    avg_loss = running_loss / total
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    superclass_acc = correct_superclass / total

    return avg_loss, top1_acc, top5_acc, superclass_acc

# Main Training Loop
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    train_loader, test_loader = get_dataloaders()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    train_loss_history, val_loss_history = [], []
    top1_acc_history, top5_acc_history, superclass_acc_history = [], [], []

    for epoch in range(EPOCHS):
        log_message(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_top1, train_top5, train_superclass = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss_history.append(train_loss)
        top1_acc_history.append(train_top1)
        top5_acc_history.append(train_top5)
        superclass_acc_history.append(train_superclass)
        log_message(f"Training Loss: {train_loss:.4f}, Top-1 Accuracy: {train_top1:.4f}, Top-5 Accuracy: {train_top5:.4f}, Superclass Accuracy: {train_superclass:.4f}")
        
        # Validate
        val_loss, val_top1, val_top5, val_superclass = validate(model, test_loader, criterion, device)
        val_loss_history.append(val_loss)
        log_message(f"Validation Loss: {val_loss:.4f}, Top-1 Accuracy: {val_top1:.4f}, Top-5 Accuracy: {val_top5:.4f}, Superclass Accuracy: {val_superclass:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch+1}.pth")

    # Save log for visualization
    np.savez("./logs/train_results.npz", train_loss=train_loss_history, val_loss=val_loss_history, top1_acc=top1_acc_history, top5_acc=top5_acc_history, superclass_acc=superclass_acc_history)
    log_message("Training Complete")

if __name__ == "__main__":
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    train_model()
