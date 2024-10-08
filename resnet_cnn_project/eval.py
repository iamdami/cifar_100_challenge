import torch
from models.resnet_cnn import ResNetWithCNN
from datasets import get_dataloaders
import os

def load_checkpoint(model, optimizer, filepath):
    if os.path.isfile(filepath):
        print(f"Loading checkpoint from '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"No checkpoint found at '{filepath}'")

def validate(model, val_loader, criterion, device):
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

def evaluate_model(config, checkpoint
