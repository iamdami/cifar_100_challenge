import torch
import torch.optim as optim
from resnet18_model import ResNet18
from datasets import get_dataloaders
import os

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{i+1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

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

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def train_model(config):
    device = torch.device(config['train']['device'])
    train_loader, val_loader = get_dataloaders(config)
    model = ResNet18(config['model']['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["epochs"]}')
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        validate(model, val_loader, criterion, device)
        if epoch % config['train']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

if __name__ == "__main__":
    import yaml
    with open("config_resnet_cifar100.yaml", "r") as file:
        config = yaml.safe_load(file)
    train_model(config)
