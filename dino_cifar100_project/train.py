import torch
import torch.optim as optim
from dino_vit import DINO_ViT  # DINO 모델을 정의한 파일
from datasets import get_dataloaders  # datasets.py에서 데이터 로드
import os

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputㅁs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # 매 100번째 배치마다 출력
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
    print(f'Saving checkpoint at epoch {epoch}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    train_loader, val_loader = get_dataloaders(config)
    
    # 모델 설정
    model = DINO_ViT(config).to(device)  # hidden_dim 관련 부분 제거
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    
    # 손실 함수
    criterion = torch.nn.CrossEntropyLoss()

    # 체크포인트 경로 설정
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 학습 시작
    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["epochs"]}')
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = validate(model, val_loader, criterion, device)
        
        if epoch % config['train']['save_every'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
    
    print("Training Complete!")


if __name__ == "__main__":
    import yaml
    
    # 설정 파일 로드
    with open("config_dino_cifar100.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # 모델 학습 시작
    train_model(config)
