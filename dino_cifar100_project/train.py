import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # AMP 관련 모듈
from dino_vit import DINO_ViT  # DINO 모델을 정의한 파일
from datasets import get_dataloaders  # 데이터셋 로드 함수
import os

# Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 한 epoch 동안의 학습 함수 정의
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # AMP: autocast 사용
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # AMP: GradScaler로 스케일링
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        if i % 100 == 99:  # 매 100번째 배치마다 출력
            print(f'[{i+1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# 검증 함수 정의
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
    return val_loss

# 모델 체크포인트 저장 함수
def save_checkpoint(model, optimizer, epoch, filepath):
    print(f'Saving checkpoint at epoch {epoch}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

# 모델 학습 함수 정의
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    train_loader, val_loader = get_dataloaders(config)
    
    # 모델 설정
    model = DINO_ViT(hidden_dim=config['model']['hidden_dim'],
                     num_heads=config['model']['num_heads'],
                     depth=config['model']['depth'],
                     patch_size=config['model']['patch_size'],
                     num_classes=config['model']['num_classes']).to(device)
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    
    # 손실 함수
    criterion = torch.nn.CrossEntropyLoss()

    # AMP: GradScaler 사용
    scaler = GradScaler()

    # 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])

    # Early Stopping 설정
    early_stopping = EarlyStopping(patience=5)

    # 체크포인트 경로 설정
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 학습 시작
    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["epochs"]}')
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # 스케줄러 업데이트
        scheduler.step()

        # Early Stopping 체크
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if epoch % config['train']['save_every'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
    
    print("Training Complete!")

# 메인 함수
if __name__ == "__main__":
    import yaml
    
    # 설정 파일 로드
    with open("config_dino_cifar100.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # 모델 학습 시작
    train_model(config)
