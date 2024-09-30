import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # AMP 관련 모듈
from dino_vit import DINO_ViT  # DINO 모델을 정의한 파일
from datasets import get_dataloaders  # 데이터셋 로드 함수
import os

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

# train_model 함수 수정
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

    # 체크포인트 경로 설정
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 학습 시작
    for epoch in range(1, config['train']['epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["epochs"]}')
        train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        validate(model, val_loader, criterion, device)
        
        if epoch % config['train']['save_every'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
    
    print("Training Complete!")
