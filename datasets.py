import os
import torch
from torchvision import datasets, transforms

def get_dataloaders(config):
    """
    학습 및 평가를 위한 데이터 로더를 반환하는 함수.
    
    Args:
        config: 설정 파일에서 로드된 설정 값들
        
    Returns:
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
    """
    # 데이터셋 경로 및 설정 불러오기
    data_root = config['data']['root']
    batch_size = config['train']['batch_size']
    num_workers = config['data']['num_workers']
    
    # 데이터셋 전처리 (augmentations 적용)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 여기에 리사이즈 추가
        transforms.RandomCrop(224, padding=4) if "RandomCrop" in config['data']['augmentations'] else None,
        transforms.RandomHorizontalFlip() if "RandomHorizontalFlip" in config['data']['augmentations'] else None,
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2) if "ColorJitter" in config['data']['augmentations'] else None,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    # transform 중 None 항목 제거
    train_transform = transforms.Compose([t for t in train_transform.transforms if t])

    # 학습용 데이터셋
    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    
    # 평가용 데이터셋 전처리 (augmentations 없이)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 평가 데이터도 동일하게 리사이즈
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_transform)
    
    # 데이터 로더 설정
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
