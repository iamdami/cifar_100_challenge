import torch
from torchvision import datasets, transforms

def get_dataloaders(config):
    data_root = config['data']['root']
    batch_size = config['train']['batch_size']
    num_workers = config['data']['num_workers']

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4) if "RandomCrop" in config['data']['augmentations'] else None,
        transforms.RandomHorizontalFlip() if "RandomHorizontalFlip" in config['data']['augmentations'] else None,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    train_transform = transforms.Compose([t for t in train_transform.transforms if t])

    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
