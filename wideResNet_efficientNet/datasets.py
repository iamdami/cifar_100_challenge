import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(config):
    # Data augmentations for training and validation
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

    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root=config['data']['root'], train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=config['data']['root'], train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    return train_loader, test_loader
