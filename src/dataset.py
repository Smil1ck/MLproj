import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from .config import Config


class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Собираем пути к изображениям
        self.image_paths = []
        self.labels = []

        # Положительные (трещины) - метка 1
        positive_dir = self.root_dir / "Positive"
        if positive_dir.exists():
            for img_path in positive_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(1)

        # Отрицательные (без трещин) - метка 0
        negative_dir = self.root_dir / "Negative"
        if negative_dir.exists():
            for img_path in negative_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(0)

        # Разделение на train/val (80/20)
        total_samples = len(self.image_paths)
        split_idx = int(0.8 * total_samples)

        if mode == 'train':
            self.image_paths = self.image_paths[:split_idx]
            self.labels = self.labels[:split_idx]
        elif mode == 'val':
            self.image_paths = self.image_paths[split_idx:]
            self.labels = self.labels[split_idx:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms():
    """Аугментация для тренировки и базовые трансформации для валидации"""
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def get_dataloaders():
    """Создание DataLoader'ов"""
    train_transform, val_transform = get_transforms()

    train_dataset = CrackDataset(
        Config.DATA_DIR,
        transform=train_transform,
        mode='train'
    )

    val_dataset = CrackDataset(
        Config.DATA_DIR,
        transform=val_transform,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader