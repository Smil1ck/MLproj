import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path
from .config import Config
from .model import create_model
from .dataset import get_dataloaders
from .utils import save_checkpoint, load_checkpoint


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Обновление progress bar
        if batch_idx % Config.PRINT_EVERY == 0:
            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Валидация"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss / len(pbar):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model():
    """Основной цикл обучения"""
    # Инициализация
    train_loader, val_loader = get_dataloaders()
    model = create_model()

    # Loss и optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    print(f"Training on: {Config.DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Цикл обучения
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 30)

        # Обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )

        # Валидация
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Обновление learning rate
        scheduler.step(val_loss)

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, filename=Config.SAVE_DIR / 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Сохранение последней модели
    save_checkpoint({
        'epoch': Config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history
    }, filename=Config.SAVE_DIR / 'last_model.pth')

    return model, history