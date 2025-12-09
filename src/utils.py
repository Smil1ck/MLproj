import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .config import Config
from .model import create_model


def save_checkpoint(state, filename='checkpoint.pth'):
    """Сохранение чекпоинта"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

    # Проверка размера файла
    size_mb = Path(filename).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


def load_checkpoint(filename, model=None, optimizer=None, model_type=None):
    """Загрузка чекпоинта"""
    if not Path(filename).exists():
        print(f"No checkpoint found at {filename}")
        return None

    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    # Определяем тип модели
    if model_type is None:
        model_type = checkpoint.get('model_type', 'simple')

    # Создаем модель, если не предоставлена
    if model is None:
        from .model import create_model
        model = create_model(model_type)

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from {filename}")
    print(f"Model type: {model_type}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")

    return checkpoint


def plot_training_history(history):
    """Визуализация процесса обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # График loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # График accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = Config.SAVE_DIR / f'training_history_{Config.MODEL_TYPE}.png'
    plt.savefig(save_path, dpi=150)
    print(f"Training history saved to {save_path}")
    plt.show()


def predict_image(model, image_path):
    """Предсказание для одного изображения"""
    model.eval()

    # Загрузка и преобразование изображения
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = output.max(1)

    # Интерпретация результата
    class_names = ['No Crack', 'Crack']
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[0][predicted.item()].item()

    return predicted_class, confidence


def compare_models(models_dict):
    """Сравнение нескольких моделей"""
    results = {}

    for name, model in models_dict.items():
        # Тестируем на небольшом наборе
        from .dataset import get_dataloaders
        _, val_loader = get_dataloaders()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        results[name] = {
            'accuracy': accuracy,
            'params': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

    return results