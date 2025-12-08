import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .config import Config


def save_checkpoint(state, filename='checkpoint.pth'):
    """Сохранение чекпоинта"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """Загрузка чекпоинта"""
    if Path(filename).exists():
        checkpoint = torch.load(filename, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")
        return checkpoint
    else:
        print(f"No checkpoint found at {filename}")
        return None


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
    plt.savefig(Config.SAVE_DIR / 'training_history.png', dpi=150)
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