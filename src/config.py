import torch
from pathlib import Path


class Config:
    # Пути
    DATA_DIR = Path("data")
    POSITIVE_DIR = DATA_DIR / "Positive"
    NEGATIVE_DIR = DATA_DIR / "Negative"

    # Параметры модели (оптимизированы под вашу 1050Ti)
    BATCH_SIZE = 16  # Меньше из-за ограниченной памяти GPU
    IMG_SIZE = (128, 128)  # Уменьшаем с 227x227 для экономии памяти
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # Архитектура
    NUM_CLASSES = 2

    # Настройки обучения
    NUM_WORKERS = 4  # i5-9400f имеет 6 ядер
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Сохранение модели
    SAVE_DIR = Path("checkpoints")
    SAVE_DIR.mkdir(exist_ok=True)

    # Отладка
    DEBUG = False
    PRINT_EVERY = 50