import torch
from pathlib import Path


class Config:
    # Пути
    DATA_DIR = Path("data")
    POSITIVE_DIR = DATA_DIR / "Positive"
    NEGATIVE_DIR = DATA_DIR / "Negative"

    # Параметры модели
    MODEL_TYPE = "simple"  # "simple", "dynamic", "gap"
    BATCH_SIZE = 16
    IMG_SIZE = (128, 128)
    NUM_EPOCHS = 4
    LEARNING_RATE = 0.001

    # Архитектура
    NUM_CLASSES = 2

    # Настройки обучения
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Сохранение модели
    SAVE_DIR = Path("checkpoints")
    SAVE_DIR.mkdir(exist_ok=True)

    # Отладка
    DEBUG = False
    PRINT_EVERY = 50