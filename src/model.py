import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class SimpleCNN(nn.Module):
    """Очень простая CNN архитектура для 1050Ti"""

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Вычисляем размер фичей после сверток
        # Для 128x128 изображения: 128 -> 64 -> 32 -> 16
        self.feature_size = 64 * 16 * 16

        # Полносвязные слои
        self.fc1 = nn.Linear(self.feature_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Конволюции с активациями и пулингом
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Выравнивание
        x = x.view(x.size(0), -1)

        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


def create_model():
    """Создание и инициализация модели"""
    model = SimpleCNN(num_classes=Config.NUM_CLASSES)
    return model.to(Config.DEVICE)


def count_parameters(model):
    """Подсчет параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)