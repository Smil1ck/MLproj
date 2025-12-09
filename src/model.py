import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class SimpleCNN(nn.Module):
    """Очень простая CNN архитектура для 1050Ti (только для 128x128)"""

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
        # Проверка размера (опционально)
        if x.shape[2] != 128 or x.shape[3] != 128:
            print(f"Warning: SimpleCNN ожидает 128x128, получено {x.shape[2]}x{x.shape[3]}")

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


class DynamicCNN(nn.Module):
    """Динамическая CNN, работает с любым размером изображения"""

    def __init__(self, num_classes=2):
        super(DynamicCNN, self).__init__()

        # Сверточные слои (не зависят от размера)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Адаптивный пулинг для фиксирования размера
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Фиксированный размер
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Конволюции с активациями и пулингом
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Адаптивный пулинг (приводит к фиксированному размеру)
        x = self.adaptive_pool(x)

        # Выравнивание
        x = x.view(x.size(0), -1)

        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class GAPCNN(nn.Module):
    """CNN с Global Average Pooling, работает с любым размером"""

    def __init__(self, num_classes=2):
        super(GAPCNN, self).__init__()

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

        # Глобальный средний пулинг
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Полносвязные слои
        self.fc1 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Конволюции с активациями и пулингом
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Глобальный пулинг
        x = self.global_pool(x)

        # Выравнивание
        x = x.view(x.size(0), -1)

        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class ResNetMini(nn.Module):
    """Миниатюрная версия ResNet"""

    def __init__(self, num_classes=2):
        super(ResNetMini, self).__init__()

        # Начальный слой
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Residual блоки
        self.res_block1 = self._make_res_block(16, 32)
        self.res_block2 = self._make_res_block(32, 64)

        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Полносвязные слои
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Начальный слой
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Residual блоки
        identity = x
        x = self.res_block1(x)
        x += identity  # Skip connection
        x = self.relu(x)

        identity = x
        x = self.res_block2(x)
        x += identity
        x = self.relu(x)

        # Глобальный пулинг
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Полносвязные слои
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def create_model(model_type=None):
    """Создание и инициализация модели"""
    if model_type is None:
        model_type = Config.MODEL_TYPE

    if model_type == "simple":
        model = SimpleCNN(num_classes=Config.NUM_CLASSES)
        print(f"Создана модель SimpleCNN (только для 128x128)")
    elif model_type == "dynamic":
        model = DynamicCNN(num_classes=Config.NUM_CLASSES)
        print(f"Создана модель DynamicCNN (работает с любым размером)")
    elif model_type == "gap":
        model = GAPCNN(num_classes=Config.NUM_CLASSES)
        print(f"Создана модель GAPCNN (работает с любым размером)")
    elif model_type == "resnet":
        model = ResNetMini(num_classes=Config.NUM_CLASSES)
        print(f"Создана модель ResNetMini (работает с любым размером)")
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    return model.to(Config.DEVICE)


def get_model_info(model):
    """Информация о модели"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_class": model.__class__.__name__
    }


def count_parameters(model):
    """Подсчет параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)