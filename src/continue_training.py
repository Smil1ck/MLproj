import argparse
import torch
import os
from pathlib import Path
from src.config import Config
from src.model import create_model
from src.dataset import get_dataloaders
from src.utils import load_checkpoint, save_checkpoint
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def continue_training(
        checkpoint_path="checkpoints/last_model.pth",
        additional_epochs=5,
        new_lr=None,
        new_batch_size=None,
        new_img_size=None,
        model_name=None
):
    """
    Продолжение обучения модели из чекпоинта

    Аргументы:
    - checkpoint_path: путь к сохраненной модели
    - additional_epochs: сколько эпох дообучать
    - new_lr: новая скорость обучения (если None, берется из чекпоинта)
    - new_batch_size: новый размер батча
    - new_img_size: новый размер изображения (tuple)
    - model_name: имя для сохранения дообученной модели
    """

    print(f"=== ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ ===")
    print(f"Чекпоинт: {checkpoint_path}")
    print(f"Дополнительные эпохи: {additional_epochs}")

    # Проверка наличия чекпоинта
    if not Path(checkpoint_path).exists():
        print(f"Ошибка: чекпоинт {checkpoint_path} не найден!")
        return None

    # Загрузка чекпоинта для получения информации
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Определяем тип модели из чекпоинта
    model_type = checkpoint.get('model_type', 'simple')
    original_img_size = checkpoint.get('img_size', (128, 128))

    # Обновляем конфиг
    Config.MODEL_TYPE = model_type

    if new_img_size:
        Config.IMG_SIZE = tuple(new_img_size)
    else:
        Config.IMG_SIZE = original_img_size

    if new_batch_size:
        Config.BATCH_SIZE = new_batch_size

    print(f"Тип модели: {model_type}")
    print(f"Размер изображения: {Config.IMG_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE}")

    # Создание модели
    model = create_model(model_type)

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Загружены веса модели")

    # Создаем оптимизатор
    if new_lr:
        lr = new_lr
    else:
        lr = checkpoint.get('learning_rate', Config.LEARNING_RATE)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Загружено состояние оптимизатора")

    # Функция потерь
    criterion = nn.CrossEntropyLoss()

    # Получение номера последней эпохи
    last_epoch = checkpoint.get('epoch', 0)
    print(f"Последняя обученная эпоха: {last_epoch}")
    print(f"Лучшая точность: {checkpoint.get('val_acc', 'N/A')}%")
    print(f"Learning rate: {lr}")

    # Загрузка истории если есть
    history = checkpoint.get('history', {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    })

    # Загрузка данных (с новым размером если изменился)
    train_loader, val_loader = get_dataloaders()

    # Переводим модель в режим обучения
    model.train()

    # Цикл дообучения
    for epoch in range(additional_epochs):
        current_epoch = last_epoch + epoch + 1
        print(f"\nЭпоха {current_epoch}/{last_epoch + additional_epochs}")
        print("-" * 40)

        # Обучение на одной эпохе
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Дообучение")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Статистика
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % Config.PRINT_EVERY == 0:
                pbar.set_postfix({
                    'Loss': f'{train_loss / (batch_idx + 1):.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Вычисление метрик
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Обновление истории
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Сохранение промежуточного чекпоинта
        if (epoch + 1) % 2 == 0:  # Сохранять каждые 2 эпохи
            checkpoint_filename = Config.SAVE_DIR / f"checkpoint_{model_type}_epoch_{current_epoch}.pth"
            save_checkpoint({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'model_type': model_type,
                'img_size': Config.IMG_SIZE,
                'model_class': model.__class__.__name__
            }, filename=checkpoint_filename)
            print(f"Сохранен чекпоинт: {checkpoint_filename}")

    # Определяем имя для финальной модели
    if model_name is None:
        model_name = f"continued_{model_type}.pth"

    final_path = Config.SAVE_DIR / model_name
    save_checkpoint({
        'epoch': last_epoch + additional_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'model_type': model_type,
        'img_size': Config.IMG_SIZE,
        'model_class': model.__class__.__name__
    }, filename=final_path)

    print(f"\n=== ДООБУЧЕНИЕ ЗАВЕРШЕНО ===")
    print(f"Финальная модель сохранена: {final_path}")
    print(f"Итоговая точность: {val_acc:.2f}%")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Продолжение обучения модели')

    # Обязательные аргументы
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Путь к чекпоинту для продолжения обучения')

    # Дополнительные аргументы
    parser.add_argument('--epochs', type=int, default=5,
                        help='Количество дополнительных эпох (по умолчанию: 5)')

    parser.add_argument('--lr', type=float, default=None,
                        help='Новая скорость обучения')

    parser.add_argument('--batch_size', type=int, default=None,
                        help='Новый размер батча')

    parser.add_argument('--img_size', type=int, nargs=2, default=None,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Новый размер изображения')

    parser.add_argument('--output', type=str, default=None,
                        help='Имя выходного файла модели')

    args = parser.parse_args()

    # Проверка наличия файла
    if not os.path.exists(args.checkpoint):
        print(f"Ошибка: файл {args.checkpoint} не найден!")
        print("Доступные чекпоинты:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for file in checkpoints_dir.glob("*.pth"):
                print(f"  - {file}")
        return

    # Запуск дообучения
    continue_training(
        checkpoint_path=args.checkpoint,
        additional_epochs=args.epochs,
        new_lr=args.lr,
        new_batch_size=args.batch_size,
        new_img_size=args.img_size,
        model_name=args.output
    )


if __name__ == "__main__":
    main()