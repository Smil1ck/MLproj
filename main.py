import argparse
import torch
from pathlib import Path
from src.config import Config
from src.train import train_model
from src.utils import plot_training_history, predict_image, load_checkpoint, compare_models
from src.model import create_model, SimpleCNN, DynamicCNN, GAPCNN, ResNetMini


def main():
    parser = argparse.ArgumentParser(description='Surface Crack Detection')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'test', 'compare', 'info'],
                        help='Режим работы: train, predict, test, compare, info')
    parser.add_argument('--image', type=str,
                        help='Путь к изображению для предсказания')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Путь к чекпоинту модели')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'dynamic', 'gap', 'resnet'],
                        help='Тип модели: simple, dynamic, gap, resnet')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Размер изображения (по умолчанию: 128 128)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    # Обновляем конфиг из аргументов
    Config.MODEL_TYPE = args.model
    Config.IMG_SIZE = tuple(args.img_size)
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr

    # Проверка наличия данных
    if not Config.DATA_DIR.exists():
        print(f"Ошибка: Папка с данными {Config.DATA_DIR} не найдена!")
        print("Пожалуйста, разместите данные в следующей структуре:")
        print("data/Positive/ - изображения с трещинами")
        print("data/Negative/ - изображения без трещин")
        return

    # Проверка наличия изображений
    pos_count = len(list(Config.POSITIVE_DIR.glob("*.jpg")))
    neg_count = len(list(Config.NEGATIVE_DIR.glob("*.jpg")))

    print(f"=== КОНФИГУРАЦИЯ ===")
    print(f"Модель: {Config.MODEL_TYPE}")
    print(f"Размер изображения: {Config.IMG_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Эпохи: {Config.NUM_EPOCHS}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Устройство: {Config.DEVICE}")
    print(f"\n=== ДАННЫЕ ===")
    print(f"С трещинами: {pos_count}")
    print(f"Без трещин: {neg_count}")

    if pos_count == 0 or neg_count == 0:
        print("Ошибка: Недостаточно данных для обучения!")
        return

    if args.mode == 'train':
        print("\n=== НАЧАЛО ОБУЧЕНИЯ ===")
        model, history = train_model(model_type=args.model)
        print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")

        # Визуализация результатов
        plot_training_history(history)

    elif args.mode == 'predict':
        if not args.image:
            print("Ошибка: Укажите путь к изображению с помощью --image")
            return

        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Ошибка: Изображение {image_path} не найдено")
            return

        # Загрузка модели
        model = create_model(args.model)

        # Определяем имя чекпоинта по умолчанию
        if args.checkpoint == 'checkpoints/best_model.pth':
            args.checkpoint = f'checkpoints/best_{args.model}.pth'

        checkpoint = load_checkpoint(args.checkpoint, model, model_type=args.model)

        if checkpoint:
            # Предсказание
            class_name, confidence = predict_image(model, image_path)
            print(f"\n=== РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ ===")
            print(f"Модель: {args.model}")
            print(f"Изображение: {image_path.name}")
            print(f"Класс: {class_name}")
            print(f"Уверенность: {confidence * 100:.2f}%")
            print(f"Разрешение: {Config.IMG_SIZE}")

    elif args.mode == 'test':
        print("Режим тестирования")
        # Здесь можно добавить код для тестирования на тестовом наборе

    elif args.mode == 'compare':
        print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")

        # Создаем все модели
        models = {
            'simple': SimpleCNN(num_classes=2),
            'dynamic': DynamicCNN(num_classes=2),
            'gap': GAPCNN(num_classes=2),
            'resnet': ResNetMini(num_classes=2)
        }

        # Сравниваем
        results = compare_models(models)

        print("\nРезультаты сравнения:")
        print("-" * 60)
        for name, result in results.items():
            print(f"Модель: {name}")
            print(f"  Точность: {result['accuracy']:.2f}%")
            print(f"  Параметры: {result['params']:,}")
            print(f"  Обучаемые: {result['trainable']:,}")
            print("-" * 60)

    elif args.mode == 'info':
        print("\n=== ИНФОРМАЦИЯ О МОДЕЛЯХ ===")
        print("\nДоступные модели:")
        print("1. simple - Простая CNN (только для 128x128)")
        print("2. dynamic - Динамическая CNN (любой размер)")
        print("3. gap - CNN с Global Average Pooling (любой размер)")
        print("4. resnet - Миниатюрная ResNet (любой размер)")

        print("\nРекомендации для 1050Ti (4GB):")
        print(f"- simple: IMG_SIZE=(128,128), BATCH_SIZE=16")
        print(f"- dynamic: IMG_SIZE=(160,160), BATCH_SIZE=12")
        print(f"- gap: IMG_SIZE=(192,192), BATCH_SIZE=8")
        print(f"- resnet: IMG_SIZE=(128,128), BATCH_SIZE=8")


if __name__ == "__main__":
    main()