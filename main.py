import argparse
import torch
from pathlib import Path
from src.config import Config
from src.train import train_model
from src.utils import plot_training_history, predict_image, load_checkpoint
from src.model import create_model


def main():
    parser = argparse.ArgumentParser(description='Surface Crack Detection')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'test'],
                        help='Режим работы: train, predict, test')
    parser.add_argument('--image', type=str,
                        help='Путь к изображению для предсказания')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Путь к чекпоинту модели')
    args = parser.parse_args()

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

    print(f"Найдено изображений:")
    print(f"  С трещинами: {pos_count}")
    print(f"  Без трещин: {neg_count}")

    if pos_count == 0 or neg_count == 0:
        print("Ошибка: Недостаточно данных для обучения!")
        return

    if args.mode == 'train':
        print("Начало обучения...")
        model, history = train_model()
        print("Обучение завершено!")

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
        model = create_model()
        checkpoint = load_checkpoint(args.checkpoint, model)

        if checkpoint:
            # Предсказание
            class_name, confidence = predict_image(model, image_path)
            print(f"\nРезультат предсказания:")
            print(f"  Изображение: {image_path.name}")
            print(f"  Класс: {class_name}")
            print(f"  Уверенность: {confidence * 100:.2f}%")

    elif args.mode == 'test':
        print("Режим тестирования")
        # Здесь можно добавить код для тестирования на тестовом наборе


if __name__ == "__main__":
    main()