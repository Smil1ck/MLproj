import argparse
import torch
from pathlib import Path
from src.config import Config
from src.train import train_model
from src.utils import plot_training_history, predict_image, load_checkpoint
from src.model import create_model
from src.evaluate import evaluate_model, test_model_on_directory, batch_predict, load_and_test_checkpoint, \
    compare_checkpoints
from src.visualize import ResultsVisualizer
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Surface Crack Detection')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'test', 'evaluate',
                                 'compare', 'info', 'batch_predict', 'dashboard'],
                        help='Режим работы')
    parser.add_argument('--image', type=str,
                        help='Путь к изображению для предсказания')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Путь к чекпоинту модели')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'dynamic', 'gap', 'resnet'],
                        help='Тип модели')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Размер изображения')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--test_dir', type=str,
                        help='Директория для тестирования')
    parser.add_argument('--input_dir', type=str,
                        help='Директория с изображениями для батч-предсказаний')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Файл для сохранения результатов')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Порог уверенности')
    parser.add_argument('--visualize', action='store_true',
                        help='Создать визуализации результатов')
    parser.add_argument('--dashboard', action='store_true',
                        help='Создать интерактивный дашборд')

    args = parser.parse_args()

    # Обновляем конфиг из аргументов
    Config.MODEL_TYPE = args.model
    Config.IMG_SIZE = tuple(args.img_size)
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr

    # Инициализация визуализатора
    visualizer = ResultsVisualizer()

    # Проверка наличия данных
    if args.mode in ['train', 'test', 'evaluate', 'dashboard'] and args.test_dir is None:
        if not Config.DATA_DIR.exists():
            print(f"Ошибка: Папка с данными {Config.DATA_DIR} не найдена!")
            print("Пожалуйста, разместите данные в следующей структуре:")
            print("data/Positive/ - изображения с трещинами")
            print("data/Negative/ - изображения без трещин")
            return

    # Проверка наличия изображений
    if args.mode in ['train', 'test', 'evaluate', 'dashboard'] and args.test_dir is None:
        pos_count = len(list(Config.POSITIVE_DIR.glob("*.jpg")))
        neg_count = len(list(Config.NEGATIVE_DIR.glob("*.jpg")))
    elif args.test_dir:
        test_dir = Path(args.test_dir)
        pos_count = len(list((test_dir / "Positive").glob("*.jpg")))
        neg_count = len(list((test_dir / "Negative").glob("*.jpg")))
    else:
        pos_count = neg_count = 0

    print(f"=== КОНФИГУРАЦИЯ ===")
    print(f"Модель: {Config.MODEL_TYPE}")
    print(f"Размер изображения: {Config.IMG_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Эпохи: {Config.NUM_EPOCHS}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Устройство: {Config.DEVICE}")

    if args.mode in ['train', 'test', 'evaluate', 'dashboard']:
        print(f"\n=== ДАННЫЕ ===")
        print(f"С трещинами: {pos_count}")
        print(f"Без трещин: {neg_count}")

        if pos_count == 0 or neg_count == 0:
            print("Ошибка: Недостаточно данных для работы!")
            return

    if args.mode == 'train':
        print("\n=== НАЧАЛО ОБУЧЕНИЯ ===")
        model, history = train_model(model_type=args.model)
        print("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")

        # Визуализация истории обучения
        if args.visualize:
            visualizer.plot_training_history(history, args.model)
        else:
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
        print("\n=== ТЕСТИРОВАНИЕ МОДЕЛИ НА ДАННЫХ ===")

        # Определяем директорию для тестирования
        test_dir = Path(args.test_dir) if args.test_dir else Config.DATA_DIR

        # Загрузка модели
        model = create_model(args.model)

        if args.checkpoint == 'checkpoints/best_model.pth':
            args.checkpoint = f'checkpoints/best_{args.model}.pth'

        checkpoint = load_checkpoint(args.checkpoint, model, model_type=args.model)

        if checkpoint:
            # Тестирование на данных
            results = evaluate_model(
                model=model,
                data_dir=test_dir,
                batch_size=args.batch_size,
                threshold=args.threshold
            )

            # Добавляем информацию о модели
            results['model_name'] = args.model
            results['test_dir'] = str(test_dir)
            results['image_size'] = Config.IMG_SIZE

            # Сохранение результатов
            save_results(results, model_name=args.model, test_dir=str(test_dir))

            # Визуализация
            if args.visualize:
                visualizer.plot_performance_metrics(results, args.model)
                visualizer.plot_confusion_matrix(results['confusion_matrix'], args.model)

    elif args.mode == 'evaluate':
        print("\n=== ДЕТАЛЬНАЯ ОЦЕНКА МОДЕЛИ ===")

        # Определяем директорию для тестирования
        test_dir = Path(args.test_dir) if args.test_dir else Config.DATA_DIR

        # Загрузка модели из чекпоинта
        model = create_model(args.model)
        checkpoint = load_checkpoint(args.checkpoint, model, model_type=args.model)

        if checkpoint:
            # Полная оценка с метриками
            metrics = test_model_on_directory(
                model=model,
                test_dir=test_dir,
                batch_size=args.batch_size,
                model_name=args.model
            )

            if metrics:
                # Сохранение метрик
                save_metrics(metrics, model_name=args.model)

                # Визуализация
                if args.visualize:
                    visualizer.plot_performance_metrics(metrics, args.model)
                    visualizer.plot_confusion_matrix(metrics['confusion_matrix'], args.model)

    elif args.mode == 'dashboard':
        print("\n=== СОЗДАНИЕ ДАШБОРДА ===")

        # Определяем директорию для тестирования
        test_dir = Path(args.test_dir) if args.test_dir else Config.DATA_DIR

        # Загрузка модели из чекпоинта
        model = create_model(args.model)
        checkpoint = load_checkpoint(args.checkpoint, model, model_type=args.model)

        if checkpoint:
            # Тестирование модели
            metrics = test_model_on_directory(
                model=model,
                test_dir=test_dir,
                batch_size=args.batch_size,
                model_name=args.model
            )

            if metrics:
                # Создаем дашборд
                dashboard_dir = visualizer.create_dashboard(metrics, args.model)
                print(f"\nДашборд создан в: {dashboard_dir}")

    elif args.mode == 'batch_predict':
        if not args.input_dir:
            print("Ошибка: Укажите директорию с изображениями с помощью --input_dir")
            return

        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Ошибка: Директория {input_dir} не найдена")
            return

        print(f"\n=== БАТЧ-ПРЕДСКАЗАНИЯ ===")
        print(f"Входная директория: {input_dir}")
        print(f"Выходной файл: {args.output_csv}")

        # Загрузка модели
        model = create_model(args.model)
        checkpoint = load_checkpoint(args.checkpoint, model, model_type=args.model)

        if checkpoint:
            # Пакетное предсказание
            results_df = batch_predict(
                model=model,
                input_dir=input_dir,
                output_file=args.output_csv,
                batch_size=args.batch_size,
                threshold=args.threshold
            )

            print(f"\nОбработано {len(results_df)} изображений")
            print(f"Результаты сохранены в {args.output_csv}")

            # Визуализация результатов
            if args.visualize and not results_df.empty:
                visualizer.plot_batch_predictions(results_df, args.model)

    elif args.mode == 'compare':
        print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")

        # Сравниваем несколько чекпоинтов
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("best_*.pth"))
            if len(checkpoints) >= 2:
                print(f"Найдено {len(checkpoints)} чекпоинтов для сравнения")

                test_dir = Path(args.test_dir) if args.test_dir else Config.DATA_DIR
                df = compare_checkpoints(checkpoints, test_dir, batch_size=args.batch_size)

                if args.visualize and not df.empty:
                    # Загружаем метрики для всех моделей
                    metrics_list = []
                    for checkpoint in checkpoints:
                        try:
                            metrics = load_and_test_checkpoint(
                                checkpoint_path=checkpoint,
                                test_dir=test_dir,
                                batch_size=args.batch_size
                            )
                            if metrics:
                                metrics_list.append(metrics)
                        except:
                            continue

                    if len(metrics_list) >= 2:
                        visualizer.compare_multiple_models(metrics_list)
            else:
                print("Недостаточно чекпоинтов для сравнения")
        else:
            print("Директория checkpoints не найдена")

    elif args.mode == 'info':
        print("\n=== ИНФОРМАЦИЯ О МОДЕЛЯХ ===")
        print("\nДоступные модели:")
        print("1. simple - Простая CNN (только для 128x128)")
        print("2. dynamic - Динамическая CNN (любой размер)")
        print("3. gap - CNN с Global Average Pooling (любой размер)")
        print("4. resnet - Миниатюрная ResNet (любой размер)")

        print("\nДоступные чекпоинты:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.pth"))
            if checkpoints:
                for cp in checkpoints:
                    try:
                        data = torch.load(cp, map_location='cpu')
                        model_type = data.get('model_type', 'unknown')
                        accuracy = data.get('val_acc', 'N/A')
                        epoch = data.get('epoch', 'N/A')
                        size = data.get('img_size', 'N/A')
                        print(f"  - {cp.name}: type={model_type}, epoch={epoch}, acc={accuracy}%, size={size}")
                    except:
                        print(f"  - {cp.name}: error loading")
            else:
                print("  Нет чекпоинтов")
        else:
            print("  Директория checkpoints не существует")


def save_results(results, model_name, test_dir):
    """Сохранение результатов тестирования"""
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Сохранение в JSON
    json_file = results_dir / f"results_{model_name}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Сохранение в CSV
    csv_file = results_dir / f"results_{model_name}_{timestamp}.csv"
    results_df = pd.DataFrame(results['predictions'])
    results_df.to_csv(csv_file, index=False)

    print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
    print(f"Модель: {model_name}")
    print(f"Тестовая директория: {test_dir}")
    print(f"Всего изображений: {results['total_images']}")
    print(f"Верно классифицировано: {results['correct_predictions']}")
    print(f"Точность: {results['accuracy']:.2f}%")
    print(f"С трещинами (верно): {results['true_positives']}")
    print(f"С трещинами (ложно): {results['false_positives']}")
    print(f"Без трещин (верно): {results['true_negatives']}")
    print(f"Без трещин (ложно): {results['false_negatives']}")
    print(f"Время обработки: {results['processing_time']:.2f} сек")
    print(f"Результаты сохранены в:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")


def save_metrics(metrics, model_name):
    """Сохранение метрик оценки"""
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)

    # Сохранение метрик
    json_file = metrics_dir / f"metrics_{model_name}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== МЕТРИКИ ОЦЕНКИ ===")
    print(f"Модель: {model_name}")
    print(f"\nОсновные метрики:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")

    print(f"\nМатрица ошибок:")
    cm = metrics['confusion_matrix']
    print(f"  TP: {cm['true_positives']}, FP: {cm['false_positives']}")
    print(f"  FN: {cm['false_negatives']}, TN: {cm['true_negatives']}")

    print(f"\nДетали по классам:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    F1-Score: {class_metrics['f1_score']:.4f}")

    print(f"\nПроизводительность:")
    print(f"  Время обработки: {metrics['processing_time']:.2f} сек")
    print(f"  Изображений в секунду: {metrics['images_per_second']:.1f}")
    print(f"  Использование GPU: {metrics['gpu_used']:.1f}%")

    print(f"\nМетрики сохранены в: {json_file}")


if __name__ == "__main__":
    main()