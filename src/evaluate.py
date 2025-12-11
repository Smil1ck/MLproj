import torch
import numpy as np
from pathlib import Path
import time
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as transforms
from tqdm import tqdm
import psutil
import GPUtil

from .config import Config
from .dataset import CrackDataset


def evaluate_model(model, data_dir, batch_size=32, threshold=0.5):
    """
    Оценивает модель на данных в директории

    Args:
        model: обученная модель
        data_dir: путь к директории с данными (должна содержать Positive и Negative)
        batch_size: размер батча для обработки
        threshold: порог уверенности

    Returns:
        словарь с результатами
    """
    print(f"Оценка модели на данных из: {data_dir}")

    # Создание DataLoader
    test_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CrackDataset(
        root_dir=data_dir,
        transform=test_transform,
        mode='test'  # Используем все данные
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Переводим модель в режим оценки
    model.eval()
    model.to(Config.DEVICE)

    # Подсчет метрик
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_filepaths = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Оценка")):
            images = images.to(Config.DEVICE)

            # Предсказания
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Получаем предсказанные классы и уверенности
            confidences, predicted = torch.max(probabilities, 1)

            # Сохраняем результаты
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

            # Сохраняем пути к файлам (для батча)
            start_idx = batch_idx * batch_size
            for i in range(len(images)):
                if hasattr(test_dataset, 'image_paths'):
                    idx = start_idx + i
                    if idx < len(test_dataset.image_paths):
                        all_filepaths.append(str(test_dataset.image_paths[idx]))
                else:
                    all_filepaths.append(f"batch_{batch_idx}_img_{i}")

    end_time = time.time()
    processing_time = end_time - start_time

    # Преобразуем в numpy массивы
    predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)
    confidences = np.array(all_confidences)

    # Вычисляем метрики
    correct_predictions = np.sum(predictions == true_labels)
    accuracy = 100.0 * correct_predictions / len(true_labels)

    # Матрица ошибок
    cm = confusion_matrix(true_labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # Precision, Recall, F1
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

    # Собираем детальные результаты
    results = []
    for i in range(len(predictions)):
        results.append({
            'filepath': all_filepaths[i] if i < len(all_filepaths) else f"unknown_{i}",
            'true_label': 'Crack' if true_labels[i] == 1 else 'No Crack',
            'predicted_label': 'Crack' if predictions[i] == 1 else 'No Crack',
            'confidence': float(confidences[i]),
            'correct': bool(predictions[i] == true_labels[i]),
            'crack_probability': float(confidences[i] if predictions[i] == 1 else 1 - confidences[i])
        })

    # Мониторинг использования ресурсов
    gpu_usage = get_gpu_usage()
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    return {
        'total_images': len(true_labels),
        'correct_predictions': int(correct_predictions),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'processing_time': float(processing_time),
        'images_per_second': len(true_labels) / processing_time,
        'gpu_used': float(gpu_usage),
        'cpu_used': float(cpu_usage),
        'memory_used': float(memory_usage),
        'predictions': results,
        'confusion_matrix': {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }
    }


def test_model_on_directory(model, test_dir, batch_size=32, model_name='unknown'):
    """
    Полное тестирование модели на директории с метриками
    """
    print(f"\nТестирование модели '{model_name}' на директории: {test_dir}")

    # Проверяем структуру директории
    test_dir = Path(test_dir)
    positive_dir = test_dir / "Positive"
    negative_dir = test_dir / "Negative"

    if not positive_dir.exists() or not negative_dir.exists():
        print("Ошибка: директория должна содержать подпапки Positive и Negative")
        return None

    # Загружаем данные
    positive_images = list(positive_dir.glob("*.jpg")) + list(positive_dir.glob("*.png"))
    negative_images = list(negative_dir.glob("*.jpg")) + list(negative_dir.glob("*.png"))

    total_images = len(positive_images) + len(negative_images)
    print(f"Найдено {len(positive_images)} изображений с трещинами")
    print(f"Найдено {len(negative_images)} изображений без трещин")
    print(f"Всего: {total_images} изображений")

    if total_images == 0:
        print("Ошибка: нет изображений для тестирования")
        return None

    # Оцениваем модель
    metrics = evaluate_model(model, test_dir, batch_size=batch_size)

    # Добавляем информацию о модели
    metrics['model_name'] = model_name
    metrics['test_dir'] = str(test_dir)
    metrics['image_size'] = Config.IMG_SIZE

    # Детали по классам
    if metrics['total_images'] > 0:
        metrics['class_metrics'] = {
            'Crack': {
                'precision': metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
                if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0,
                'recall': metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
                if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0,
                'f1_score': 2 * metrics['true_positives'] / (
                            2 * metrics['true_positives'] + metrics['false_positives'] + metrics['false_negatives'])
                if (2 * metrics['true_positives'] + metrics['false_positives'] + metrics['false_negatives']) > 0 else 0
            },
            'No Crack': {
                'precision': metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_negatives'])
                if (metrics['true_negatives'] + metrics['false_negatives']) > 0 else 0,
                'recall': metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_positives'])
                if (metrics['true_negatives'] + metrics['false_positives']) > 0 else 0,
                'f1_score': 2 * metrics['true_negatives'] / (
                            2 * metrics['true_negatives'] + metrics['false_positives'] + metrics['false_negatives'])
                if (2 * metrics['true_negatives'] + metrics['false_positives'] + metrics['false_negatives']) > 0 else 0
            }
        }

    return metrics


def batch_predict(model, input_dir, output_file='predictions.csv', batch_size=32, threshold=0.5):
    """
    Пакетное предсказание для всех изображений в директории

    Args:
        model: обученная модель
        input_dir: директория с изображениями
        output_file: файл для сохранения результатов
        batch_size: размер батча
        threshold: порог уверенности

    Returns:
        DataFrame с результатами
    """
    print(f"Пакетное предсказание для изображений в: {input_dir}")

    input_dir = Path(input_dir)

    # Находим все изображения
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(ext))
        image_paths.extend(input_dir.glob(ext.upper()))

    print(f"Найдено {len(image_paths)} изображений")

    if len(image_paths) == 0:
        print("Ошибка: нет изображений для обработки")
        return pd.DataFrame()

    # Трансформации
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Подготавливаем модель
    model.eval()
    model.to(Config.DEVICE)

    results = []
    start_time = time.time()

    # Обработка батчами
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Обработка"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        # Загружаем и преобразуем изображения
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Ошибка загрузки {img_path}: {e}")
                batch_images.append(torch.zeros(3, *Config.IMG_SIZE))

        # Создаем батч
        batch_tensor = torch.stack(batch_images).to(Config.DEVICE)

        # Предсказание
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

        # Сохраняем результаты
        for j, img_path in enumerate(batch_paths):
            pred_class = predicted[j].item()
            confidence = confidences[j].item()

            is_crack = pred_class == 1
            crack_prob = confidence if is_crack else 1 - confidence

            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'predicted_class': 'Crack' if is_crack else 'No Crack',
                'confidence': float(confidence),
                'crack_probability': float(crack_prob),
                'has_crack': bool(is_crack),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })

    end_time = time.time()
    processing_time = end_time - start_time

    # Создаем DataFrame
    df = pd.DataFrame(results)

    # Сохраняем в CSV
    df.to_csv(output_file, index=False)

    # Статистика
    crack_count = df['has_crack'].sum()
    no_crack_count = len(df) - crack_count

    print(f"\n=== СТАТИСТИКА БАТЧ-ПРЕДСКАЗАНИЙ ===")
    print(f"Обработано изображений: {len(df)}")
    print(f"Время обработки: {processing_time:.2f} сек")
    print(f"Скорость: {len(df) / processing_time:.1f} изображений/сек")
    print(f"С трещинами: {crack_count} ({crack_count / len(df) * 100:.1f}%)")
    print(f"Без трещин: {no_crack_count} ({no_crack_count / len(df) * 100:.1f}%)")
    print(f"Результаты сохранены в: {output_file}")

    return df


def get_gpu_usage():
    """Получает использование GPU"""
    try:
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed / gpus[0].memoryTotal * 100
    except:
        pass
    return 0.0


def load_and_test_checkpoint(checkpoint_path, test_dir, model_type=None, batch_size=32):
    """
    Загружает модель из чекпоинта и тестирует на данных

    Args:
        checkpoint_path: путь к чекпоинту
        test_dir: директория для тестирования
        model_type: тип модели (если None, определяется из чекпоинта)
        batch_size: размер батча

    Returns:
        метрики тестирования
    """
    print(f"Загрузка модели из чекпоинта: {checkpoint_path}")

    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Определяем тип модели
    if model_type is None:
        model_type = checkpoint.get('model_type', 'simple')

    # Определяем размер изображения
    img_size = checkpoint.get('img_size', Config.IMG_SIZE)
    Config.IMG_SIZE = img_size

    # Создаем модель
    from .model import create_model
    model = create_model(model_type)

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)

    print(f"Модель загружена: {model_type}")
    print(f"Размер изображения: {img_size}")
    print(f"Эпоха обучения: {checkpoint.get('epoch', 'N/A')}")
    print(f"Точность валидации: {checkpoint.get('val_acc', 'N/A')}%")

    # Тестируем модель
    metrics = test_model_on_directory(
        model=model,
        test_dir=test_dir,
        batch_size=batch_size,
        model_name=f"{model_type}_checkpoint"
    )

    if metrics:
        metrics['checkpoint_info'] = {
            'checkpoint_path': checkpoint_path,
            'model_type': model_type,
            'training_epoch': checkpoint.get('epoch', 'N/A'),
            'training_accuracy': checkpoint.get('val_acc', 'N/A'),
            'checkpoint_size_mb': Path(checkpoint_path).stat().st_size / (1024 * 1024)
        }

    return metrics


def compare_checkpoints(checkpoint_paths, test_dir, batch_size=32):
    """
    Сравнивает несколько чекпоинтов на одних данных

    Args:
        checkpoint_paths: список путей к чекпоинтам
        test_dir: директория для тестирования
        batch_size: размер батча

    Returns:
        DataFrame с сравнением
    """
    print(f"Сравнение {len(checkpoint_paths)} чекпоинтов")

    results = []

    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Ошибка: чекпоинт {checkpoint_path} не найден")
            continue

        print(f"\nТестирование чекпоинта: {checkpoint_path.name}")

        try:
            metrics = load_and_test_checkpoint(
                checkpoint_path=checkpoint_path,
                test_dir=test_dir,
                batch_size=batch_size
            )

            if metrics:
                results.append({
                    'checkpoint': checkpoint_path.name,
                    'model_type': metrics.get('model_name', 'unknown'),
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'images_per_second': metrics['images_per_second'],
                    'true_positives': metrics['true_positives'],
                    'false_positives': metrics['false_positives'],
                    'true_negatives': metrics['true_negatives'],
                    'false_negatives': metrics['false_negatives'],
                    'training_epoch': metrics.get('checkpoint_info', {}).get('training_epoch', 'N/A'),
                    'training_accuracy': metrics.get('checkpoint_info', {}).get('training_accuracy', 'N/A'),
                    'file_size_mb': metrics.get('checkpoint_info', {}).get('checkpoint_size_mb', 0)
                })
        except Exception as e:
            print(f"Ошибка тестирования {checkpoint_path}: {e}")

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('accuracy', ascending=False)

        # Сохраняем сравнение
        output_file = "checkpoints_comparison.csv"
        df.to_csv(output_file, index=False)

        print(f"\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
        print(df.to_string())
        print(f"\nСравнение сохранено в: {output_file}")

        return df
    else:
        print("Нет результатов для сравнения")
        return pd.DataFrame()