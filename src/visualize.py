import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib

# Настройки для красивого отображения
plt.style.use('seaborn-v0_8-darkgrid')
matplotlib.rcParams['figure.figsize'] = [12, 8]
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['font.size'] = 12
sns.set_palette("husl")


class ResultsVisualizer:
    """Класс для визуализации результатов тестирования моделей"""

    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Цвета для моделей
        self.model_colors = {
            'simple': '#FF6B6B',
            'dynamic': '#4ECDC4',
            'gap': '#45B7D1',
            'resnet': '#96CEB4',
            'default': '#95A5A6'
        }

        # Стили для графиков
        self.plot_styles = {
            'line_width': 3,
            'marker_size': 100,
            'grid_alpha': 0.3,
            'title_size': 16,
            'label_size': 14
        }

    def get_model_color(self, model_name):
        """Получает цвет для модели"""
        model_key = model_name.lower().split('_')[0] if '_' in model_name else model_name.lower()
        return self.model_colors.get(model_key, self.model_colors['default'])

    def plot_confusion_matrix(self, cm_data, model_name, save_path=None):
        """Визуализация матрицы ошибок"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Извлекаем данные
        tp = cm_data['true_positives']
        fp = cm_data['false_positives']
        fn = cm_data['false_negatives']
        tn = cm_data['true_negatives']

        # Создаем матрицу
        cm = np.array([[tp, fp], [fn, tn]])

        # Визуализация
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=['Predicted Crack', 'Predicted No Crack'],
                    yticklabels=['Actual Crack', 'Actual No Crack'],
                    ax=ax)

        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)

        # Добавляем процент точности
        accuracy = (tp + tn) / (tp + fp + fn + tn) * 100
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%',
                transform=ax.transAxes, ha='center', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()
        return fig

    def plot_metrics_comparison(self, metrics_list, save_path=None):
        """Сравнение метрик нескольких моделей"""
        if not metrics_list:
            print("No metrics to compare")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Подготавливаем данные
        model_names = [m.get('model_name', f'Model_{i}') for i, m in enumerate(metrics_list)]

        # Основные метрики для сравнения
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy (%)', 'Precision', 'Recall', 'F1-Score']

        for idx, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
            ax = axes[idx]
            values = []
            colors = []

            for metrics in metrics_list:
                value = metrics.get(metric, 0)
                if metric == 'accuracy':
                    value = value  # Уже в процентах
                else:
                    value = value * 100  # Конвертируем в проценты

                values.append(value)
                colors.append(self.get_model_color(metrics.get('model_name', 'default')))

            # Создаем bar plot
            bars = ax.bar(range(len(values)), values, color=colors, edgecolor='black')

            # Добавляем значения на столбцы
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

            ax.set_title(label, fontsize=14)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Percentage (%)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)

        plt.suptitle('Model Performance Comparison', fontsize=18, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")

        plt.show()
        return fig

    def plot_performance_metrics(self, metrics, model_name, save_path=None):
        """Визуализация всех метрик одной модели"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        model_color = self.get_model_color(model_name)

        # 1. Основные метрики (radar chart)
        ax = axes[0]
        metrics_radar = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_values = [metrics.get(m, 0) * 100 for m in metrics_radar]
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        # Создаем radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
        metrics_values += metrics_values[:1]
        angles += angles[:1]
        metrics_labels += metrics_labels[:1]

        ax = plt.subplot(2, 3, 1, polar=True)
        ax.plot(angles, metrics_values, 'o-', linewidth=2, color=model_color)
        ax.fill(angles, metrics_values, alpha=0.25, color=model_color)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics_labels[:-1])
        ax.set_title('Model Performance Radar', fontsize=14, pad=20)
        ax.grid(True)

        # 2. Матрица ошибок (упрощенная)
        ax = axes[1]
        cm_data = metrics.get('confusion_matrix', {})
        tp = cm_data.get('true_positives', 0)
        fp = cm_data.get('false_positives', 0)
        fn = cm_data.get('false_negatives', 0)
        tn = cm_data.get('true_negatives', 0)

        cm_matrix = np.array([[tp, fp], [fn, tn]])
        im = ax.imshow(cm_matrix, cmap='YlOrRd', interpolation='nearest')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Crack', 'No Crack'])
        ax.set_yticklabels(['Crack', 'No Crack'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix', fontsize=14)

        # Добавляем значения в ячейки
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_matrix[i, j]),
                        ha='center', va='center',
                        color='white' if cm_matrix[i, j] > (cm_matrix.max() / 2) else 'black')

        plt.colorbar(im, ax=ax)

        # 3. Статистика по классам
        ax = axes[2]
        class_data = metrics.get('class_metrics', {})
        if class_data:
            classes = list(class_data.keys())
            precision_vals = [class_data[c].get('precision', 0) * 100 for c in classes]
            recall_vals = [class_data[c].get('recall', 0) * 100 for c in classes]
            f1_vals = [class_data[c].get('f1_score', 0) * 100 for c in classes]

            x = np.arange(len(classes))
            width = 0.25

            ax.bar(x - width, precision_vals, width, label='Precision', color='#3498db')
            ax.bar(x, recall_vals, width, label='Recall', color='#2ecc71')
            ax.bar(x + width, f1_vals, width, label='F1-Score', color='#9b59b6')

            ax.set_xlabel('Class')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Metrics by Class', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(classes)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Производительность
        ax = axes[3]
        perf_metrics = {
            'Processing Time (s)': metrics.get('processing_time', 0),
            'Images/s': metrics.get('images_per_second', 0),
            'GPU Used (%)': metrics.get('gpu_used', 0),
            'CPU Used (%)': metrics.get('cpu_used', 0),
        }

        bars = ax.bar(range(len(perf_metrics)), list(perf_metrics.values()),
                      color=[model_color, '#e74c3c', '#f39c12', '#8e44ad'])

        ax.set_title('Performance Metrics', fontsize=14)
        ax.set_xticks(range(len(perf_metrics)))
        ax.set_xticklabels(list(perf_metrics.keys()), rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        # Добавляем значения
        for bar, value in zip(bars, perf_metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)

        # 5. Распределение уверенности (если есть)
        ax = axes[4]
        predictions = metrics.get('predictions', [])
        if predictions:
            confidences = [p.get('confidence', 0) * 100 for p in predictions]
            correct = [p.get('correct', False) for p in predictions]

            correct_conf = [c for c, cr in zip(confidences, correct) if cr]
            incorrect_conf = [c for c, cr in zip(confidences, correct) if not cr]

            ax.hist([correct_conf, incorrect_conf], bins=20,
                    label=['Correct', 'Incorrect'],
                    color=['#2ecc71', '#e74c3c'],
                    alpha=0.7, stacked=True)

            ax.set_xlabel('Confidence (%)')
            ax.set_ylabel('Count')
            ax.set_title('Confidence Distribution', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6. Информация о модели
        ax = axes[5]
        ax.axis('off')

        info_text = f"""
        Model: {model_name}
        {'=' * 30}

        Test Results:
        • Total Images: {metrics.get('total_images', 0)}
        • Correct: {metrics.get('correct_predictions', 0)}
        • Accuracy: {metrics.get('accuracy', 0):.2f}%

        Confusion Matrix:
        • True Positives: {tp}
        • False Positives: {fp}
        • False Negatives: {fn}
        • True Negatives: {tn}

        Performance:
        • Time: {metrics.get('processing_time', 0):.2f}s
        • Speed: {metrics.get('images_per_second', 0):.1f} img/s
        • GPU: {metrics.get('gpu_used', 0):.1f}%
        • CPU: {metrics.get('cpu_used', 0):.1f}%
        """

        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle(f'Model Evaluation: {model_name}', fontsize=18, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Performance metrics saved to {save_path}")

        plt.show()
        return fig

    def plot_batch_predictions(self, predictions_df, model_name, save_path=None):
        """Визуализация результатов пакетного предсказания"""
        if predictions_df.empty:
            print("No predictions to visualize")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        model_color = self.get_model_color(model_name)

        # 1. Распределение классов
        ax = axes[0]
        class_counts = predictions_df['predicted_class'].value_counts()
        wedges, texts, autotexts = ax.pie(class_counts.values, labels=class_counts.index,
                                          autopct='%1.1f%%', startangle=90,
                                          colors=['#e74c3c', '#2ecc71'])

        ax.set_title('Class Distribution', fontsize=14)

        # 2. Распределение уверенности
        ax = axes[1]
        sns.histplot(data=predictions_df, x='confidence', bins=30,
                     hue='predicted_class', multiple='stack',
                     palette={'Crack': '#e74c3c', 'No Crack': '#2ecc71'},
                     ax=ax)
        ax.set_title('Confidence Distribution', fontsize=14)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')

        # 3. Вероятность трещин
        ax = axes[2]
        crack_probs = predictions_df['crack_probability']
        ax.hist(crack_probs, bins=30, color=model_color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        ax.set_title('Crack Probability Distribution', fontsize=14)
        ax.set_xlabel('Crack Probability')
        ax.set_ylabel('Count')
        ax.legend()

        # 4. Топ-10 самых уверенных трещин
        ax = axes[3]
        top_cracks = predictions_df[predictions_df['has_crack']].nlargest(10, 'confidence')
        bars = ax.barh(range(len(top_cracks)), top_cracks['confidence'],
                       color=model_color, edgecolor='black')
        ax.set_yticks(range(len(top_cracks)))
        ax.set_yticklabels(top_cracks['filename'].str[:20] + '...')
        ax.invert_yaxis()
        ax.set_title('Top 10 Most Confident Cracks', fontsize=14)
        ax.set_xlabel('Confidence')

        # Добавляем значения
        for bar, value in zip(bars, top_cracks['confidence']):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}', ha='left', va='center')

        # 5. Топ-10 самых уверенных "без трещин"
        ax = axes[4]
        top_no_cracks = predictions_df[~predictions_df['has_crack']].nlargest(10, 'confidence')
        bars = ax.barh(range(len(top_no_cracks)), top_no_cracks['confidence'],
                       color='#2ecc71', edgecolor='black')
        ax.set_yticks(range(len(top_no_cracks)))
        ax.set_yticklabels(top_no_cracks['filename'].str[:20] + '...')
        ax.invert_yaxis()
        ax.set_title('Top 10 Most Confident No-Cracks', fontsize=14)
        ax.set_xlabel('Confidence')

        # Добавляем значения
        for bar, value in zip(bars, top_no_cracks['confidence']):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}', ha='left', va='center')

        # 6. Статистика
        ax = axes[5]
        ax.axis('off')

        total = len(predictions_df)
        crack_count = predictions_df['has_crack'].sum()
        no_crack_count = total - crack_count

        avg_confidence = predictions_df['confidence'].mean()
        avg_crack_prob = predictions_df['crack_probability'].mean()

        stats_text = f"""
        Batch Prediction Summary
        {'=' * 30}

        Total Images: {total}
        • With Cracks: {crack_count} ({crack_count / total * 100:.1f}%)
        • Without Cracks: {no_crack_count} ({no_crack_count / total * 100:.1f}%)

        Confidence Statistics:
        • Average Confidence: {avg_confidence:.3f}
        • Average Crack Probability: {avg_crack_prob:.3f}

        Model: {model_name}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle(f'Batch Predictions Analysis - {model_name}', fontsize=18, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch predictions visualization saved to {save_path}")

        plt.show()
        return fig

    def plot_training_history(self, history, model_name, save_path=None):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        model_color = self.get_model_color(model_name)

        # 1. Loss кривая
        ax = axes[0]
        epochs = range(1, len(history['train_loss']) + 1)

        ax.plot(epochs, history['train_loss'], 'b-', linewidth=2,
                label='Training Loss', marker='o')
        ax.plot(epochs, history['val_loss'], 'r-', linewidth=2,
                label='Validation Loss', marker='s')

        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Accuracy кривая
        ax = axes[1]
        ax.plot(epochs, history['train_acc'], 'b-', linewidth=2,
                label='Training Accuracy', marker='o')
        ax.plot(epochs, history['val_acc'], 'r-', linewidth=2,
                label='Validation Accuracy', marker='s')

        ax.set_title('Training and Validation Accuracy', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Learning rate (если есть)
        ax = axes[2]
        if 'learning_rate' in history:
            ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2,
                    marker='^')
            ax.set_title('Learning Rate Schedule', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No learning rate data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate Schedule', fontsize=14)

        # 4. Статистика обучения
        ax = axes[3]
        ax.axis('off')

        final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0

        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        best_val_epoch = history['val_acc'].index(best_val_acc) + 1 if best_val_acc else 0

        info_text = f"""
        Training Summary - {model_name}
        {'=' * 30}

        Final Epoch Results:
        • Training Accuracy: {final_train_acc:.2f}%
        • Validation Accuracy: {final_val_acc:.2f}%
        • Training Loss: {final_train_loss:.4f}
        • Validation Loss: {final_val_loss:.4f}

        Best Validation:
        • Best Accuracy: {best_val_acc:.2f}% (Epoch {best_val_epoch})

        Training Statistics:
        • Total Epochs: {len(epochs)}
        • Initial Train Acc: {history['train_acc'][0]:.2f}%
        • Final Train Acc: {final_train_acc:.2f}%
        • Improvement: {final_train_acc - history['train_acc'][0]:.2f}%
        """

        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle(f'Training History - {model_name}', fontsize=18, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history visualization saved to {save_path}")

        plt.show()
        return fig

    def create_dashboard(self, metrics_data, model_name, save_dir=None):
        """Создает полноценный дашборд с результатами"""
        if save_dir is None:
            save_dir = self.results_dir / f"dashboard_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        print(f"Creating dashboard for model: {model_name}")
        print(f"Saving to: {save_dir}")

        # 1. Основной график производительности
        perf_path = save_dir / f"performance_{model_name}.png"
        self.plot_performance_metrics(metrics_data, model_name, save_path=perf_path)

        # 2. Матрица ошибок
        if 'confusion_matrix' in metrics_data:
            cm_path = save_dir / f"confusion_matrix_{model_name}.png"
            self.plot_confusion_matrix(metrics_data['confusion_matrix'], model_name, save_path=cm_path)

        # 3. Сохраняем метрики в JSON
        json_path = save_dir / f"metrics_{model_name}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        # 4. Сохраняем сводную таблицу
        if 'predictions' in metrics_data:
            csv_path = save_dir / f"predictions_{model_name}.csv"
            df = pd.DataFrame(metrics_data['predictions'])
            df.to_csv(csv_path, index=False)

        # 5. Создаем HTML отчет
        self.create_html_report(metrics_data, model_name, save_dir)

        print(f"\nDashboard created successfully!")
        print(f"Files saved in: {save_dir}")

        return save_dir

    def create_html_report(self, metrics, model_name, save_dir):
        """Создает HTML отчет с результатами"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Evaluation Report - {model_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 36px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .metric-label {{
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .images {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .image-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                .image-item img {{
                    width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Model Evaluation Report</h1>
                    <h2>Model: {model_name}</h2>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('accuracy', 0):.2f}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <div class="metric-value">{metrics.get('precision', 0):.3f}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                        <div class="metric-value">{metrics.get('recall', 0):.3f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                        <div class="metric-value">{metrics.get('f1_score', 0):.3f}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                </div>

                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Images Processed</td>
                        <td>{metrics.get('total_images', 0)}</td>
                    </tr>
                    <tr>
                        <td>Correct Predictions</td>
                        <td>{metrics.get('correct_predictions', 0)}</td>
                    </tr>
                    <tr>
                        <td>True Positives</td>
                        <td>{metrics.get('confusion_matrix', {{}}).get('true_positives', 0)}</td>
                    </tr>
                    <tr>
                        <td>False Positives</td>
                        <td>{metrics.get('confusion_matrix', {{}}).get('false_positives', 0)}</td>
                    </tr>
                    <tr>
                        <td>False Negatives</td>
                        <td>{metrics.get('confusion_matrix', {{}}).get('false_negatives', 0)}</td>
                    </tr>
                    <tr>
                        <td>True Negatives</td>
                        <td>{metrics.get('confusion_matrix', {{}}).get('true_negatives', 0)}</td>
                    </tr>
                    <tr>
                        <td>Processing Time</td>
                        <td>{metrics.get('processing_time', 0):.2f} seconds</td>
                    </tr>
                    <tr>
                        <td>Images per Second</td>
                        <td>{metrics.get('images_per_second', 0):.1f}</td>
                    </tr>
                </table>

                <div class="images">
                    <h2>Visualizations</h2>
                    <div class="image-grid">
                        <div class="image-item">
                            <h3>Performance Metrics</h3>
                            <img src="performance_{model_name}.png" alt="Performance Metrics">
                        </div>
                        <div class="image-item">
                            <h3>Confusion Matrix</h3>
                            <img src="confusion_matrix_{model_name}.png" alt="Confusion Matrix">
                        </div>
                    </div>
                </div>

                <h2>Model Information</h2>
                <table>
                    <tr>
                        <td>Test Directory</td>
                        <td>{metrics.get('test_dir', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Image Size</td>
                        <td>{metrics.get('image_size', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>GPU Usage</td>
                        <td>{metrics.get('gpu_used', 0):.1f}%</td>
                    </tr>
                    <tr>
                        <td>CPU Usage</td>
                        <td>{metrics.get('cpu_used', 0):.1f}%</td>
                    </tr>
                </table>

                <div class="footer">
                    <p>Report generated automatically by Crack Detection System</p>
                    <p>Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        html_path = save_dir / f"report_{model_name}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report saved to: {html_path}")

        return html_path

    def compare_multiple_models(self, metrics_list, save_path=None):
        """Сравнивает несколько моделей и создает отчет"""
        if len(metrics_list) < 2:
            print("Need at least 2 models for comparison")
            return

        fig = self.plot_metrics_comparison(metrics_list, save_path)

        # Создаем таблицу сравнения
        comparison_data = []
        for metrics in metrics_list:
            comparison_data.append({
                'Model': metrics.get('model_name', 'Unknown'),
                'Accuracy (%)': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'Processing Time (s)': metrics.get('processing_time', 0),
                'Images/s': metrics.get('images_per_second', 0),
                'Total Images': metrics.get('total_images', 0),
            })

        df = pd.DataFrame(comparison_data)

        # Сохраняем таблицу
        if save_path:
            csv_path = Path(save_path).with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            print(f"Comparison table saved to: {csv_path}")

        return df, fig