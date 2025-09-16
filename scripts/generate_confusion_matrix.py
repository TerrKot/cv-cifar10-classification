#!/usr/bin/env python3
"""
Генерация confusion matrix для всех моделей
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.simply_cnn import SimpleCNN
from models.improve_cnn import ImprovedCNN

def load_model(model_path, model_type, device):
    """Загрузка модели"""
    if model_type == 'baseline':
        model = SimpleCNN(num_classes=10)
    elif model_type == 'improved':
        model = ImprovedCNN(num_classes=10, dropout_rate=0.3)
    else:
        raise ValueError("model_type должен быть 'baseline' или 'improved'")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def generate_confusion_matrix(model, test_loader, device, model_name, save_path):
    """Генерация confusion matrix"""
    
    print(f"Генерация confusion matrix для {model_name}...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Создаем confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Классы CIFAR-10
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Создаем график
    plt.figure(figsize=(12, 10))
    
    # Нормализованная confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Вычисляем метрики
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Precision, Recall, F1 для каждого класса
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Сохраняем детальный отчет
    report_path = save_path.replace('.png', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"ДЕТАЛЬНЫЙ ОТЧЕТ - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Общая точность: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        f.write("МЕТРИКИ ПО КЛАССАМ:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Класс':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
        f.write("-" * 50 + "\n")
        
        for i, class_name in enumerate(classes):
            f.write(f"{class_name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f}\n")
        
        f.write(f"\nСРЕДНИЕ МЕТРИКИ:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Средняя Precision: {np.mean(precision):.4f}\n")
        f.write(f"Средняя Recall: {np.mean(recall):.4f}\n")
        f.write(f"Средняя F1-Score: {np.mean(f1):.4f}\n")
    
    print(f"✅ Confusion matrix сохранена в {save_path}")
    print(f"✅ Детальный отчет сохранен в {report_path}")
    
    return cm, accuracy, precision, recall, f1

def main():
    """Главная функция"""
    
    print("📊 ГЕНЕРАЦИЯ CONFUSION MATRIX ДЛЯ ВСЕХ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Создаем DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Создаем директории
    os.makedirs('results/confusion_matrices', exist_ok=True)
    
    # Список моделей для анализа
    models_to_analyze = [
        ('models/baseline_5epochs_model.pth', 'baseline', 'Baseline CNN (5 эпох)'),
        ('models/improved_5epochs_model.pth', 'improved', 'Improved CNN (5 эпох)'),
        ('models/baseline_25epochs_model.pth', 'baseline', 'Baseline CNN (25 эпох)'),
        ('models/improved_25epochs_model.pth', 'improved', 'Improved CNN (25 эпох)')
    ]
    
    results = {}
    
    for model_path, model_type, model_name in models_to_analyze:
        if os.path.exists(model_path):
            try:
                # Загружаем модель
                model = load_model(model_path, model_type, device)
                
                # Генерируем confusion matrix
                save_path = f'results/confusion_matrices/{model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()}_confusion_matrix.png'
                cm, accuracy, precision, recall, f1 = generate_confusion_matrix(
                    model, test_loader, device, model_name, save_path
                )
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': np.mean(precision),
                    'recall': np.mean(recall),
                    'f1': np.mean(f1),
                    'confusion_matrix': cm
                }
                
            except Exception as e:
                print(f"❌ Ошибка при анализе {model_name}: {e}")
        else:
            print(f"⚠️  Модель не найдена: {model_path}")
    
    # Создаем сравнительный отчет
    if results:
        create_comparison_report(results)
    
    print(f"\n🎉 ГЕНЕРАЦИЯ CONFUSION MATRIX ЗАВЕРШЕНА!")

def create_comparison_report(results):
    """Создание сравнительного отчета"""
    
    filename = 'results/confusion_matrices/comparison_report.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("СРАВНИТЕЛЬНЫЙ ОТЧЕТ CONFUSION MATRIX\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("МЕТРИКИ ПО МОДЕЛЯМ:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Модель':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
        f.write("-" * 50 + "\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                   f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}\n")
        
        f.write(f"\nЛУЧШИЕ РЕЗУЛЬТАТЫ:\n")
        f.write("-" * 30 + "\n")
        
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_precision = max(results.items(), key=lambda x: x[1]['precision'])
        best_recall = max(results.items(), key=lambda x: x[1]['recall'])
        best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
        
        f.write(f"Лучшая точность: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})\n")
        f.write(f"Лучшая precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})\n")
        f.write(f"Лучший recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})\n")
        f.write(f"Лучший F1-Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})\n")
    
    print(f"✅ Сравнительный отчет сохранен в {filename}")

if __name__ == "__main__":
    main()
