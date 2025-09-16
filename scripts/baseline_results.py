#!/usr/bin/env python3
"""
Baseline тестирование необученной модели
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import sys

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.simply_cnn import SimpleCNN, count_parameters

def test_baseline_model():
    """Тестирование необученной базовой модели"""
    
    print("🧪 BASELINE ТЕСТИРОВАНИЕ НЕОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 60)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Создаем модель
    model = SimpleCNN(num_classes=10).to(device)
    total_params = count_parameters(model)
    print(f"Архитектура: SimpleCNN")
    print(f"Общее количество параметров: {total_params:,}")
    
    # Создаем DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"Тестовых образцов: {len(test_dataset):,}")
    print(f"Тестовых батчей: {len(test_loader)}")
    
    # Тестирование
    print(f"\n🔍 ТЕСТИРОВАНИЕ НЕОБУЧЕННОЙ МОДЕЛИ...")
    
    model.eval()
    correct = 0
    total = 0
    batch_accuracies = []
    batch_times = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_start = time.time()
            
            data, target = data.to(device), target.to(device)
            
            # Прямой проход
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            # Статистика
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            batch_correct = (predicted == target).sum().item()
            batch_total = target.size(0)
            batch_acc = 100. * batch_correct / batch_total
            batch_accuracies.append(batch_acc)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Выводим прогресс каждые 2 батча
            if (batch_idx + 1) % 2 == 0 or batch_idx == 0:
                print(f"  Батч {batch_idx + 1:2d}: {batch_acc:6.1f}% "
                      f"({batch_correct:3d}/{batch_total:3d}) "
                      f"время: {batch_time*1000:.1f}ms")
    
    total_time = time.time() - start_time
    
    # Результаты
    overall_accuracy = 100. * correct / total
    avg_batch_accuracy = np.mean(batch_accuracies)
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"Протестировано батчей: {len(test_loader)}")
    print(f"Общее количество образцов: {total:,}")
    print(f"Правильных предсказаний: {correct:,}")
    print(f"Средняя точность по батчам: {avg_batch_accuracy:.2f}%")
    print(f"Общая точность: {overall_accuracy:.2f}%")
    print(f"Общее время тестирования: {total_time:.2f} сек")
    print(f"Средний time на батч: {avg_batch_time*1000:.2f}ms")
    print(f"Стандартное отклонение: {std_batch_time*1000:.2f}ms")
    print(f"Скорость: {total/avg_batch_time:.0f} изображений/сек")
    
    # Детальная статистика по батчам
    print(f"\n📈 ТОЧНОСТЬ ПО БАТЧАМ:")
    for i, acc in enumerate(batch_accuracies):
        print(f"Батч {i+1:2d}: {acc:6.1f}%")
    
    # Анализ результатов
    print(f"\n🔍 АНАЛИЗ РЕЗУЛЬТАТОВ:")
    
    # Случайная точность для 10 классов
    random_accuracy = 100.0 / 10  # 10%
    print(f"Случайная точность (10 классов): {random_accuracy:.1f}%")
    
    if overall_accuracy < random_accuracy * 0.5:
        print(f"❌ Модель работает хуже случайного выбора")
    elif overall_accuracy < random_accuracy * 1.5:
        print(f"⚠️  Модель работает примерно как случайный выбор")
    else:
        print(f"✅ Модель показывает признаки обучения")
    
    # Проверка на переобучение
    if max(batch_accuracies) - min(batch_accuracies) > 50:
        print(f"⚠️  Большой разброс точности по батчам (возможно, нестабильность)")
    else:
        print(f"✅ Стабильная работа по батчам")
    
    # Сохранение результатов
    print(f"\n💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
    
    # Создаем директории
    os.makedirs('results/5_epochs', exist_ok=True)
    
    # Сохраняем детальный отчет
    filename = 'results/5_epochs/baseline_results.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("БАЗОВЫЕ РЕЗУЛЬТАТЫ CNN ДО ОБУЧЕНИЯ\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Архитектура: SimpleCNN\n")
        f.write(f"Общее количество параметров: {total_params:,}\n")
        f.write(f"Устройство: {device}\n\n")
        
        f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:\n")
        f.write(f"Протестировано батчей: {len(test_loader)}\n")
        f.write(f"Общее количество образцов: {total:,}\n")
        f.write(f"Средняя точность по батчам: {avg_batch_accuracy:.2f}%\n")
        f.write(f"Общая точность: {overall_accuracy:.2f}%\n\n")
        
        f.write("ТОЧНОСТЬ ПО БАТЧАМ:\n")
        for i, acc in enumerate(batch_accuracies):
            f.write(f"Батч {i+1:2d}: {acc:6.1f}%\n")
        
        f.write(f"\nВЫВОД: Модель показывает случайную точность ~{overall_accuracy:.1f}% ")
        f.write(f"(ожидаемо для необученной модели). Готова к обучению.\n")
    
    print(f"✅ Отчет сохранен в {filename}")
    
    # Создаем график
    print(f"📊 Создание графика...")
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График точности по батчам
    batch_indices = list(range(1, len(batch_accuracies) + 1))
    ax1.plot(batch_indices, batch_accuracies, 'b-', linewidth=2, marker='o')
    ax1.axhline(y=random_accuracy, color='r', linestyle='--', 
                label=f'Случайная точность ({random_accuracy:.1f}%)')
    ax1.set_xlabel('Номер батча')
    ax1.set_ylabel('Точность (%)')
    ax1.set_title('Точность по батчам (необученная модель)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Гистограмма времени обработки батчей
    ax2.hist(batch_times, bins=10, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=avg_batch_time, color='r', linestyle='--', 
                label=f'Среднее время ({avg_batch_time*1000:.1f}ms)')
    ax2.set_xlabel('Время обработки батча (сек)')
    ax2.set_ylabel('Количество батчей')
    ax2.set_title('Распределение времени обработки')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/5_epochs/baseline_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ График сохранен в results/5_epochs/baseline_test_results.png")
    
    print(f"\n🎉 BASELINE ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"Созданы файлы:")
    print(f"  - {filename} (отчет)")
    print(f"  - results/5_epochs/baseline_test_results.png (график)")
    
    return overall_accuracy

def main():
    """Главная функция"""
    
    try:
        accuracy = test_baseline_model()
        
        # Проверяем, что результаты разумные
        if accuracy < 5 or accuracy > 20:
            print(f"\n⚠️  Неожиданная точность: {accuracy:.2f}%")
            print("Ожидается ~10% для необученной модели")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
