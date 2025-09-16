#!/usr/bin/env python3
"""
Улучшенная CNN для классификации CIFAR-10
Включает:
- Dropout для регуляризации
- Batch Normalization для стабилизации обучения
- Data Augmentation для увеличения разнообразия данных
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys

class ImprovedCNN(nn.Module):
    """Улучшенная CNN для CIFAR-10 с Dropout и BatchNorm"""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        
        # Сверточные блоки с BatchNorm и Dropout
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2d = nn.Dropout2d(0.2)  # Spatial dropout
        
        # Полносвязные слои
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Активация
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Первый блок: Conv -> BN -> ReLU -> Pool -> Dropout2D
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        # Второй блок: Conv -> BN -> ReLU -> Pool -> Dropout2D
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        # Третий блок: Conv -> BN -> ReLU -> Pool -> Dropout2D
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        
        # Четвертый блок: Conv -> BN -> ReLU -> Pool -> Dropout2D
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        
        # Разворачиваем в вектор
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои с Dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def get_augmented_data_loaders(batch_size=64, shuffle=True):
    """Создание DataLoader с Data Augmentation для CIFAR-10"""
    
    # Трансформации для тренировочного набора (с аугментацией)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Случайное горизонтальное отражение
        transforms.RandomRotation(10),            # Случайный поворот на ±10 градусов
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение цвета
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Случайный сдвиг
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet нормализация
    ])
    
    # Трансформации для тестового набора (без аугментации)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Загрузка датасетов
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Создание DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader

def count_parameters(model):
    """Подсчет общего количества параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TrainingLogger:
    """Класс для логирования процесса обучения"""
    
    def __init__(self, model_name="ImprovedCNN"):
        self.model_name = model_name
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs = []
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Логирование результатов эпохи"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        
    def save_logs(self, filename='improved_training_logs.txt'):
        """Сохранение логов в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"ЛОГИ ОБУЧЕНИЯ {self.model_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Эпох: {len(self.epochs)}\n")
            f.write(f"Финальная точность на валидации: {self.val_accuracies[-1]:.2f}%\n")
            f.write(f"Финальная loss на валидации: {self.val_losses[-1]:.4f}\n\n")
            
            f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
            f.write("Эпоха | Train Loss | Train Acc | Val Loss | Val Acc\n")
            f.write("-" * 50 + "\n")
            
            for i, epoch in enumerate(self.epochs):
                f.write(f"{epoch:5d} | {self.train_losses[i]:9.4f} | {self.train_accuracies[i]:8.2f}% | "
                       f"{self.val_losses[i]:8.4f} | {self.val_accuracies[i]:7.2f}%\n")
        
        print(f"✅ Логи сохранены в {filename}")

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Обучение модели на одной эпохе"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        output = model(data)
        loss = criterion(output, target)
        
        # Обратный проход
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Прогресс каждые 100 батчей
        if batch_idx % 100 == 0:
            print(f'  Батч {batch_idx}/{len(train_loader)}: '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Валидация модели на одной эпохе"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def plot_training_results(logger, save_path='improved_training_results.png'):
    """Построение графиков результатов обучения"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График Loss
    ax1.plot(logger.epochs, logger.train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(logger.epochs, logger.val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{logger.model_name}: Loss по эпохам')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График Accuracy
    ax2.plot(logger.epochs, logger.train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(logger.epochs, logger.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{logger.model_name}: Accuracy по эпохам')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Графики сохранены в {save_path}")

def train_improved_model(epochs=5, learning_rate=0.001, batch_size=64, dropout_rate=0.5):
    """Основная функция обучения улучшенной модели"""
    
    print("🚀 ОБУЧЕНИЕ УЛУЧШЕННОЙ CNN ДЛЯ CIFAR-10")
    print("=" * 60)
    print("Улучшения:")
    print("  ✓ Dropout для регуляризации")
    print("  ✓ Batch Normalization для стабилизации")
    print("  ✓ Data Augmentation для разнообразия данных")
    print("  ✓ Более глубокая архитектура")
    print("=" * 60)
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Создаем модель
    model = ImprovedCNN(num_classes=10, dropout_rate=dropout_rate).to(device)
    total_params = count_parameters(model)
    print(f"Параметров модели: {total_params:,}")
    
    # Создаем DataLoader с аугментацией
    print(f"\n📊 Загрузка данных с Data Augmentation...")
    train_loader, val_loader = get_augmented_data_loaders(batch_size=batch_size)
    print(f"Тренировочных батчей: {len(train_loader)}")
    print(f"Валидационных батчей: {len(val_loader)}")
    
    # Настройка обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    
    print(f"\n⚙️ НАСТРОЙКИ ОБУЧЕНИЯ:")
    print(f"Функция потерь: CrossEntropyLoss")
    print(f"Оптимизатор: Adam (lr={learning_rate}, weight_decay=1e-4)")
    print(f"Scheduler: StepLR (step_size=2, gamma=0.7)")
    print(f"Эпох: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Dropout rate: {dropout_rate}")
    
    # Инициализация логгера
    logger = TrainingLogger("ImprovedCNN")
    
    # Обучение
    print(f"\n🎯 НАЧАЛО ОБУЧЕНИЯ:")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        print(f"\nЭпоха {epoch}/{epochs}:")
        
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Обновление learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Логирование
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Результаты эпохи {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Время эпохи: {epoch_time:.1f}с")
    
    total_time = time.time() - start_time
    
    # Финальные результаты
    print(f"\n📈 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"Общее время обучения: {total_time/60:.1f} минут")
    print(f"Лучшая точность на валидации: {max(logger.val_accuracies):.2f}%")
    print(f"Финальная точность на валидации: {logger.val_accuracies[-1]:.2f}%")
    print(f"Финальная loss на валидации: {logger.val_losses[-1]:.4f}")
    
    # Сохранение логов
    logger.save_logs('results/improved_training_logs.txt')
    
    # Построение графиков
    plot_training_results(logger, 'results/improved_training_results.png')
    
    # Сохранение модели
    model_path = 'models/improved_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'val_accuracy': logger.val_accuracies[-1],
        'val_loss': logger.val_losses[-1],
        'total_params': total_params,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate
    }, model_path)
    
    print(f"✅ Модель сохранена в {model_path}")
    
    return model, logger

def main():
    """Главная функция"""
    # Параметры обучения
    EPOCHS = 5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    DROPOUT_RATE = 0.5
    
    print("🎯 ЗАПУСК ОБУЧЕНИЯ УЛУЧШЕННОЙ CNN")
    print("=" * 50)
    
    # Запуск обучения
    model, logger = train_improved_model(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        dropout_rate=DROPOUT_RATE
    )
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Результаты сохранены в:")
    print(f"  - improved_training_logs.txt (логи)")
    print(f"  - improved_training_results.png (графики)")
    print(f"  - improved_model.pth (модель)")

if __name__ == "__main__":
    main()
