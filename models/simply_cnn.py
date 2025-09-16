#!/usr/bin/env python3
"""
Простая CNN для классификации CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

class SimpleCNN(nn.Module):
    """Простая CNN для CIFAR-10"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Активация
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Первый блок: Conv -> BN -> ReLU -> Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Второй блок: Conv -> BN -> ReLU -> Pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Третий блок: Conv -> BN -> ReLU -> Pool
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Разворачиваем в вектор
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    """Подсчет общего количества параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_data_loaders(batch_size=64, shuffle=True):
    """Создание DataLoader для CIFAR-10"""
    
    # Трансформации
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Загрузка датасетов
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Создание DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader

def test_model_on_batch(model, data_loader, device):
    """Тестирование модели на одном батче"""
    model.eval()
    
    # Получаем один батч
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Перемещаем на устройство
    images = images.to(device)
    labels = labels.to(device)
    
    # Получаем предсказания
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    return images, labels, outputs, predicted

def main():
    """Основная функция"""
    print("🚀 Создание простой CNN для CIFAR-10")
    print("=" * 50)
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Создаем модель
    model = SimpleCNN(num_classes=10).to(device)
    
    # Подсчитываем параметры
    total_params = count_parameters(model)
    print(f"Общее количество параметров: {total_params:,}")
    
    # Создаем DataLoader
    print(f"\n📊 Создание DataLoader (batch_size=64, shuffle=True)...")
    train_loader, test_loader = get_data_loaders(batch_size=64, shuffle=True)
    
    print(f"Размер тренировочного набора: {len(train_loader.dataset)}")
    print(f"Количество батчей в тренировочном наборе: {len(train_loader)}")
    print(f"Размер тестового набора: {len(test_loader.dataset)}")
    print(f"Количество батчей в тестовом наборе: {len(test_loader)}")
    
    # Тестируем модель на одном батче
    print(f"\n🧪 Тестирование модели на одном батче...")
    images, labels, outputs, predicted = test_model_on_batch(model, train_loader, device)
    
    # Выводим информацию о батче и выходе
    print(f"Batch shape: {images.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Predicted shape: {predicted.shape}")
    
    # Проверяем точность на этом батче
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    
    print(f"\n📈 Результаты на одном батче:")
    print(f"Правильных предсказаний: {correct}/{total}")
    print(f"Точность: {accuracy:.2f}%")
    
    # Показываем несколько примеров предсказаний
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"\n🔍 Примеры предсказаний (первые 5):")
    for i in range(min(5, len(predicted))):
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]
        correct_symbol = "✅" if predicted[i] == labels[i] else "❌"
        print(f"  {correct_symbol} Истинный: {true_label}, Предсказанный: {pred_label}")
    
    print(f"\n✅ Тест завершен успешно!")
    print(f"Модель готова к обучению на {total_params:,} параметрах")

if __name__ == "__main__":
    main()
