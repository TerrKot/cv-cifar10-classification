#!/usr/bin/env python3
"""
Обучение Baseline CNN на 25 эпохах
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

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.simply_cnn import SimpleCNN, count_parameters

def train_baseline_25_epochs():
    """Обучение Baseline CNN на 25 эпохах"""
    
    print("🚀 ОБУЧЕНИЕ BASELINE CNN НА 25 ЭПОХАХ")
    print("=" * 60)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Создаем модель
    model = SimpleCNN(num_classes=10).to(device)
    total_params = count_parameters(model)
    print(f"Параметров модели: {total_params:,}")
    
    # Создаем DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"Тренировочных батчей: {len(train_loader)}")
    print(f"Валидационных батчей: {len(test_loader)}")
    
    # Настройка обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Увеличиваем LR для длительного обучения
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)  # Снижаем LR каждые 8 эпох
    
    # Логирование
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Обучение
    start_time = time.time()
    
    for epoch in range(1, 26):  # 25 эпох
        print(f"\nЭпоха {epoch}/25:")
        
        # Обучение
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Валидация
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss = running_loss / len(test_loader)
        val_acc = 100. * correct / total
        
        # Обновление learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Логирование
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR:    {current_lr:.6f}")
        
        # Сохраняем лучшую модель
        if epoch == 1 or val_acc > max(val_accuracies[:-1]):
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            best_acc = val_acc
    
    total_time = time.time() - start_time
    
    # Результаты
    print(f"\n📈 РЕЗУЛЬТАТЫ BASELINE CNN (25 ЭПОХ):")
    print(f"Время обучения: {total_time/60:.1f} минут")
    print(f"Лучшая точность: {max(val_accuracies):.2f}% (эпоха {best_epoch})")
    print(f"Финальная точность: {val_accuracies[-1]:.2f}%")
    print(f"Финальная loss: {val_losses[-1]:.4f}")
    
    # Создаем директории
    os.makedirs('results/25_epochs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Сохранение модели
    model_path = 'models/baseline_25epochs_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 25,
        'best_epoch': best_epoch,
        'val_accuracy': val_accuracies[-1],
        'best_val_accuracy': max(val_accuracies),
        'val_loss': val_losses[-1],
        'total_params': total_params
    }, model_path)
    
    print(f"✅ Модель сохранена в {model_path}")
    
    # Построение графиков
    print(f"\n📊 Создание графиков...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График Loss
    epochs = list(range(1, 26))
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title('Baseline CNN (25 epochs): Loss по эпохам')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График Accuracy
    ax2.plot(epochs, train_accuracies, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Baseline CNN (25 epochs): Accuracy по эпохам')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/25_epochs/baseline_25epochs_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Графики сохранены в results/25_epochs/baseline_25epochs_results.png")
    
    # Сохранение отчета
    print(f"\n📝 Создание отчета...")
    
    filename = 'results/25_epochs/baseline_25epochs_report.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ОБУЧЕНИЯ BASELINE CNN НА 25 ЭПОХАХ\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Модель: Baseline CNN (25 epochs)\n")
        f.write(f"Параметров: {total_params:,}\n")
        f.write(f"Время обучения: {total_time/60:.1f} минут\n")
        f.write(f"Лучшая точность: {max(val_accuracies):.2f}% (эпоха {best_epoch})\n")
        f.write(f"Финальная точность: {val_accuracies[-1]:.2f}%\n")
        f.write(f"Финальная loss: {val_losses[-1]:.4f}\n\n")
        
        f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
        f.write("Эпоха | Train Loss | Train Acc | Val Loss | Val Acc\n")
        f.write("-" * 50 + "\n")
        
        for i, epoch in enumerate(epochs):
            f.write(f"{epoch:5d} | {train_losses[i]:9.4f} | "
                   f"{train_accuracies[i]:8.2f}% | "
                   f"{val_losses[i]:8.4f} | "
                   f"{val_accuracies[i]:7.2f}%\n")
    
    print(f"✅ Отчет сохранен в {filename}")
    
    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Созданы файлы:")
    print(f"  - {model_path} (модель)")
    print(f"  - results/25_epochs/baseline_25epochs_results.png (графики)")
    print(f"  - {filename} (отчет)")

if __name__ == "__main__":
    train_baseline_25_epochs()
