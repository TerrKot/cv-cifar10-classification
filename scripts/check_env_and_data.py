#!/usr/bin/env python3
"""
Проверка окружения и данных CIFAR-10
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import platform
import numpy as np
from torch.utils.data import DataLoader

def check_environment():
    """Проверка окружения"""
    
    print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 50)
    
    # Python версия
    print(f"Python версия: {sys.version}")
    
    # PyTorch версия
    print(f"PyTorch версия: {torch.__version__}")
    
    # CUDA доступность
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступна: {'✅ Да' if cuda_available else '❌ Нет'}")
    
    if cuda_available:
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Память: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Система
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Архитектура: {platform.machine()}")
    
    # Доступная память
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.total / 1024**3:.1f} GB (доступно: {memory.available / 1024**3:.1f} GB)")
    except ImportError:
        print("RAM: psutil не установлен")
    
    return cuda_available

def check_data():
    """Проверка данных CIFAR-10"""
    
    print(f"\n📊 ПРОВЕРКА ДАННЫХ CIFAR-10")
    print("=" * 50)
    
    # Проверяем существование папки данных
    data_dir = './data'
    if not os.path.exists(data_dir):
        print(f"❌ Папка данных не найдена: {data_dir}")
        return False
    
    print(f"✅ Папка данных найдена: {data_dir}")
    
    # Проверяем содержимое папки
    data_contents = os.listdir(data_dir)
    print(f"Содержимое папки данных:")
    for item in data_contents:
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            size = os.path.getsize(item_path) / 1024**2  # MB
            print(f"  📄 {item} ({size:.1f} MB)")
    
    # Проверяем наличие CIFAR-10
    cifar_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.exists(cifar_dir):
        print(f"❌ CIFAR-10 не найден в {cifar_dir}")
        print("Загружаем CIFAR-10...")
        return download_cifar10()
    
    print(f"✅ CIFAR-10 найден в {cifar_dir}")
    
    # Проверяем файлы CIFAR-10
    required_files = [
        'batches.meta',
        'data_batch_1',
        'data_batch_2', 
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
        'test_batch'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(cifar_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            size = os.path.getsize(file_path) / 1024**2  # MB
            print(f"  ✅ {file} ({size:.1f} MB)")
    
    if missing_files:
        print(f"❌ Отсутствуют файлы: {missing_files}")
        return False
    
    return True

def download_cifar10():
    """Загрузка CIFAR-10"""
    
    print("📥 ЗАГРУЗКА CIFAR-10")
    print("=" * 50)
    
    try:
        # Создаем директорию
        os.makedirs('./data', exist_ok=True)
        
        # Загружаем CIFAR-10
        print("Загружаем CIFAR-10...")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True
        )
        
        print("✅ CIFAR-10 успешно загружен")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки CIFAR-10: {e}")
        return False

def test_data_loading():
    """Тестирование загрузки данных"""
    
    print(f"\n🧪 ТЕСТИРОВАНИЕ ЗАГРУЗКИ ДАННЫХ")
    print("=" * 50)
    
    try:
        # Создаем DataLoader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        
        print(f"✅ Тренировочных образцов: {len(train_dataset):,}")
        print(f"✅ Тестовых образцов: {len(test_dataset):,}")
        print(f"✅ Тренировочных батчей: {len(train_loader)}")
        print(f"✅ Тестовых батчей: {len(test_loader)}")
        
        # Тестируем загрузку батча
        print("\nТестируем загрузку батча...")
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        print(f"✅ Размер батча: {images.shape}")
        print(f"✅ Тип данных: {images.dtype}")
        print(f"✅ Диапазон значений: [{images.min():.3f}, {images.max():.3f}]")
        print(f"✅ Количество классов: {len(torch.unique(labels))}")
        
        # Классы CIFAR-10
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck')
        print(f"✅ Классы: {classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования данных: {e}")
        return False

def check_performance():
    """Проверка производительности"""
    
    print(f"\n⚡ ПРОВЕРКА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 50)
    
    try:
        import time
        
        # Тест CPU
        print("Тестируем CPU...")
        start_time = time.time()
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.mm(x, y)
        cpu_time = time.time() - start_time
        print(f"✅ CPU: {cpu_time:.3f} сек (матрица 1000x1000)")
        
        # Тест GPU (если доступна)
        if torch.cuda.is_available():
            print("Тестируем GPU...")
            start_time = time.time()
            x_gpu = torch.randn(1000, 1000).cuda()
            y_gpu = torch.randn(1000, 1000).cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            torch.cuda.synchronize()  # Ждем завершения GPU операций
            gpu_time = time.time() - start_time
            print(f"✅ GPU: {gpu_time:.3f} сек (матрица 1000x1000)")
            print(f"✅ Ускорение GPU: {cpu_time/gpu_time:.1f}x")
        else:
            print("❌ GPU недоступна для тестирования")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования производительности: {e}")
        return False

def main():
    """Главная функция"""
    
    print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ И ДАННЫХ CIFAR-10")
    print("=" * 60)
    
    # Проверяем окружение
    cuda_available = check_environment()
    
    # Проверяем данные
    data_ok = check_data()
    
    if not data_ok:
        print("\n❌ ПРОБЛЕМЫ С ДАННЫМИ")
        return False
    
    # Тестируем загрузку данных
    loading_ok = test_data_loading()
    
    if not loading_ok:
        print("\n❌ ПРОБЛЕМЫ С ЗАГРУЗКОЙ ДАННЫХ")
        return False
    
    # Проверяем производительность
    perf_ok = check_performance()
    
    # Итоговый отчет
    print(f"\n📋 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    
    print(f"Окружение: {'✅ Готово' if cuda_available else '⚠️  CPU только'}")
    print(f"Данные: {'✅ Готовы' if data_ok else '❌ Проблемы'}")
    print(f"Загрузка: {'✅ Работает' if loading_ok else '❌ Проблемы'}")
    print(f"Производительность: {'✅ Тест пройден' if perf_ok else '⚠️  Проблемы'}")
    
    if cuda_available and data_ok and loading_ok:
        print(f"\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! Система готова к работе.")
        return True
    else:
        print(f"\n⚠️  ЕСТЬ ПРОБЛЕМЫ! Проверьте ошибки выше.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
