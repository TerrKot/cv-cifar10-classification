"""
Централизованная конфигурация для CIFAR-10 Classification
Фиксирует все параметры для воспроизводимости результатов
"""

import torch
import random
import numpy as np
import os

class Config:
    """Централизованная конфигурация проекта"""
    
    # ===== СЕЕДЫ ДЛЯ ВОСПРОИЗВОДИМОСТИ =====
    RANDOM_SEED = 42
    TORCH_SEED = 42
    NUMPY_SEED = 42
    CUDA_SEED = 42
    
    # ===== ПАРАМЕТРЫ ДАННЫХ =====
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    IMAGE_SIZE = 32
    NUM_CLASSES = 10
    
    # ===== ПАРАМЕТРЫ ОБУЧЕНИЯ =====
    # Baseline CNN
    BASELINE_EPOCHS_5 = 5
    BASELINE_EPOCHS_25 = 25
    BASELINE_LEARNING_RATE = 0.001
    BASELINE_MOMENTUM = 0.9
    BASELINE_WEIGHT_DECAY = 0.0
    
    # Improved CNN  
    IMPROVED_EPOCHS_5 = 5
    IMPROVED_EPOCHS_25 = 25
    IMPROVED_LEARNING_RATE = 0.001
    IMPROVED_WEIGHT_DECAY = 1e-4
    
    # ===== АРХИТЕКТУРНЫЕ ПАРАМЕТРЫ =====
    # Baseline CNN
    BASELINE_DROPOUT = 0.5
    
    # Improved CNN
    IMPROVED_DROPOUT = 0.3
    IMPROVED_BATCH_NORM = True
    IMPROVED_DATA_AUGMENTATION = True
    
    # ===== ПАРАМЕТРЫ АУГМЕНТАЦИИ =====
    AUGMENTATION_ROTATION = 15
    AUGMENTATION_HORIZONTAL_FLIP = True
    AUGMENTATION_RANDOM_CROP = True
    AUGMENTATION_PADDING = 4
    
    # ===== ПАРАМЕТРЫ ВАЛИДАЦИИ =====
    VALIDATION_SPLIT = 0.1
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 0.001
    
    # ===== ПУТИ =====
    DATA_DIR = './data'
    MODELS_DIR = './models'
    RESULTS_DIR = './results'
    
    # ===== УСТРОЙСТВО =====
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def set_seeds(cls):
        """Устанавливает все seeds для воспроизводимости"""
        # Python random
        random.seed(cls.RANDOM_SEED)
        
        # NumPy
        np.random.seed(cls.NUMPY_SEED)
        
        # PyTorch
        torch.manual_seed(cls.TORCH_SEED)
        torch.cuda.manual_seed(cls.CUDA_SEED)
        torch.cuda.manual_seed_all(cls.CUDA_SEED)
        
        # Дополнительные настройки для воспроизводимости
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"✅ Seeds установлены: Python={cls.RANDOM_SEED}, "
              f"NumPy={cls.NUMPY_SEED}, PyTorch={cls.TORCH_SEED}")
    
    @classmethod
    def create_directories(cls):
        """Создает необходимые директории"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(f"{cls.RESULTS_DIR}/5_epochs", exist_ok=True)
        os.makedirs(f"{cls.RESULTS_DIR}/25_epochs", exist_ok=True)
        os.makedirs(f"{cls.RESULTS_DIR}/confusion_matrices", exist_ok=True)
        os.makedirs(f"{cls.RESULTS_DIR}/final_analysis", exist_ok=True)
        
        print("✅ Директории созданы")
    
    @classmethod
    def get_baseline_config(cls, epochs=5):
        """Возвращает конфиг для Baseline CNN"""
        return {
            'epochs': epochs,
            'learning_rate': cls.BASELINE_LEARNING_RATE,
            'momentum': cls.BASELINE_MOMENTUM,
            'weight_decay': cls.BASELINE_WEIGHT_DECAY,
            'dropout': cls.BASELINE_DROPOUT,
            'batch_size': cls.BATCH_SIZE,
            'optimizer': 'SGD',
            'scheduler': None
        }
    
    @classmethod
    def get_improved_config(cls, epochs=5):
        """Возвращает конфиг для Improved CNN"""
        return {
            'epochs': epochs,
            'learning_rate': cls.IMPROVED_LEARNING_RATE,
            'weight_decay': cls.IMPROVED_WEIGHT_DECAY,
            'dropout': cls.IMPROVED_DROPOUT,
            'batch_size': cls.BATCH_SIZE,
            'optimizer': 'Adam',
            'scheduler': 'StepLR',
            'batch_norm': cls.IMPROVED_BATCH_NORM,
            'data_augmentation': cls.IMPROVED_DATA_AUGMENTATION
        }
    
    @classmethod
    def get_data_transforms(cls, train=True):
        """Возвращает трансформации для данных"""
        if train and cls.IMPROVED_DATA_AUGMENTATION:
            # Улучшенные трансформации с аугментацией
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            # Базовые трансформации
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        return transform
    
    @classmethod
    def print_config(cls):
        """Выводит текущую конфигурацию"""
        print("🔧 КОНФИГУРАЦИЯ ПРОЕКТА")
        print("=" * 50)
        print(f"Seeds: Python={cls.RANDOM_SEED}, NumPy={cls.NUMPY_SEED}, PyTorch={cls.TORCH_SEED}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Baseline epochs: {cls.BASELINE_EPOCHS_5}/{cls.BASELINE_EPOCHS_25}")
        print(f"Improved epochs: {cls.IMPROVED_EPOCHS_5}/{cls.IMPROVED_EPOCHS_25}")
        print(f"Data augmentation: {cls.IMPROVED_DATA_AUGMENTATION}")
        print("=" * 50)

# Импорт torchvision для трансформаций
import torchvision
