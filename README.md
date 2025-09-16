# CIFAR-10 Classification with CNN

Проект для классификации изображений CIFAR-10 с использованием CNN архитектур.

## 📁 Структура проекта

```
cv-cifar10-classification/
├── models/                     # Модели и архитектуры
│   ├── simply_cnn.py          # Baseline CNN
│   ├── improve_cnn.py         # Улучшенная CNN
│   └── *.pth                  # Обученные модели
├── scripts/                    # Скрипты для работы
│   ├── train_baseline_5_epochs.py    # Обучение Baseline (5 эпох)
│   ├── train_improved_5_epochs.py    # Обучение Improved (5 эпох)
│   └── compare_models.py      # Сравнение моделей
├── results/                    # Результаты и логи
│   ├── *.png                  # Графики результатов
│   ├── *.txt                  # Отчеты о результатах
│   └── *.pth                  # Сохраненные модели
├── data/                      # Датасет CIFAR-10
│   └── cifar-10-batches-py/   # Файлы датасета
├── run_all_tests.py           # Запуск всех тестов
└── README.md                  # Этот файл
```

## 🚀 Быстрый старт

### Запуск всех тестов

```bash
# Запуск всех тестов одной командой
python run_all_tests.py
```

### Или по отдельности

```bash
# Обучение Baseline CNN (5 эпох)
python scripts/train_baseline_5_epochs.py

# Обучение Improved CNN (5 эпох)
python scripts/train_improved_5_epochs.py

# Сравнение моделей
python scripts/compare_models.py
```

## 📊 Модели

### Baseline CNN

- **Параметров**: ~1.1M
- **Архитектура**: 3 Conv2d + BatchNorm + ReLU + MaxPool, 2 Linear
- **Dropout**: 0.5
- **Оптимизатор**: SGD (lr=0.001, momentum=0.9)
- **Эпох**: 15

### Improved CNN

- **Параметров**: ~2.5M
- **Архитектура**: 4 Conv2d + BatchNorm + ReLU + MaxPool, 3 Linear
- **Улучшения**: Dropout, BatchNorm, Data Augmentation
- **Оптимизатор**: Adam (lr=0.001, weight_decay=1e-4)
- **Эпох**: 5

## 🔧 Требования

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy

## 📝 Описание скриптов

- `run_all_tests.py` - Запуск всех тестов одной командой
- `train_baseline_5_epochs.py` - Обучение Baseline CNN (5 эпох)
- `train_improved_5_epochs.py` - Обучение Improved CNN (5 эпох)
- `compare_models.py` - Сравнение моделей
