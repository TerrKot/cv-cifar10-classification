# CIFAR-10 Classification with CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/cv-cifar10-classification/blob/main/CIFAR10_Classification.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

Проект для классификации изображений CIFAR-10 с использованием CNN архитектур. Сравниваем Baseline и Improved модели на 5 и 25 эпохах обучения.

## 🎯 Что это и чем гордимся

- **Воспроизводимые результаты** - фиксированные seeds и конфиги
- **Детальное сравнение** - Baseline vs Improved CNN на разных эпохах
- **Ключевое открытие** - Improved CNN "медленный стартер", но превосходит Baseline при длительном обучении
- **Готов к запуску** - один скрипт для всех экспериментов

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

## 📊 Результаты

### Сравнительная таблица

| Модель           | Эпохи | Точность   | Параметры | Время обучения | Эффективность |
| ---------------- | ----- | ---------- | --------- | -------------- | ------------- |
| **Baseline CNN** | 5     | **72.68%** | 1.15M     | 1.6 мин        | 45.4          |
| **Improved CNN** | 5     | 67.94%     | 4.18M     | 2.5 мин        | 27.2          |
| **Baseline CNN** | 25    | 81.93%     | 1.15M     | 8.2 мин        | 10.0          |
| **Improved CNN** | 25    | **84.86%** | 4.18M     | 12.8 мин       | 6.6           |

### Ключевые выводы

- **Improved CNN - "медленный стартер"** - хуже Baseline на 5 эпохах
- **Improved CNN превосходит Baseline** на 25 эпохах (+2.93%)
- **Baseline эффективнее** для быстрого прототипирования
- **Improved CNN лучше** для финального решения

### Графики результатов

![Сравнение точности](results/comparison_5_vs_25_epochs.png)
_Сравнение моделей на 5 vs 25 эпохах_

![Эффективность обучения](results/final_analysis/learning_efficiency.png)
_Эффективность обучения (точность/время)_

## 🏗️ Архитектуры моделей

### Baseline CNN

- **Параметров**: ~1.15M
- **Архитектура**: 3 Conv2d + BatchNorm + ReLU + MaxPool, 2 Linear
- **Dropout**: 0.5
- **Оптимизатор**: SGD (lr=0.001, momentum=0.9)
- **Эпох**: 5 (быстрое тестирование) / 25 (полное обучение)

### Improved CNN

- **Параметров**: ~4.18M
- **Архитектура**: 4 Conv2d + BatchNorm + ReLU + MaxPool, 3 Linear
- **Улучшения**: Dropout, BatchNorm, Data Augmentation
- **Оптимизатор**: Adam (lr=0.001, weight_decay=1e-4)
- **Эпох**: 5 (быстрое тестирование) / 25 (полное обучение)

## 🚀 Быстрый запуск (5 минут)

### 1. Установка

```bash
git clone https://github.com/your-repo/cv-cifar10-classification
cd cv-cifar10-classification
pip install -r requirements.txt
```

### 2. Запуск всех экспериментов

```bash
python run_all_tests.py
```

### 3. Просмотр результатов

Результаты сохраняются в `results/` с графиками и отчетами.

## 🔧 Требования

- Python 3.7+
- PyTorch 1.12.1
- torchvision 0.13.1
- matplotlib 3.5.3
- numpy 1.21.6

Полный список в `requirements.txt`

## 📁 Структура проекта

```
cv-cifar10-classification/
├── models/                     # Архитектуры моделей
│   ├── simply_cnn.py          # Baseline CNN
│   └── improve_cnn.py         # Improved CNN
├── scripts/                    # Скрипты обучения и тестирования
│   ├── train_baseline_5_epochs.py
│   ├── train_improved_5_epochs.py
│   ├── train_baseline_25_epochs.py
│   ├── train_improved_25_epochs.py
│   └── compare_models.py
├── results/                    # Результаты экспериментов
│   ├── 5_epochs/              # Результаты на 5 эпохах
│   ├── 25_epochs/             # Результаты на 25 эпохах
│   └── final_analysis/        # Финальный анализ
├── config.py                  # Централизованная конфигурация
├── requirements.txt           # Зависимости
├── LICENSE                    # MIT лицензия
└── MODEL_CARD.md             # Model Card
```

## 🔬 Воспроизводимость

- **Фиксированные seeds** - все эксперименты воспроизводимы
- **Централизованные конфиги** - все параметры в `config.py`
- **Точные версии** - `requirements.txt` с фиксированными версиями
- **Документированные результаты** - полные отчеты и графики

## 📚 Дополнительно

- [Model Card](MODEL_CARD.md) - детальная информация о моделях
- [Лицензия](LICENSE) - MIT License
- [Быстрый старт](QUICK_START.md) - подробная инструкция
