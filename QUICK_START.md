# 🚀 Быстрый старт CIFAR-10 Classification

Этот проект содержит реализацию CNN для классификации изображений CIFAR-10 с базовой и улучшенной архитектурами.

## 📋 Предварительные требования

- Python 3.7+
- PyTorch 1.8+
- CUDA (опционально, для GPU)
- 4+ GB RAM

## 🛠️ Установка

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd cv-cifar10-classification

# Установите зависимости
pip install torch torchvision matplotlib numpy psutil
```

## 🎯 Быстрый запуск

### 1. Проверка окружения

```bash
python scripts/check_env_and_data.py
```

### 2. Baseline тестирование

```bash
python scripts/baseline_results.py
```

### 3. Обучение базовой модели (5 эпох)

```bash
python scripts/train_baseline_5_epochs.py
```

### 4. Обучение улучшенной модели (5 эпох)

```bash
python scripts/train_improved_5_epochs.py
```

### 5. Обучение на 25 эпохах (для полного сравнения)

```bash
python scripts/run_25_epochs_training.py
```

### 6. Сравнение результатов 5 vs 25 эпох

```bash
python scripts/compare_5_vs_25_epochs.py
```

### 7. Запуск всех тестов (5 эпох)

```bash
python run_all_tests.py
```

## 📊 Результаты

Результаты сохраняются в структурированных папках:

### `results/5_epochs/` - Результаты на 5 эпохах

- `baseline_results.txt` - результаты необученной модели
- `baseline_5epochs_report.txt` - отчет обучения базовой модели
- `improved_5epochs_report.txt` - отчет обучения улучшенной модели
- `*.png` - графики результатов

### `results/25_epochs/` - Результаты на 25 эпохах

- `baseline_25epochs_report.txt` - отчет обучения базовой модели
- `improved_25epochs_report.txt` - отчет обучения улучшенной модели
- `*.png` - графики результатов

### `results/` - Сравнительные результаты

- `comparison_5_vs_25_epochs_report.txt` - сравнение 5 vs 25 эпох
- `comparison_5_vs_25_epochs.png` - график сравнения

## 🏗️ Архитектура проекта

```
cv-cifar10-classification/
├── data/                    # Данные CIFAR-10
├── models/                  # Архитектуры моделей
│   ├── simply_cnn.py       # Базовая CNN
│   └── improve_cnn.py      # Улучшенная CNN
├── scripts/                 # Скрипты для обучения и тестирования
│   ├── check_env_and_data.py
│   ├── baseline_results.py
│   ├── train_baseline_5_epochs.py
│   ├── train_improved_5_epochs.py
│   ├── train_baseline_25_epochs.py
│   ├── train_improved_25_epochs.py
│   ├── run_25_epochs_training.py
│   ├── compare_models.py
│   ├── compare_results.py
│   └── compare_5_vs_25_epochs.py
├── results/                 # Результаты экспериментов
│   ├── 5_epochs/           # Результаты на 5 эпохах
│   └── 25_epochs/          # Результаты на 25 эпохах
└── run_all_tests.py        # Запуск всех тестов
```

## 🔧 Настройка

### GPU поддержка

Для использования GPU убедитесь, что установлена CUDA:

```bash
# Проверка CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Параметры обучения

Основные параметры можно изменить в файлах скриптов:

- `batch_size` - размер батча (по умолчанию 64)
- `learning_rate` - скорость обучения (по умолчанию 0.001)
- `epochs` - количество эпох (по умолчанию 5)

## 📈 Ожидаемые результаты

### На 5 эпохах:

- **Необученная модель**: ~10% точность (случайный выбор)
- **Базовая CNN**: ~70-75% точность
- **Улучшенная CNN**: ~65-70% точность (хуже из-за перерегуляризации)

### На 25 эпохах:

- **Базовая CNN**: ~80-85% точность
- **Улучшенная CNN**: ~85-90% точность (лучше благодаря длительному обучению)

### Ключевой вывод:

**Improved CNN - "медленный стартер"**, который превосходит Baseline только при длительном обучении!

## 🐛 Устранение неполадок

### Проблемы с данными

```bash
# Перезагрузите CIFAR-10
rm -rf data/cifar-10-batches-py
python scripts/check_env_and_data.py
```

### Проблемы с памятью

Уменьшите `batch_size` в скриптах обучения.

### Проблемы с CUDA

Убедитесь, что версия PyTorch совместима с вашей версией CUDA.

## 📚 Дополнительная информация

- [Документация PyTorch](https://pytorch.org/docs/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Проектная структура](PROJECT_STRUCTURE.md)
