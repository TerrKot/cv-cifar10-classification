# Model Card: CIFAR-10 CNN Classification

## Model Overview

**Model Name**: CIFAR-10 CNN Classification Models  
**Version**: 1.0  
**Date**: 2024  
**Task**: Image Classification  
**Dataset**: CIFAR-10

## Model Description

This project implements two CNN architectures for CIFAR-10 image classification:

### 1. Baseline CNN

- **Architecture**: 3 Conv2D + BatchNorm + ReLU + MaxPool, 2 Linear layers
- **Parameters**: ~1.1M
- **Dropout**: 0.5
- **Optimizer**: SGD (lr=0.001, momentum=0.9)

### 2. Improved CNN

- **Architecture**: 4 Conv2D + BatchNorm + ReLU + MaxPool, 3 Linear layers
- **Parameters**: ~2.5M
- **Dropout**: 0.3
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Enhancements**: Data Augmentation, Batch Normalization

## Training Data

- **Dataset**: CIFAR-10
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 50,000 training, 10,000 test
- **Resolution**: 32x32 RGB
- **Preprocessing**: Normalization (mean=0.5, std=0.5)

## Performance

### 5 Epochs Training

| Model    | Accuracy | Parameters | Training Time |
| -------- | -------- | ---------- | ------------- |
| Baseline | ~70-75%  | 1.1M       | ~2-3 min      |
| Improved | ~65-70%  | 2.5M       | ~3-4 min      |

### 25 Epochs Training

| Model    | Accuracy | Parameters | Training Time |
| -------- | -------- | ---------- | ------------- |
| Baseline | ~80-85%  | 1.1M       | ~10-15 min    |
| Improved | ~85-90%  | 2.5M       | ~15-20 min    |

## Key Findings

- **Improved CNN is a "slow starter"** - performs worse than Baseline on short training
- **Improved CNN excels with longer training** - surpasses Baseline at 25 epochs
- **Data augmentation helps** - but requires more training time to show benefits

## Limitations

- **Dataset**: Only tested on CIFAR-10 (32x32 images)
- **Classes**: Limited to 10 object categories
- **Resolution**: Low resolution images (32x32)
- **Generalization**: Performance may vary on other datasets

## Usage

```python
# Quick start
python run_all_tests.py

# Individual training
python scripts/train_baseline_5_epochs.py
python scripts/train_improved_5_epochs.py
```

## Reproducibility

- **Seeds**: Fixed random seeds (42) for reproducibility
- **Dependencies**: Exact versions in requirements.txt
- **Config**: Centralized configuration in config.py
- **Hardware**: Tested on CPU and GPU

## Citation

```bibtex
@misc{cifar10-cnn-classification,
  title={CIFAR-10 CNN Classification: Baseline vs Improved Architectures},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/cv-cifar10-classification}
}
```

## License

MIT License - see LICENSE file for details.
