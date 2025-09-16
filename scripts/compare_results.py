#!/usr/bin/env python3
"""
Сравнение результатов всех моделей без переобучения
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_results():
    """Загрузка результатов из файлов"""
    
    results = {}
    
    # Загружаем результаты необученной Baseline модели
    baseline_untrained_file = 'results/baseline_results.txt'
    if os.path.exists(baseline_untrained_file):
        with open(baseline_untrained_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Общая точность:' in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    results['baseline_untrained'] = {
                        'name': 'Baseline CNN (необученная)',
                        'accuracy': accuracy,
                        'color': 'lightblue'
                    }
                    break
    
    # Загружаем результаты Baseline (5 эпох)
    baseline_file = 'results/baseline_5epochs_report.txt'
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Финальная точность:' in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    results['baseline'] = {
                        'name': 'Baseline CNN (5 epochs)',
                        'accuracy': accuracy,
                        'color': 'blue'
                    }
                    break
    
    # Загружаем результаты Improved
    improved_file = 'results/improved_5epochs_report.txt'
    if os.path.exists(improved_file):
        with open(improved_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Финальная точность:' in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    results['improved'] = {
                        'name': 'Improved CNN (5 epochs)',
                        'accuracy': accuracy,
                        'color': 'red'
                    }
                    break
    
    # Загружаем результаты полного обучения Baseline (15 эпох)
    baseline_full_file = 'results/test_results.txt'
    if os.path.exists(baseline_full_file):
        with open(baseline_full_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Общая точность:' in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    results['baseline_full'] = {
                        'name': 'Baseline CNN (15 epochs)',
                        'accuracy': accuracy,
                        'color': 'darkblue'
                    }
                    break
    
    return results

def create_comparison_plot(results):
    """Создание графика сравнения"""
    
    if len(results) < 2:
        print("❌ Недостаточно результатов для сравнения")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # График 1: Сравнение точности всех моделей
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    colors = [results[model]['color'] for model in models]
    names = [results[model]['name'] for model in models]
    
    # Сортируем по точности для лучшего отображения
    sorted_data = sorted(zip(models, accuracies, colors, names), key=lambda x: x[1])
    models_sorted, accuracies_sorted, colors_sorted, names_sorted = zip(*sorted_data)
    
    bars = ax1.bar(range(len(models_sorted)), accuracies_sorted, color=colors_sorted, alpha=0.7)
    ax1.set_ylabel('Точность (%)')
    ax1.set_title('Сравнение точности всех моделей')
    ax1.set_ylim(0, 100)
    
    # Добавляем значения на столбцы
    for i, (bar, acc) in enumerate(zip(bars, accuracies_sorted)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Настройка подписей осей
    ax1.set_xticks(range(len(models_sorted)))
    ax1.set_xticklabels([name.replace(' CNN', '').replace(' (5 epochs)', '').replace(' (15 epochs)', '').replace(' (необученная)', '') 
                        for name in names_sorted], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Сравнение обученных моделей
    if 'baseline' in results and 'improved' in results:
        baseline_acc = results['baseline']['accuracy']
        improved_acc = results['improved']['accuracy']
        
        # Создаем столбцы для сравнения
        models_comp = ['Baseline CNN', 'Improved CNN']
        accs_comp = [baseline_acc, improved_acc]
        colors_comp = ['blue', 'red']
        
        bars = ax2.bar(models_comp, accs_comp, color=colors_comp, alpha=0.7)
        ax2.set_ylabel('Точность (%)')
        ax2.set_title('Сравнение обученных моделей\n(5 эпох)')
        ax2.set_ylim(0, 100)
        
        # Добавляем значения на столбцы
        for bar, acc in zip(bars, accs_comp):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Добавляем линию разницы
        diff = improved_acc - baseline_acc
        ax2.text(0.5, max(accs_comp) + 5, f'Разница: {diff:+.1f}%', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Недостаточно данных\nдля сравнения', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Сравнение обученных моделей')
    
    plt.tight_layout()
    plt.savefig('results/comparison_all_models.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ График сравнения сохранен в results/comparison_all_models.png")

def create_comparison_report(results):
    """Создание отчета сравнения"""
    
    if len(results) < 2:
        print("❌ Недостаточно результатов для сравнения")
        return
    
    filename = 'results/comparison_all_models_report.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ СРАВНЕНИЯ ВСЕХ МОДЕЛЕЙ CIFAR-10\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("РЕЗУЛЬТАТЫ:\n")
        f.write("-" * 30 + "\n")
        # Сортируем по точности
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for model_key, model_data in sorted_results:
            f.write(f"{model_data['name']}: {model_data['accuracy']:.2f}%\n")
        
        # Сравнение обученных моделей (5 эпох)
        if 'baseline' in results and 'improved' in results:
            baseline_acc = results['baseline']['accuracy']
            improved_acc = results['improved']['accuracy']
            diff = improved_acc - baseline_acc
            
            f.write(f"\nСРАВНЕНИЕ ОБУЧЕННЫХ МОДЕЛЕЙ (5 ЭПОХ):\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline CNN (5 epochs): {baseline_acc:.2f}%\n")
            f.write(f"Improved CNN (5 epochs): {improved_acc:.2f}%\n")
            f.write(f"Разница: {diff:+.2f}%\n")
            
            if diff > 0:
                f.write(f"Победитель: Improved CNN (+{diff:.2f}%)\n")
            elif diff < 0:
                f.write(f"Победитель: Baseline CNN ({abs(diff):.2f}% лучше)\n")
            else:
                f.write(f"Ничья: Одинаковая точность\n")
        
        # Анализ прогресса обучения Baseline
        if 'baseline_untrained' in results and 'baseline' in results:
            untrained_acc = results['baseline_untrained']['accuracy']
            trained_acc = results['baseline']['accuracy']
            improvement = trained_acc - untrained_acc
            
            f.write(f"\nАНАЛИЗ ПРОГРЕССА ОБУЧЕНИЯ BASELINE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline CNN (необученная): {untrained_acc:.2f}%\n")
            f.write(f"Baseline CNN (5 epochs): {trained_acc:.2f}%\n")
            f.write(f"Улучшение: +{improvement:.2f}%\n")
            f.write(f"Коэффициент улучшения: {trained_acc/untrained_acc:.1f}x\n")
        
        # Анализ эффекта длительного обучения
        if 'baseline' in results and 'baseline_full' in results:
            short_acc = results['baseline']['accuracy']
            full_acc = results['baseline_full']['accuracy']
            additional_improvement = full_acc - short_acc
            
            f.write(f"\nАНАЛИЗ ЭФФЕКТА ДЛИТЕЛЬНОГО ОБУЧЕНИЯ:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline CNN (5 epochs): {short_acc:.2f}%\n")
            f.write(f"Baseline CNN (15 epochs): {full_acc:.2f}%\n")
            f.write(f"Дополнительное улучшение: +{additional_improvement:.2f}%\n")
            f.write(f"Эффективность дополнительных эпох: {additional_improvement/10:.2f}% за эпоху\n")
        
        f.write(f"\nВЫВОД:\n")
        f.write("-" * 30 + "\n")
        
        # Анализ лучшей модели
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        f.write(f"Лучшая модель: {best_model[1]['name']} ({best_model[1]['accuracy']:.2f}%)\n")
        
        if 'baseline' in results and 'improved' in results:
            baseline_acc = results['baseline']['accuracy']
            improved_acc = results['improved']['accuracy']
            diff = improved_acc - baseline_acc
            
            if abs(diff) < 1:
                f.write("Обученные модели показывают практически одинаковые результаты.\n")
            elif diff > 0:
                f.write("Improved CNN показывает лучшие результаты среди обученных моделей.\n")
            else:
                f.write("Baseline CNN показывает лучшие результаты среди обученных моделей.\n")
        
        if 'baseline_untrained' in results and 'baseline' in results:
            f.write("Обучение значительно улучшает производительность модели.\n")
        
        if 'baseline' in results and 'baseline_full' in results:
            short_acc = results['baseline']['accuracy']
            full_acc = results['baseline_full']['accuracy']
            if full_acc > short_acc + 5:
                f.write("Длительное обучение дает существенное улучшение.\n")
            else:
                f.write("Длительное обучение дает незначительное улучшение.\n")
    
    print(f"✅ Отчет сравнения сохранен в {filename}")

def main():
    """Главная функция"""
    
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ ВСЕХ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Создаем директории
    os.makedirs('results', exist_ok=True)
    
    # Загружаем результаты
    results = load_results()
    
    if len(results) < 2:
        print("❌ Недостаточно результатов для сравнения")
        print("Сначала запустите:")
        print("  python scripts/baseline_results.py")
        print("  python scripts/train_baseline_5_epochs.py")
        print("  python scripts/train_improved_5_epochs.py")
        return
    
    # Выводим результаты
    print("\n📈 РЕЗУЛЬТАТЫ:")
    # Сортируем по точности
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for model_key, model_data in sorted_results:
        print(f"  {model_data['name']}: {model_data['accuracy']:.2f}%")
    
    # Сравнение обученных моделей (5 эпох)
    if 'baseline' in results and 'improved' in results:
        baseline_acc = results['baseline']['accuracy']
        improved_acc = results['improved']['accuracy']
        diff = improved_acc - baseline_acc
        
        print(f"\n🔍 СРАВНЕНИЕ ОБУЧЕННЫХ МОДЕЛЕЙ (5 ЭПОХ):")
        print(f"  Baseline CNN (5 epochs): {baseline_acc:.2f}%")
        print(f"  Improved CNN (5 epochs): {improved_acc:.2f}%")
        print(f"  Разница: {diff:+.2f}%")
        
        if diff > 0:
            print(f"  Победитель: Improved CNN (+{diff:.2f}%)")
        elif diff < 0:
            print(f"  Победитель: Baseline CNN ({abs(diff):.2f}% лучше)")
        else:
            print(f"  Ничья: Одинаковая точность")
    
    # Анализ прогресса обучения
    if 'baseline_untrained' in results and 'baseline' in results:
        untrained_acc = results['baseline_untrained']['accuracy']
        trained_acc = results['baseline']['accuracy']
        improvement = trained_acc - untrained_acc
        
        print(f"\n📈 АНАЛИЗ ПРОГРЕССА ОБУЧЕНИЯ BASELINE:")
        print(f"  Необученная модель: {untrained_acc:.2f}%")
        print(f"  Обученная модель (5 epochs): {trained_acc:.2f}%")
        print(f"  Улучшение: +{improvement:.2f}%")
        print(f"  Коэффициент улучшения: {trained_acc/untrained_acc:.1f}x")
    
    # Анализ эффекта длительного обучения
    if 'baseline' in results and 'baseline_full' in results:
        short_acc = results['baseline']['accuracy']
        full_acc = results['baseline_full']['accuracy']
        additional_improvement = full_acc - short_acc
        
        print(f"\n📈 АНАЛИЗ ЭФФЕКТА ДЛИТЕЛЬНОГО ОБУЧЕНИЯ:")
        print(f"  Baseline CNN (5 epochs): {short_acc:.2f}%")
        print(f"  Baseline CNN (15 epochs): {full_acc:.2f}%")
        print(f"  Дополнительное улучшение: +{additional_improvement:.2f}%")
        print(f"  Эффективность дополнительных эпох: {additional_improvement/10:.2f}% за эпоху")
    
    # Создаем графики и отчет
    create_comparison_plot(results)
    create_comparison_report(results)
    
    print(f"\n🎉 СРАВНЕНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
