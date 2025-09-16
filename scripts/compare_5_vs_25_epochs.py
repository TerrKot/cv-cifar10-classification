#!/usr/bin/env python3
"""
Сравнение результатов обучения на 5 и 25 эпохах
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_results_from_folder(folder_path, epoch_type):
    """Загрузка результатов из папки"""
    
    results = {}
    
    # Загружаем результаты необученной Baseline модели (только из 5_epochs)
    if epoch_type == "5_epochs":
        baseline_untrained_file = os.path.join(folder_path, 'baseline_results.txt')
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
    
    # Загружаем результаты Baseline
    if epoch_type == '5_epochs':
        baseline_file = os.path.join(folder_path, 'baseline_5epochs_report.txt')
    else:
        baseline_file = os.path.join(folder_path, 'baseline_25epochs_report.txt')
    
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Финальная точность:' in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    results['baseline'] = {
                        'name': f'Baseline CNN ({epoch_type.replace("_", " ").replace("epochs", "эпох")})',
                        'accuracy': accuracy,
                        'color': 'blue' if epoch_type == '5_epochs' else 'darkblue'
                    }
                    break
    
    # Загружаем результаты Improved
    if epoch_type == '5_epochs':
        improved_file = os.path.join(folder_path, 'improved_5epochs_report.txt')
    else:
        improved_file = os.path.join(folder_path, 'improved_25epochs_report.txt')
    
    if os.path.exists(improved_file):
        with open(improved_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Финальная точность:' in line:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    results['improved'] = {
                        'name': f'Improved CNN ({epoch_type.replace("_", " ").replace("epochs", "эпох")})',
                        'accuracy': accuracy,
                        'color': 'red' if epoch_type == '5_epochs' else 'darkred'
                    }
                    break
    
    return results

def create_comparison_plot(results_5, results_25):
    """Создание графика сравнения 5 vs 25 эпох"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # График 1: Сравнение всех моделей (5 эпох)
    if len(results_5) > 0:
        models_5 = list(results_5.keys())
        accuracies_5 = [results_5[model]['accuracy'] for model in models_5]
        colors_5 = [results_5[model]['color'] for model in models_5]
        names_5 = [results_5[model]['name'] for model in models_5]
        
        # Сортируем по точности
        sorted_data_5 = sorted(zip(models_5, accuracies_5, colors_5, names_5), key=lambda x: x[1])
        models_5_sorted, accuracies_5_sorted, colors_5_sorted, names_5_sorted = zip(*sorted_data_5)
        
        bars = ax1.bar(range(len(models_5_sorted)), accuracies_5_sorted, color=colors_5_sorted, alpha=0.7)
        ax1.set_ylabel('Точность (%)')
        ax1.set_title('Результаты на 5 эпохах')
        ax1.set_ylim(0, 100)
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies_5_sorted)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xticks(range(len(models_5_sorted)))
        ax1.set_xticklabels([name.replace(' CNN', '').replace(' (5 эпох)', '').replace(' (необученная)', '') 
                            for name in names_5_sorted], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
    
    # График 2: Сравнение всех моделей (25 эпох)
    if len(results_25) > 0:
        models_25 = list(results_25.keys())
        accuracies_25 = [results_25[model]['accuracy'] for model in models_25]
        colors_25 = [results_25[model]['color'] for model in models_25]
        names_25 = [results_25[model]['name'] for model in models_25]
        
        # Сортируем по точности
        sorted_data_25 = sorted(zip(models_25, accuracies_25, colors_25, names_25), key=lambda x: x[1])
        models_25_sorted, accuracies_25_sorted, colors_25_sorted, names_25_sorted = zip(*sorted_data_25)
        
        bars = ax2.bar(range(len(models_25_sorted)), accuracies_25_sorted, color=colors_25_sorted, alpha=0.7)
        ax2.set_ylabel('Точность (%)')
        ax2.set_title('Результаты на 25 эпохах')
        ax2.set_ylim(0, 100)
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies_25_sorted)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xticks(range(len(models_25_sorted)))
        ax2.set_xticklabels([name.replace(' CNN', '').replace(' (25 эпох)', '').replace(' (необученная)', '') 
                            for name in names_25_sorted], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    
    # График 3: Сравнение Baseline CNN (5 vs 25 эпох)
    if 'baseline' in results_5 and 'baseline' in results_25:
        baseline_5_acc = results_5['baseline']['accuracy']
        baseline_25_acc = results_25['baseline']['accuracy']
        improvement = baseline_25_acc - baseline_5_acc
        
        models_comp = ['Baseline CNN\n(5 эпох)', 'Baseline CNN\n(25 эпох)']
        accs_comp = [baseline_5_acc, baseline_25_acc]
        colors_comp = ['blue', 'darkblue']
        
        bars = ax3.bar(models_comp, accs_comp, color=colors_comp, alpha=0.7)
        ax3.set_ylabel('Точность (%)')
        ax3.set_title(f'Baseline CNN: 5 vs 25 эпох\n(+{improvement:.1f}% улучшение)')
        ax3.set_ylim(0, 100)
        
        for bar, acc in zip(bars, accs_comp):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
    
    # График 4: Сравнение Improved CNN (5 vs 25 эпох)
    if 'improved' in results_5 and 'improved' in results_25:
        improved_5_acc = results_5['improved']['accuracy']
        improved_25_acc = results_25['improved']['accuracy']
        improvement = improved_25_acc - improved_5_acc
        
        models_comp = ['Improved CNN\n(5 эпох)', 'Improved CNN\n(25 эпох)']
        accs_comp = [improved_5_acc, improved_25_acc]
        colors_comp = ['red', 'darkred']
        
        bars = ax4.bar(models_comp, accs_comp, color=colors_comp, alpha=0.7)
        ax4.set_ylabel('Точность (%)')
        ax4.set_title(f'Improved CNN: 5 vs 25 эпох\n(+{improvement:.1f}% улучшение)')
        ax4.set_ylim(0, 100)
        
        for bar, acc in zip(bars, accs_comp):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison_5_vs_25_epochs.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ График сравнения сохранен в results/comparison_5_vs_25_epochs.png")

def create_comparison_report(results_5, results_25):
    """Создание отчета сравнения"""
    
    filename = 'results/comparison_5_vs_25_epochs_report.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("СРАВНЕНИЕ РЕЗУЛЬТАТОВ: 5 ЭПОХ VS 25 ЭПОХ\n")
        f.write("=" * 60 + "\n\n")
        
        # Результаты на 5 эпохах
        f.write("РЕЗУЛЬТАТЫ НА 5 ЭПОХАХ:\n")
        f.write("-" * 30 + "\n")
        if len(results_5) > 0:
            sorted_results_5 = sorted(results_5.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for model_key, model_data in sorted_results_5:
                f.write(f"{model_data['name']}: {model_data['accuracy']:.2f}%\n")
        else:
            f.write("Нет данных\n")
        
        f.write(f"\nРЕЗУЛЬТАТЫ НА 25 ЭПОХАХ:\n")
        f.write("-" * 30 + "\n")
        if len(results_25) > 0:
            sorted_results_25 = sorted(results_25.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for model_key, model_data in sorted_results_25:
                f.write(f"{model_data['name']}: {model_data['accuracy']:.2f}%\n")
        else:
            f.write("Нет данных\n")
        
        # Сравнение Baseline CNN
        if 'baseline' in results_5 and 'baseline' in results_25:
            baseline_5_acc = results_5['baseline']['accuracy']
            baseline_25_acc = results_25['baseline']['accuracy']
            baseline_improvement = baseline_25_acc - baseline_5_acc
            
            f.write(f"\nСРАВНЕНИЕ BASELINE CNN:\n")
            f.write("-" * 30 + "\n")
            f.write(f"5 эпох:  {baseline_5_acc:.2f}%\n")
            f.write(f"25 эпох: {baseline_25_acc:.2f}%\n")
            f.write(f"Улучшение: +{baseline_improvement:.2f}%\n")
            f.write(f"Коэффициент улучшения: {baseline_25_acc/baseline_5_acc:.2f}x\n")
        
        # Сравнение Improved CNN
        if 'improved' in results_5 and 'improved' in results_25:
            improved_5_acc = results_5['improved']['accuracy']
            improved_25_acc = results_25['improved']['accuracy']
            improved_improvement = improved_25_acc - improved_5_acc
            
            f.write(f"\nСРАВНЕНИЕ IMPROVED CNN:\n")
            f.write("-" * 30 + "\n")
            f.write(f"5 эпох:  {improved_5_acc:.2f}%\n")
            f.write(f"25 эпох: {improved_25_acc:.2f}%\n")
            f.write(f"Улучшение: +{improved_improvement:.2f}%\n")
            f.write(f"Коэффициент улучшения: {improved_25_acc/improved_5_acc:.2f}x\n")
        
        # Сравнение на 5 эпохах
        if 'baseline' in results_5 and 'improved' in results_5:
            baseline_5_acc = results_5['baseline']['accuracy']
            improved_5_acc = results_5['improved']['accuracy']
            diff_5 = improved_5_acc - baseline_5_acc
            
            f.write(f"\nСРАВНЕНИЕ НА 5 ЭПОХАХ:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline CNN:  {baseline_5_acc:.2f}%\n")
            f.write(f"Improved CNN:  {improved_5_acc:.2f}%\n")
            f.write(f"Разница: {diff_5:+.2f}%\n")
            
            if diff_5 > 0:
                f.write(f"Победитель: Improved CNN (+{diff_5:.2f}%)\n")
            elif diff_5 < 0:
                f.write(f"Победитель: Baseline CNN ({abs(diff_5):.2f}% лучше)\n")
            else:
                f.write(f"Ничья: Одинаковая точность\n")
        
        # Сравнение на 25 эпохах
        if 'baseline' in results_25 and 'improved' in results_25:
            baseline_25_acc = results_25['baseline']['accuracy']
            improved_25_acc = results_25['improved']['accuracy']
            diff_25 = improved_25_acc - baseline_25_acc
            
            f.write(f"\nСРАВНЕНИЕ НА 25 ЭПОХАХ:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Baseline CNN:  {baseline_25_acc:.2f}%\n")
            f.write(f"Improved CNN:  {improved_25_acc:.2f}%\n")
            f.write(f"Разница: {diff_25:+.2f}%\n")
            
            if diff_25 > 0:
                f.write(f"Победитель: Improved CNN (+{diff_25:.2f}%)\n")
            elif diff_25 < 0:
                f.write(f"Победитель: Baseline CNN ({abs(diff_25):.2f}% лучше)\n")
            else:
                f.write(f"Ничья: Одинаковая точность\n")
        
        # Анализ эффекта длительного обучения
        f.write(f"\nАНАЛИЗ ЭФФЕКТА ДЛИТЕЛЬНОГО ОБУЧЕНИЯ:\n")
        f.write("-" * 30 + "\n")
        
        if 'baseline' in results_5 and 'baseline' in results_25:
            baseline_5_acc = results_5['baseline']['accuracy']
            baseline_25_acc = results_25['baseline']['accuracy']
            baseline_improvement = baseline_25_acc - baseline_5_acc
            f.write(f"Baseline CNN: +{baseline_improvement:.2f}% за 20 дополнительных эпох\n")
            f.write(f"Эффективность: {baseline_improvement/20:.2f}% за эпоху\n")
        
        if 'improved' in results_5 and 'improved' in results_25:
            improved_5_acc = results_5['improved']['accuracy']
            improved_25_acc = results_25['improved']['accuracy']
            improved_improvement = improved_25_acc - improved_5_acc
            f.write(f"Improved CNN: +{improved_improvement:.2f}% за 20 дополнительных эпох\n")
            f.write(f"Эффективность: {improved_improvement/20:.2f}% за эпоху\n")
        
        # Выводы
        f.write(f"\nВЫВОДЫ:\n")
        f.write("-" * 30 + "\n")
        
        if 'baseline' in results_5 and 'improved' in results_5 and 'baseline' in results_25 and 'improved' in results_25:
            baseline_5_acc = results_5['baseline']['accuracy']
            improved_5_acc = results_5['improved']['accuracy']
            baseline_25_acc = results_25['baseline']['accuracy']
            improved_25_acc = results_25['improved']['accuracy']
            
            diff_5 = improved_5_acc - baseline_5_acc
            diff_25 = improved_25_acc - baseline_25_acc
            
            if diff_5 < 0 and diff_25 > 0:
                f.write("✅ Improved CNN показывает лучшие результаты на длительном обучении!\n")
                f.write("✅ Короткое обучение (5 эпох) не раскрывает потенциал Improved CNN\n")
                f.write("✅ Длительное обучение (25 эпох) демонстрирует преимущества сложной архитектуры\n")
            elif diff_5 > 0 and diff_25 > 0:
                f.write("✅ Improved CNN превосходит Baseline на всех этапах обучения\n")
            elif diff_5 < 0 and diff_25 < 0:
                f.write("⚠️  Baseline CNN показывает лучшие результаты даже на длительном обучении\n")
            else:
                f.write("🤔 Смешанные результаты требуют дополнительного анализа\n")
    
    print(f"✅ Отчет сравнения сохранен в {filename}")

def main():
    """Главная функция"""
    
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ: 5 ЭПОХ VS 25 ЭПОХ")
    print("=" * 60)
    
    # Создаем директории
    os.makedirs('results', exist_ok=True)
    
    # Загружаем результаты
    results_5 = load_results_from_folder('results/5_epochs', '5_epochs')
    results_25 = load_results_from_folder('results/25_epochs', '25_epochs')
    
    if len(results_5) == 0 and len(results_25) == 0:
        print("❌ Нет результатов для сравнения")
        print("Сначала запустите:")
        print("  python run_all_tests.py  # для 5 эпох")
        print("  python scripts/run_25_epochs_training.py  # для 25 эпох")
        return
    
    # Выводим результаты
    print("\n📈 РЕЗУЛЬТАТЫ НА 5 ЭПОХАХ:")
    if len(results_5) > 0:
        sorted_results_5 = sorted(results_5.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for model_key, model_data in sorted_results_5:
            print(f"  {model_data['name']}: {model_data['accuracy']:.2f}%")
    else:
        print("  Нет данных")
    
    print("\n📈 РЕЗУЛЬТАТЫ НА 25 ЭПОХАХ:")
    if len(results_25) > 0:
        sorted_results_25 = sorted(results_25.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for model_key, model_data in sorted_results_25:
            print(f"  {model_data['name']}: {model_data['accuracy']:.2f}%")
    else:
        print("  Нет данных")
    
    # Создаем графики и отчет
    create_comparison_plot(results_5, results_25)
    create_comparison_report(results_5, results_25)
    
    print(f"\n🎉 СРАВНЕНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
