#!/usr/bin/env python3
"""
Запуск обучения на 25 эпохах для Baseline и Improved CNN
"""

import subprocess
import sys
import os
import time

def run_training_script(script_name, description):
    """Запуск скрипта обучения с описанием"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {description} завершен за {duration/60:.1f} минут")
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ {description} завершился с ошибкой за {duration/60:.1f} минут")
        print(f"Код ошибки: {e.returncode}")
        return False

def main():
    """Главная функция"""
    
    print("🎯 ОБУЧЕНИЕ МОДЕЛЕЙ НА 25 ЭПОХАХ")
    print("=" * 60)
    print("Это займет значительно больше времени, но покажет")
    print("реальные возможности Improved CNN!")
    print("=" * 60)
    
    # Создаем директории
    os.makedirs('results/25_epochs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Список скриптов обучения
    training_scripts = [
        ("scripts/train_baseline_25_epochs.py", "Обучение Baseline CNN (25 эпох)"),
        ("scripts/train_improved_25_epochs.py", "Обучение Improved CNN (25 эпох)")
    ]
    
    # Результаты
    results = []
    
    # Запускаем обучение
    for script, description in training_scripts:
        success = run_training_script(script, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠️  Обучение '{description}' завершилось с ошибкой")
            response = input("Продолжить с остальными моделями? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ ОБУЧЕНИЯ НА 25 ЭПОХАХ")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for description, success in results:
        status = "✅ ЗАВЕРШЕНО" if success else "❌ ОШИБКА"
        print(f"{status} - {description}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nВсего моделей: {len(results)}")
    print(f"Успешно обучено: {passed}")
    print(f"С ошибками: {failed}")
    
    if failed == 0:
        print(f"\n🎉 ВСЕ МОДЕЛИ ОБУЧЕНЫ УСПЕШНО!")
        print(f"Результаты сохранены в папке results/25_epochs/")
        print(f"Теперь можно запустить сравнение:")
        print(f"  python scripts/compare_5_vs_25_epochs.py")
    else:
        print(f"\n⚠️  {failed} модель(ей) не удалось обучить")
    
    print(f"\n💡 СОВЕТ:")
    print(f"Теперь Improved CNN должен показать лучшие результаты!")
    print(f"Сложная архитектура + регуляризация + много эпох = победа!")

if __name__ == "__main__":
    main()
