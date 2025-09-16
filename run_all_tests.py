#!/usr/bin/env python3
"""
Простой скрипт для запуска всех тестов CIFAR-10
"""

import subprocess
import sys
import os
import time

def run_test(script_name, description):
    """Запуск теста с описанием"""
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
    
    print("🎯 ЗАПУСК ВСЕХ ТЕСТОВ CIFAR-10")
    print("=" * 60)
    
    # Создаем директории
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Список тестов
    tests = [
        ("scripts/check_env_and_data.py", "Проверка окружения и данных"),
        ("scripts/baseline_results.py", "Baseline тестирование необученной модели"),
        ("scripts/train_baseline_5_epochs.py", "Обучение Baseline CNN (5 эпох)"),
        ("scripts/train_improved_5_epochs.py", "Обучение Improved CNN (5 эпох)")
    ]
    
    # Результаты
    results = []
    
    # Запускаем тесты
    for script, description in tests:
        success = run_test(script, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠️  Тест '{description}' завершился с ошибкой")
            response = input("Продолжить с остальными тестами? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for description, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"{status} - {description}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nВсего тестов: {len(results)}")
    print(f"Пройдено: {passed}")
    print(f"Провалено: {failed}")
    
    if failed == 0:
        print(f"\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print(f"\n⚠️  {failed} тест(ов) провалено")

if __name__ == "__main__":
    main()
