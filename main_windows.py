import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import time

from data_preprocessing import MIDIDataPreprocessor
from feature_engineering import FeatureEngineer
from models import BaseModels, ModelEvaluator
from ensemble_methods import EnsembleMethods

def main_windows():
    """Версия для Windows с исправлениями проблем памяти"""
    print("=" * 70)
    print("🪟 WINDOWS ВЕРСИЯ: Генерация музыки с исправлениями проблем памяти")
    print("=" * 70)
    
    total_start_time = time.time()
    
    print("📁 Создание структуры проекта...")
    folders = [
        'data/raw',
        'data/processed',
        'models',
        'models/base_models',
        'results/plots',
        'results/generated_music'
    ]
    
    for folder in folders:
        os.makedirs(os.path.join(current_dir, folder), exist_ok=True)
        print(f"  ✓ {folder}")
    
    maestro_path = os.path.join(current_dir, 'data', 'raw', 'maestro-v3.0.0')
    if not os.path.exists(maestro_path):
        print("❌ Датасет не найден!")
        print(f"Положите датасет MAESTRO в: {maestro_path}")
        print("\nИнструкция:")
        print("1. Скачайте датасет MAESTRO v3.0.0")
        print("2. Разархивируйте в папку data/raw/maestro-v3.0.0/")
        print("3. Перезапустите программу")
        return
    
    config = {
        'max_files': 8,          
        'seq_length': 25,         
        'test_size': 0.2,
        'random_state': 42,
        'fast_mode': True
    }
    
    print(f"\n⚙️ КОНФИГУРАЦИЯ ДЛЯ WINDOWS:")
    print(f"  • Максимум файлов: {config['max_files']}")
    print(f"  • Длина последовательности: {config['seq_length']}")
    print(f"  • Режим: {'БЫСТРЫЙ' if config['fast_mode'] else 'ПОЛНЫЙ'}")
    print("  ⚠️  Отключен параллелизм и сложные модели для избежания ошибок памяти")
    
    try:
        print("\n" + "=" * 50)
        print("📦 ЭТАП 1: ПОДГОТОВКА ДАННЫХ")
        print("=" * 50)
        
        preprocessor = MIDIDataPreprocessor(
            data_path=maestro_path,
            max_files=config['max_files'],
            seq_length=config['seq_length']
        )
        
        X, y = preprocessor.preprocess()
        
        if len(X) == 0 or len(y) == 0:
            print("❌ Не удалось загрузить данные!")
            return
        
        print("\n" + "=" * 50)
        print("🔧 ЭТАП 2: ИНЖЕНЕРИЯ ПРИЗНАКОВ")
        print("=" * 50)
        
        engineer = FeatureEngineer(project_root=current_dir, fast_mode=True)
        
        
        X_normalized = engineer.normalize_features(X)
        
        scaler = engineer.scaler
        
        X_features = engineer.extract_temporal_features(X_normalized, feature_types=['basic'])
        
        print("\n" + "=" * 50)
        print("✂️ ЭТАП 3: РАЗДЕЛЕНИЕ ДАННЫХ")
        print("=" * 50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, 
            test_size=config['test_size'], 
            random_state=config['random_state'],
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        print(f"✅ Обучающая выборка: {X_train.shape[0]:,} примеров")
        print(f"✅ Тестовая выборка: {X_test.shape[0]:,} примеров")
        print(f"✅ Количество признаков: {X_train.shape[1]}")
        
        print("\n" + "=" * 50)
        print("🤖 ЭТАП 4: ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
        print("=" * 50)
        print("⚠️  Обучение с отключенным параллелизмом для Windows")
        
        base_trainer = BaseModels(project_root=current_dir, fast_mode=True)
        base_results = base_trainer.train_models(X_train, y_train, X_test, y_test)
        
        print("\n" + "=" * 50)
        print("🎭 ЭТАП 5: ПРОСТОЙ АНСАМБЛЬ (Voting)")
        print("=" * 50)
        
        base_models = {}
        successful_models = []
        
        for name, result in base_results.items():
            if result['success']:
                base_models[name] = result['model']
                successful_models.append((name, result['accuracy']))
        
        voting_model = None
        voting_accuracy = 0.0
        voting_f1 = 0.0
        voting_time = 0.0
        
        if len(successful_models) >= 2:
            successful_models.sort(key=lambda x: x[1], reverse=True)
            top_models = successful_models[:2]
            
            print(f"✅ Используем 2 лучшие модели для Voting:")
            for name, acc in top_models:
                print(f"   • {name}: {acc:.4f}")
            
            from sklearn.ensemble import VotingClassifier
            
            estimators = [(name, base_models[name]) for name, _ in top_models]
            voting = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=1,  
                verbose=0
            )
            
            print("\n🎯 Обучение Voting Ensemble...")
            start_time = time.time()
            voting.fit(X_train, y_train)
            voting_time = time.time() - start_time
            
            y_pred_voting = voting.predict(X_test)
            from sklearn.metrics import accuracy_score, f1_score
            voting_accuracy = accuracy_score(y_test, y_pred_voting)
            voting_f1 = f1_score(y_test, y_pred_voting, average='weighted')
            
            print(f"✅ Voting Ensemble: Точность={voting_accuracy:.4f}, F1={voting_f1:.4f}")
            print(f"⏱️  Время обучения: {voting_time:.2f} секунд")
            
            model_path = os.path.join(current_dir, 'models', 'voting_ensemble_windows.pkl')
            joblib.dump(voting, model_path, compress=3)
            print(f"💾 Модель сохранена: {model_path}")
            
            voting_model = voting
            
            base_results['Voting Ensemble'] = {
                'model': voting,
                'accuracy': voting_accuracy,
                'f1_score': voting_f1,
                'precision': voting_accuracy,  
                'recall': voting_accuracy,     
                'training_time': voting_time,
                'success': True
            }
        else:
            print("⚠️  Недостаточно успешных моделей для создания ансамбля")
            if successful_models:
                best_name, best_acc = successful_models[0]
                best_model = base_models[best_name]
                model_path = os.path.join(current_dir, 'models', 'best_model_windows.pkl')
                joblib.dump(best_model, model_path, compress=3)
                print(f"💾 Лучшая модель сохранена: {model_path} ({best_acc:.4f})")
                voting_model = best_model
                voting_accuracy = best_acc
        
               
        print("\n💾 Сохранение scaler для генерации музыки...")
        
        scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')
        
        if hasattr(engineer.scaler, 'mean_'):
            joblib.dump(engineer.scaler, scaler_path, compress=3)
            print(f"✅ Scaler сохранен: {scaler_path}")
            
            print(f"   • Обучен на {len(engineer.scaler.mean_)} признаках")
            print(f"   • Средние значения: {engineer.scaler.mean_[:3]}...")
            print(f"   • Стандартные отклонения: {engineer.scaler.scale_[:3]}...")
        else:
            print("❌ Scaler не был обучен! Создаю и обучаю новый...")
            
            from sklearn.preprocessing import StandardScaler
            new_scaler = StandardScaler()
            
            if len(X_train.shape) == 3:
                X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            else:
                X_train_flat = X_train
            
            new_scaler.fit(X_train_flat)
            joblib.dump(new_scaler, scaler_path, compress=3)
            print(f"✅ Новый scaler обучен и сохранен: {scaler_path}")
        
        model_info = {
            'seq_length': config['seq_length'],
            'fast_mode': config['fast_mode'],
            'model_type': 'Voting Ensemble' if voting_model is not None else 'Best Model',
            'accuracy': float(voting_accuracy) if voting_accuracy else 0.0,
            'num_classes': len(np.unique(y)),
            'num_features': X_train.shape[1],
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'notes': 'Модель обучена в Windows режиме с ограниченными ресурсами'
        }
        
        info_path = os.path.join(current_dir, 'models', 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        print(f"✅ Информация о модели сохранена: {info_path}")
        
        print("\n" + "=" * 70)
        print("📈 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 70)
        
        from tabulate import tabulate
        
        table_data = []
        for name, result in base_results.items():
            if result.get('success', False):
                table_data.append([
                    name,
                    f"{result['accuracy']:.4f}",
                    f"{result['f1_score']:.4f}",
                    f"{result['training_time']:.2f}с"
                ])
        
        if table_data:
            table_data.sort(key=lambda x: float(x[1]), reverse=True)
            
            print(tabulate(table_data, 
                          headers=['Модель', 'Точность', 'F1-Score', 'Время обучения'],
                          tablefmt='grid'))
            
            print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {table_data[0][0]} с точностью {table_data[0][1]}")
            
            print("\n📊 Создание графика сравнения моделей...")
            models = [row[0] for row in table_data[:8]]  # Топ-8
            accuracies = [float(row[1]) for row in table_data[:8]]
            
            plt.figure(figsize=(10, 6))
            
            colors = []
            for model_name in models:
                if 'Voting' in model_name:
                    colors.append('#FF6B6B')  
                elif 'Random' in model_name or 'Decision' in model_name:
                    colors.append('#4ECDC4')  
                else:
                    colors.append('#45B7D1')  
            
            bars = plt.bar(models, accuracies, color=colors, edgecolor='black')
            
            plt.title('Сравнение точности моделей (Windows версия)', fontsize=14, fontweight='bold')
            plt.xlabel('Модели', fontsize=12)
            plt.ylabel('Точность', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            plot_path = os.path.join(current_dir, 'results', 'plots', 'windows_results.png')
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            print(f"✅ График сохранен: {plot_path}")
            plt.show()
            
            print("\n📋 Создание финального отчета...")
            report_path = os.path.join(current_dir, 'results', 'training_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛИ ГЕНЕРАЦИИ МУЗЫКИ\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("КОНФИГУРАЦИЯ:\n")
                f.write(f"• Файлов для обучения: {config['max_files']}\n")
                f.write(f"• Длина последовательности: {config['seq_length']}\n")
                f.write(f"• Тестовая выборка: {config['test_size']*100}%\n")
                f.write(f"• Режим: {'БЫСТРЫЙ' if config['fast_mode'] else 'ПОЛНЫЙ'}\n\n")
                
                f.write("ДАННЫЕ:\n")
                f.write(f"• Обучающих примеров: {X_train.shape[0]}\n")
                f.write(f"• Тестовых примеров: {X_test.shape[0]}\n")
                f.write(f"• Количество признаков: {X_train.shape[1]}\n")
                f.write(f"• Количество классов (нот): {len(np.unique(y))}\n\n")
                
                f.write("РЕЗУЛЬТАТЫ МОДЕЛЕЙ:\n")
                for row in table_data:
                    f.write(f"• {row[0]}: Точность={row[1]}, F1={row[2]}, Время={row[3]}\n")
                
                f.write(f"\nЛУЧШАЯ МОДЕЛЬ: {table_data[0][0]} с точностью {table_data[0][1]}\n\n")
                
                f.write("СОХРАНЕННЫЕ ФАЙЛЫ:\n")
                f.write(f"• Модель: models/voting_ensemble_windows.pkl\n")
                f.write(f"• Scaler: models/scaler.pkl\n")
                f.write(f"• Информация о модели: models/model_info.json\n")
                f.write(f"• График результатов: results/plots/windows_results.png\n\n")
                
                f.write("ИНСТРУКЦИЯ ДЛЯ ГЕНЕРАЦИИ МУЗЫКИ:\n")
                f.write("1. Установите зависимости: pip install -r requirements_generate.txt\n")
                f.write("2. Запустите генерацию: python generate_music.py\n")
                f.write("3. Результаты появятся в папке results/generated_music/\n")
                f.write("=" * 70 + "\n")
            
            print(f"✅ Отчет сохранен: {report_path}")
            
        else:
            print("❌ Нет результатов для отображения!")
            
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"⏱️  Общее время выполнения: {total_time:.2f} секунд ({total_time/60:.1f} минут)")
    print(f"💾 Результаты сохранены в папках models/ и results/")
    print("\n🎵 Теперь вы можете генерировать музыку!")
    print("Запустите: python generate_music.py")
    print("=" * 70)

if __name__ == "__main__":
    main_windows()