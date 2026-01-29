import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class BaseModels:
    """Класс для обучения базовых моделей с отключенным параллелизмом для Windows"""
    def __init__(self, project_root='.', fast_mode=True):
        """
        Инициализация моделей
        
        Args:
            project_root: корневая директория проекта
            fast_mode: режим быстрого обучения (меньше данных, проще модели)
        """
        self.project_root = project_root
        self.fast_mode = fast_mode
        
        if fast_mode:
            self.models = self._get_fast_models()
        else:
            self.models = self._get_full_models()
        
    def _get_fast_models(self):
        """Быстрые модели для обучения на ноутбуке с Windows"""
        return {
            'Random Forest': RandomForestClassifier(
                n_estimators=30,  
                max_depth=8,
                min_samples_split=20,
                random_state=42,
                n_jobs=1,  
                verbose=0,
                max_samples=0.7  
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=20,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=500,
                random_state=42,
                n_jobs=1,  
                solver='saga',
                tol=1e-2,
                verbose=0
            ),
            'K-Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=1,  
                algorithm='ball_tree' 
            ),
            'Naive Bayes': GaussianNB(),
            'Linear Discriminant': LinearDiscriminantAnalysis(),
            'Perceptron': Perceptron(
                max_iter=100,
                random_state=42,
                n_jobs=1,  
                tol=1e-2,
                verbose=0
            )
        }
    
    def _get_full_models(self):
        """Полный набор моделей (для мощного сервера)"""
        return {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=1,  
                verbose=0
            ),
            'Linear SVM': LinearSVC(
                random_state=42,
                max_iter=1000,
                tol=1e-2,
                dual=False,
                verbose=0
            ),
            'SGD Classifier': SGDClassifier(
                loss='hinge',
                penalty='l2',
                max_iter=500,
                tol=1e-2,
                random_state=42,
                n_jobs=1, 
                learning_rate='optimal',
                verbose=0
            ),
            'K-Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=1  
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=12
            ),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                random_state=42,
                verbose=0,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,
                random_state=42,
                verbose=0
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=30,
                random_state=42
            )
        }
    
    def ensure_directory_exists(self, path):
        """Проверка и создание директории если нужно"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def train_single_model(self, name, model, X_train, y_train, X_val, y_val):
        """Обучение одной модели с обработкой исключений"""
        try:
            start_time = time.time()
            
            if self.fast_mode and X_train.shape[0] > 3000:
                np.random.seed(42)
                indices = np.random.choice(len(X_train), min(3000, len(X_train)), replace=False)
                X_train_sub = X_train[indices]
                y_train_sub = y_train[indices]
                print(f"  ⚡ {name}: используем подвыборку {len(X_train_sub)} из {len(X_train)} примеров")
                model.fit(X_train_sub, y_train_sub)
            else:
                model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            
            return {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'training_time': training_time,
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️  Ошибка при обучении {name}: {e}")
            return {
                'model': None,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'training_time': 0.0,
                'success': False
            }
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Обучение всех моделей с красивыми прогресс-барами"""
        results = {}
        
        print("\n" + "=" * 60)
        print("🤖 ОБУЧЕНИЕ БАЗОВЫХ МОДЕЛЕЙ")
        print("=" * 60)
        
        if self.fast_mode:
            print("⚡ РЕЖИМ: БЫСТРЫЙ (оптимизирован для Windows и ноутбука)")
        else:
            print("🐢 РЕЖИМ: ПОЛНЫЙ")
        
        print(f"📊 Размер обучающих данных: {X_train.shape}")
        print(f"🎯 Количество классов: {len(np.unique(y_train))}")
        print("💡 ПОДСКАЗКА: Для Windows отключен параллелизм для избежания ошибок памяти")
        print("-" * 60)
        
        model_names = list(self.models.keys())
        
        with tqdm(total=len(model_names), desc="Прогресс обучения", unit="модель", 
                 bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}") as pbar:
            
            for name in model_names:
                pbar.set_description(f"Обучение: {name[:20]:<20}")
                model = self.models[name]
                
                result = self.train_single_model(name, model, X_train, y_train, X_val, y_val)
                results[name] = result
                
                if result['success']:
                    pbar.set_postfix({
                        'точность': f"{result['accuracy']:.3f}",
                        'время': f"{result['training_time']:.1f}с"
                    })
                    
                    model_path = os.path.join(self.project_root, 'models', 'base_models', 
                                            f'{name.replace("/", "_").replace(" ", "_")}.pkl')
                    self.ensure_directory_exists(model_path)
                    joblib.dump(model, model_path, compress=3)  
                else:
                    pbar.set_postfix({'статус': 'ошибка'})
                
                pbar.update(1)
        
        self._print_results_table(results)
        
        return results
    
    def _print_results_table(self, results):
        """Вывод результатов в виде красивой таблицы"""
        from tabulate import tabulate
        
        print("\n" + "=" * 70)
        print("📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ БАЗОВЫХ МОДЕЛЕЙ")
        print("=" * 70)
        
        table_data = []
        for name, result in results.items():
            if result['success']:
                table_data.append([
                    name,
                    f"{result['accuracy']:.4f}",
                    f"{result['f1_score']:.4f}",
                    f"{result['precision']:.4f}",
                    f"{result['recall']:.4f}",
                    f"{result['training_time']:.2f}с"
                ])
        
        if table_data:
            headers = ['Модель', 'Точность', 'F1-Score', 'Precision', 'Recall', 'Время']
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
            
            best_idx = np.argmax([r['accuracy'] for r in results.values() if r['success']])
            best_name = list(results.keys())[best_idx]
            best_acc = table_data[best_idx][1]
            print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_name} с точностью {best_acc}")
        else:
            print("❌ Ни одна модель не была успешно обучена!")
    
    def get_best_model(self, results):
        """Получение лучшей модели из результатов"""
        best_name = None
        best_accuracy = 0
        
        for name, result in results.items():
            if result['success'] and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_name = name
        
        if best_name and results[best_name]['model']:
            return best_name, results[best_name]['model']
        return None, None


class AdvancedModels:
    """Продвинутые модели (градиентный бустинг) с отключенным параллелизмом"""
    def __init__(self, project_root='.', fast_mode=True):
        self.project_root = project_root
        self.fast_mode = fast_mode
        
        if fast_mode:
            self.models = {
                'XGBoost': None,  
                'LightGBM': None  
            }
        else:
            self.models = {
                'XGBoost': None,
                'LightGBM': None,
                'CatBoost': None
            }
        print("⚠️  В Windows режиме отключены XGBoost/LightGBM/CatBoost из-за проблем с памятью")
    
    def train(self, X_train, y_train, X_val, y_val):
        """Обучение продвинутых моделей - в Windows просто возвращаем пустые результаты"""
        print("\n" + "=" * 60)
        print("⚠️  ПРОПУСК ПРОДВИНУТЫХ МОДЕЛЕЙ")
        print("=" * 60)
        print("В Windows режиме отключены XGBoost/LightGBM/CatBoost")
        print("из-за частых ошибок памяти и DLL.")
        print("Используйте только базовые модели.")
        
        return {}


class ModelEvaluator:
    """Класс для оценки и сравнения моделей"""
    def __init__(self):
        pass
    
    def compare_models(self, base_results, advanced_results):
        """Сравнение всех моделей"""
        from tabulate import tabulate
        
        print("\n" + "=" * 70)
        print("📈 СРАВНИТЕЛЬНАЯ ТАБЛИЦА ВСЕХ МОДЕЛЕЙ")
        print("=" * 70)
        
        all_results = []
        
        for name, result in base_results.items():
            if result.get('success', False):
                all_results.append([
                    name,
                    f"{result['accuracy']:.4f}",
                    f"{result['f1_score']:.4f}",
                    f"{result['training_time']:.2f}с",
                    "Базовая"
                ])
        
        for name, result in advanced_results.items():
            if result.get('success', False):
                all_results.append([
                    name,
                    f"{result['accuracy']:.4f}",
                    f"{result['f1_score']:.4f}",
                    f"{result['training_time']:.2f}с",
                    "Продвинутая"
                ])
        
        all_results.sort(key=lambda x: float(x[1]), reverse=True)
        
        if all_results:
            print(tabulate(all_results, 
                          headers=['Модель', 'Точность', 'F1-Score', 'Время', 'Тип'],
                          tablefmt='grid'))
            
            print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {all_results[0][0]} с точностью {all_results[0][1]}")
            
            return all_results
        else:
            print("❌ Нет результатов для сравнения!")
            return []
    
    def plot_comparison(self, all_results, save_path='results/plots/model_comparison.png'):
        """Визуализация сравнения моделей"""
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        models = [r[0] for r in all_results[:10]]  # Топ-10 моделей
        accuracies = [float(r[1]) for r in all_results[:10]]
        colors = ['#FF6B6B' if 'Ансамбль' in m else 
                 '#4ECDC4' if 'Продвинут' in r[4] else 
                 '#45B7D1' for m, r in zip(models, all_results[:10])]
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(models)), accuracies, color=colors, edgecolor='black')
        
        plt.title('ТОП-10 МОДЕЛЕЙ ПО ТОЧНОСТИ', fontsize=16, fontweight='bold')
        plt.xlabel('Точность', fontsize=12)
        plt.ylabel('Модели', fontsize=12)
        plt.xlim(0, max(accuracies) * 1.1)
        plt.yticks(range(len(models)), models)
        plt.grid(axis='x', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.3f}', ha='left', va='center', fontsize=10)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#45B7D1', edgecolor='black', label='Базовые модели'),
            Patch(facecolor='#4ECDC4', edgecolor='black', label='Продвинутые модели'),
            Patch(facecolor='#FF6B6B', edgecolor='black', label='Ансамблевые методы')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"📊 График сохранен: {save_path}")
        plt.show()