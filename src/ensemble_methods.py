import numpy as np
import os
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class EnsembleMethods:
    def __init__(self, base_models, project_root='.'):
        self.base_models = base_models
        self.project_root = project_root
        
    def ensure_directory_exists(self, path):
        """Проверка и создание директории если нужно"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    def create_voting_ensemble(self):
        """Создание ансамбля методом голосования - БЕЗ параллелизма для Windows"""
        print("🤝 Создание Voting Ensemble...")
        
        estimators = []
        for name, model in self.base_models.items():
            if model is not None:
                estimators.append((name, model))
                if len(estimators) >= 3:  
                    break
        
        if not estimators:
            print("⚠️ Нет моделей для создания ансамбля!")
            return None
            
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=1,  
            verbose=0
        )
        
        print(f"✅ Voting Ensemble создан из {len(estimators)} моделей")
        return voting_clf
    
    def create_stacking_ensemble(self):
        """Создание стекинг-ансамбля"""
        print("🏗️ Создание Stacking Ensemble...")
        
        estimators = []
        for name, model in self.base_models.items():
            if model is not None:
                estimators.append((name, model))
                if len(estimators) >= 3: 
                    break
        
        if not estimators:
            print("⚠️ Нет моделей для создания ансамбля!")
            return None
            
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, n_jobs=1),
            cv=3,
            n_jobs=1,  
            verbose=0
        )
        
        print(f"✅ Stacking Ensemble создан из {len(estimators)} моделей")
        return stacking_clf
    
    def create_bagging_ensemble(self, base_model, n_estimators=5):
        """Создание бэггинг-ансамбля"""
        print("🎒 Создание Bagging Ensemble...")
        
        bagging_clf = BaggingClassifier(
            estimator=base_model,
            n_estimators=n_estimators,
            max_samples=0.8,
            max_features=0.7,  
            n_jobs=1,  
            random_state=42,
            verbose=0
        )
        
        print(f"✅ Bagging Ensemble создан с {n_estimators} базовыми моделями")
        return bagging_clf
    
    def train_ensemble(self, ensemble, X_train, y_train, ensemble_name="Ансамбль"):
        """Обучение ансамбля с прогресс-баром"""
        if ensemble is None:
            print(f"⚠️ {ensemble_name} не создан!")
            return None, 0.0
            
        print(f"\n🎯 Обучение {ensemble_name}...")
        
        start_time = time.time()
        
        with tqdm(total=1, desc=f"Обучение {ensemble_name}", unit="модель") as pbar:
            try:
                ensemble.fit(X_train, y_train)
                pbar.update(1)
            except Exception as e:
                print(f"❌ Ошибка при обучении {ensemble_name}: {e}")
                return None, 0.0
        
        training_time = time.time() - start_time
        
        print(f"✅ {ensemble_name} обучен за {training_time:.2f} секунд")
        
        return ensemble, training_time
    
    def evaluate_ensemble(self, ensemble, X_test, y_test, ensemble_name="Ансамбль"):
        """Оценка ансамбля"""
        if ensemble is None:
            return 0.0, 0.0, []
            
        print(f"\n📊 Оценка {ensemble_name}...")
        
        with tqdm(total=2, desc="Оценка модели", unit="этап") as pbar:
            y_pred = ensemble.predict(X_test)
            pbar.update(1)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            pbar.update(1)
        
        print(f"✅ {ensemble_name} оценен: Точность={accuracy:.4f}, F1={f1:.4f}")
        
        return accuracy, f1, y_pred
    
    def save_ensemble(self, ensemble, ensemble_name):
        """Сохранение ансамбля"""
        if ensemble is None:
            return
            
        model_path = os.path.join(self.project_root, 'models', f'{ensemble_name}.pkl')
        self.ensure_directory_exists(model_path)
        
        joblib.dump(ensemble, model_path, compress=3)  
        print(f"💾 {ensemble_name} сохранен: {model_path}")