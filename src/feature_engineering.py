import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, project_root='.', fast_mode=True):
        """
        Инициализация инженера признаков
        
        Args:
            project_root: корневая директория проекта
            fast_mode: режим быстрой обработки
        """
        self.project_root = project_root
        self.fast_mode = fast_mode
        
        if fast_mode:
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()  
        
        self.pca = PCA(n_components=0.95, random_state=42)
        self.feature_selector = None
    
    def ensure_directory_exists(self, path):
        """Проверка и создание директории если нужно"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def normalize_features(self, X, scaler_type='standard'):
        """
        Нормализация признаков с выбором метода
        
        Args:
            X: входные данные
            scaler_type: тип нормализации ('standard', 'minmax', 'robust')
        """
        print("🔧 Нормализация признаков...")
        
        if len(X) == 0:
            return X
            
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        with tqdm(total=2, desc="Нормализация", unit="этап",
                 bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
            
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            X_normalized = scaler.fit_transform(X_flat)
            pbar.update(2)
        
        return X_normalized.reshape(original_shape)
    
    def extract_temporal_features(self, X, feature_types=['basic']):
        """
        Извлечение временных признаков из последовательностей
        
        Args:
            X: входные последовательности
            feature_types: типы извлекаемых признаков 
                         ['basic', 'statistical', 'temporal', 'all']
        """
        print("🔧 Извлечение временных признаков...")
        
        temporal_features = []
        n_samples = len(X)
        
        with tqdm(total=n_samples, desc="Извлечение признаков", unit="послед.",
                 bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}") as pbar:
            
            for seq in X:
                features = []
                
                if 'basic' in feature_types or 'all' in feature_types:
                    mean_features = np.mean(seq, axis=0)
                    features.extend(mean_features)
                    
                    std_features = np.std(seq, axis=0)
                    features.extend(std_features)
                
                if 'statistical' in feature_types or 'all' in feature_types:
                    min_features = np.min(seq, axis=0)
                    max_features = np.max(seq, axis=0)
                    features.extend(min_features)
                    features.extend(max_features)
                    
                    q25 = np.percentile(seq, 25, axis=0)
                    q50 = np.percentile(seq, 50, axis=0)
                    q75 = np.percentile(seq, 75, axis=0)
                    features.extend(q25)
                    features.extend(q50)
                    features.extend(q75)
                
                if 'temporal' in feature_types or 'all' in feature_types:
                    if len(seq) > 1:
                        diff_mean = np.diff(seq, axis=0).mean(axis=0)
                        features.extend(diff_mean)
                        
                        if len(seq) > 2:
                            autocorr = []
                            for i in range(seq.shape[1]):
                                corr = np.corrcoef(seq[:-1, i], seq[1:, i])[0, 1]
                                autocorr.append(corr if not np.isnan(corr) else 0)
                            features.extend(autocorr)
                
                if 'all' in feature_types:
                    energy = np.sum(seq ** 2, axis=0) / len(seq)
                    features.extend(energy)
                
                temporal_features.append(features)
                pbar.update(1)
        
        feature_lengths = [len(f) for f in temporal_features]
        if len(set(feature_lengths)) > 1:
            print(f"⚠️  Разная длина признаков: {set(feature_lengths)}")
            min_len = min(feature_lengths)
            temporal_features = [f[:min_len] for f in temporal_features]
        
        print(f"✅ Извлечено признаков на пример: {len(temporal_features[0])}")
        return np.array(temporal_features)
    
    def reduce_dimensionality(self, X, method='pca', n_components=0.95):
        """
        Уменьшение размерности признаков
        
        Args:
            X: входные данные
            method: метод ('pca', 'svd', 'tsne')
            n_components: количество компонентов или доля дисперсии
        """
        print(f"🔧 Уменьшение размерности ({method})...")
        
        if self.fast_mode and X.shape[1] > 50:
            print(f"⚡ Быстрый режим: ограничиваем {X.shape[1]} -> 50 признаков")
            return X[:, :50]
        
        with tqdm(total=2, desc="Уменьшение размерности", unit="этап",
                 bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
            
            if method == 'pca':
                if isinstance(n_components, float):
                    self.pca = PCA(n_components=n_components, random_state=42)
                else:
                    self.pca = PCA(n_components=min(n_components, X.shape[1]), random_state=42)
                
                X_reduced = self.pca.fit_transform(X)
                pbar.update(1)
                
                explained_var = self.pca.explained_variance_ratio_.sum()
                print(f"   Сохранено {X_reduced.shape[1]} компонент "
                      f"({explained_var*100:.1f}% дисперсии)")
                
            elif method == 'svd':
                n_comp = min(n_components if isinstance(n_components, int) else 50, X.shape[1])
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                X_reduced = svd.fit_transform(X)
                pbar.update(1)
                
                print(f"   Сохранено {X_reduced.shape[1]} компонент (SVD)")
                
            elif method == 'tsne':
                print("⚠️  t-SNE может быть медленным для больших данных...")
                tsne = TSNE(n_components=2 if isinstance(n_components, int) else 2, 
                          random_state=42, 
                          perplexity=min(30, X.shape[0] // 3))
                X_reduced = tsne.fit_transform(X)
                pbar.update(1)
                
                print(f"   Уменьшено до 2D (t-SNE)")
            else:
                print("⚠️  Неизвестный метод, возвращаем исходные данные")
                X_reduced = X
            
            pbar.update(1)
        
        return X_reduced
    
    def select_features(self, X, y, method='kbest', k=10):
        """
        Отбор наиболее важных признаков
        
        Args:
            X: признаки
            y: целевая переменная
            method: метод отбора ('kbest', 'mutual_info')
            k: количество признаков для отбора
        """
        print(f"🔍 Отбор {k} лучших признаков ({method})...")
        
        if X.shape[1] <= k:
            print(f"   Уже меньше {k} признаков, пропускаем отбор")
            return X
        
        with tqdm(total=2, desc="Отбор признаков", unit="этап",
                 bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
            
            if method == 'kbest':
                self.feature_selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
            elif method == 'mutual_info':
                self.feature_selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
            else:
                print(f"⚠️  Неизвестный метод отбора: {method}")
                return X
            
            X_selected = self.feature_selector.fit_transform(X, y)
            pbar.update(1)
            
            selected_indices = self.feature_selector.get_support(indices=True)
            scores = self.feature_selector.scores_
            
            print(f"   Выбрано {len(selected_indices)} признаков из {X.shape[1]}")
            print(f"   Лучший признак: индекс {selected_indices[0]}, score={scores[selected_indices[0]]:.3f}")
            
            pbar.update(1)
        
        return X_selected
    
    def cluster_data(self, X, n_clusters=8, method='kmeans'):
        """
        Кластеризация данных
        
        Args:
            X: данные для кластеризации
            n_clusters: количество кластеров
            method: метод кластеризации ('kmeans', 'minibatch')
        """
        print(f"🔧 Кластеризация на {n_clusters} кластеров ({method})...")
        
        with tqdm(total=2, desc="Кластеризация", unit="этап",
                 bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
            
            if method == 'kmeans':
                if self.fast_mode:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, verbose=0)
                else:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=0)
            elif method == 'minibatch':
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, verbose=0)
            else:
                print(f"⚠️  Неизвестный метод кластеризации: {method}")
                return None
            
            labels = kmeans.fit_predict(X)
            pbar.update(1)
            
            inertia = kmeans.inertia_
            print(f"   Инерция кластеризации: {inertia:.2f}")
            
            pbar.update(1)
        
        return labels
    
    def create_visualizations(self, X, y, save_plots=True):
        """
        Создание визуализаций для анализа данных
        
        Args:
            X: признаки (может быть 3D или 2D)
            y: целевая переменная
            save_plots: сохранять ли графики
        """
        print("📊 Создание визуализаций...")
        
        if save_plots:
            plot_dir = os.path.join(self.project_root, 'results', 'plots')
            self.ensure_directory_exists(plot_dir)
        
        n_plots = 4 if self.fast_mode else 6
        with tqdm(total=n_plots, desc="Построение графиков", unit="график",
                 bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as pbar:
            
            fig = plt.figure(figsize=(15, 10))
            
            ax1 = plt.subplot(2, 3, 1)
            plt.hist(y, bins=min(50, len(np.unique(y))), alpha=0.7, 
                    color='skyblue', edgecolor='black')
            plt.title('Распределение нот', fontsize=12, fontweight='bold')
            plt.xlabel('Высота ноты')
            plt.ylabel('Частота')
            plt.grid(alpha=0.3)
            pbar.update(1)
            
            ax2 = plt.subplot(2, 3, 2)
            
            if len(X.shape) == 3:
                X_2d = X.reshape(-1, X.shape[-1])
            else:
                X_2d = X
            
            if X_2d.shape[1] <= 20 and X_2d.shape[0] > 1: 
                try:
                    corr_matrix = np.corrcoef(X_2d.T)
                    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                               square=True, cbar_kws={"shrink": 0.8})
                    plt.title('Корреляция признаков', fontsize=12, fontweight='bold')
                except Exception as e:
                    plt.text(0.5, 0.5, f'Ошибка корреляции\n{str(e)[:30]}', 
                            ha='center', va='center')
            else:
                plt.text(0.5, 0.5, f'Слишком много признаков\n({X_2d.shape[1]} > 20)', 
                        ha='center', va='center')
            pbar.update(1)
            
            ax3 = plt.subplot(2, 3, 3)
            
            if len(X.shape) == 3:
                if X.shape[2] > 0:
                    first_feature_values = X[:, :, 0].flatten()
                else:
                    first_feature_values = np.array([])
            else:
                if X.shape[1] > 0:
                    first_feature_values = X[:, 0]
                else:
                    first_feature_values = np.array([])
            
            if len(first_feature_values) > 0:
                plt.hist(first_feature_values, bins=30, alpha=0.7, 
                        color='lightcoral', edgecolor='black')
                plt.title('Распределение 1-го признака', fontsize=12, fontweight='bold')
                plt.xlabel('Значение')
                plt.ylabel('Частота')
                plt.grid(alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            pbar.update(1)
            
            ax4 = plt.subplot(2, 3, 4)
            if len(y) > 0:
                unique_notes, counts = np.unique(y, return_counts=True)
                top_n = min(20, len(unique_notes))
                top_indices = np.argsort(counts)[-top_n:][::-1]
                top_notes = unique_notes[top_indices]
                top_counts = counts[top_indices]
                
                bars = plt.bar(range(len(top_notes)), top_counts, 
                              color='gold', edgecolor='black')
                plt.title(f'Топ-{top_n} самых частых нот', fontsize=12, fontweight='bold')
                plt.xlabel('Нота')
                plt.ylabel('Частота')
                plt.xticks(range(len(top_notes)), [f"{int(n)}" for n in top_notes], 
                          rotation=45, fontsize=8)
                plt.grid(alpha=0.3, axis='y')
                
                for i, (bar, count) in enumerate(zip(bars[:10], top_counts[:10])):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(count)}', ha='center', va='bottom', fontsize=8)
            else:
                plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            pbar.update(1)
            
            if not self.fast_mode:
                ax5 = plt.subplot(2, 3, 5)
                
                if len(X.shape) == 3:
                    X_mean = X.mean(axis=1)
                    n_features_to_plot = min(5, X_mean.shape[1])
                    data_to_plot = [X_mean[:, i] for i in range(n_features_to_plot)]
                else:
                    n_features_to_plot = min(5, X.shape[1])
                    data_to_plot = [X[:, i] for i in range(n_features_to_plot)]
                
                if len(data_to_plot) > 0:
                    box = plt.boxplot(data_to_plot, patch_artist=True)
                    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightgray']
                    for patch, color in zip(box['boxes'], colors[:len(data_to_plot)]):
                        patch.set_facecolor(color)
                    
                    plt.title(f'Box plot первых {len(data_to_plot)} признаков', 
                             fontsize=12, fontweight='bold')
                    plt.xlabel('Признак')
                    plt.ylabel('Значение')
                    plt.xticks(range(1, len(data_to_plot) + 1), 
                              [f'Призн.{i}' for i in range(len(data_to_plot))])
                    plt.grid(alpha=0.3, axis='y')
                else:
                    plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                pbar.update(1)
                
                ax6 = plt.subplot(2, 3, 6)
                ax6.axis('off')
                
                if len(X.shape) == 3:
                    shape_info = f"3D: {X.shape[0]}×{X.shape[1]}×{X.shape[2]}"
                    n_features = X.shape[2]
                else:
                    shape_info = f"2D: {X.shape[0]}×{X.shape[1]}"
                    n_features = X.shape[1]
                
                if len(y) > 0:
                    y_stats = f"Медиана: {np.median(y):.1f}\nСреднее: {np.mean(y):.1f}\nСтд: {np.std(y):.1f}"
                else:
                    y_stats = "Нет данных"
                
                info_text = f"""
                ИНФОРМАЦИЯ:
                
                Размерность: {shape_info}
                Всего примеров: {X.shape[0]:,}
                Признаков: {n_features}
                Уникальных нот: {len(np.unique(y)) if len(y) > 0 else 0}
                
                СТАТИСТИКА НОТ:
                {y_stats}
                
                ПРИЗНАКИ:
                Min: {X.min():.3f}
                Max: {X.max():.3f}
                Mean: {X.mean():.3f}
                Std: {X.std():.3f}
                """
                plt.text(0.1, 0.5, info_text, fontsize=9, 
                        verticalalignment='center', fontfamily='monospace')
                pbar.update(1)
            
            plt.suptitle('АНАЛИЗ МУЗЫКАЛЬНОГО ДАТАСЕТА', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_plots:
                plot_path = os.path.join(plot_dir, 'data_analysis.png')
                plt.savefig(plot_path, dpi=120, bbox_inches='tight')
                print(f"✅ График сохранен: {plot_path}")
            
            plt.show()
            pbar.update(n_plots - pbar.n)  
    
    def create_feature_importance_plot(self, X, y, model=None, save_path=None):
        """
        Создание графика важности признаков
        
        Args:
            X: признаки
            y: целевая переменная
            model: модель для оценки важности (если None, использует RandomForest)
            save_path: путь для сохранения
        """
        if save_path:
            self.ensure_directory_exists(save_path)
        
        print("📊 Анализ важности признаков...")
        
        if model is None:
            from sklearn.ensemble import RandomForestClassifier
            
            if len(X.shape) == 3:
                X_2d = X.mean(axis=1)
            else:
                X_2d = X
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_2d, y)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = [f'Признак_{i}' for i in range(len(importances))]
            
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Важность признаков", fontsize=14, fontweight='bold')
            plt.bar(range(min(20, len(indices))), importances[indices[:20]])
            plt.xticks(range(min(20, len(indices))), 
                      [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
            plt.xlabel("Признаки")
            plt.ylabel("Важность")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"✅ График важности признаков сохранен: {save_path}")
            
            plt.show()
            
            return importances, indices
        else:
            print("⚠️  Модель не поддерживает feature_importances_")
            return None, None
    
    def get_feature_summary(self, X, y):
        """Получение сводки по признакам"""
        summary = {
            'n_samples': X.shape[0],
            'n_features': X.shape[-1] if len(X.shape) == 3 else X.shape[1],
            'feature_stats': {}
        }
        
        if len(X.shape) == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X
        
        for i in range(min(10, X_flat.shape[1])):
            summary['feature_stats'][f'feature_{i}'] = {
                'mean': float(np.mean(X_flat[:, i])),
                'std': float(np.std(X_flat[:, i])),
                'min': float(np.min(X_flat[:, i])),
                'max': float(np.max(X_flat[:, i])),
                'median': float(np.median(X_flat[:, i]))
            }
        
        if len(y) > 0:
            summary['target_stats'] = {
                'n_classes': len(np.unique(y)),
                'class_distribution': {int(cls): int(count) 
                                      for cls, count in zip(*np.unique(y, return_counts=True))}
            }
        else:
            summary['target_stats'] = {}
        
        return summary


    def get_fitted_scaler(self):
        """
        Получение обученного scaler
        
        Returns:
            обученный scaler или None если не обучен
        """
        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            return self.scaler
        else:
            print("⚠️ Scaler не был обучен!")
            return None
    