import os
import pretty_midi
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

class MIDIDataPreprocessor:
    def __init__(self, data_path, max_files=50, seq_length=100):
        """
        Инициализация препроцессора
        
        Args:
            data_path: путь к данным MAESTRO
            max_files: максимальное количество файлов для обработки
            seq_length: длина последовательности для обучения
        """
        self.data_path = data_path
        self.max_files = max_files
        self.seq_length = seq_length
        self.midi_data = []
        
    def load_midi_files(self):
        """Загрузка MIDI файлов с прогресс-баром"""
        midi_files = []
        count = 0
        
        all_files = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.midi') or file.endswith('.mid'):
                    all_files.append(os.path.join(root, file))
        
        if len(all_files) > self.max_files:
            all_files = all_files[:self.max_files]
        
        print(f"Найдено {len(all_files)} MIDI файлов, обрабатываем {self.max_files}...")
        
        for midi_path in tqdm(all_files, desc="Загрузка MIDI файлов", unit="файл"):
            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
                midi_files.append((os.path.basename(midi_path), midi))
                count += 1
            except Exception as e:
                tqdm.write(f"Ошибка при загрузке {os.path.basename(midi_path)}: {e}")
        
        print(f"Успешно загружено {len(midi_files)} MIDI файлов")
        return midi_files
    
    def extract_features(self, midi, file_name):
        """Извлечение признаков из MIDI файла"""
        features = []
        
        try:
            for instrument in midi.instruments:
                for note in instrument.notes:
                    feature_vector = [
                        note.pitch,           
                        note.velocity / 127,  
                        note.start,          
                        note.end - note.start  
                    ]
                    features.append(feature_vector)
            
            return np.array(features)
        except Exception as e:
            tqdm.write(f"Ошибка при извлечении признаков из {file_name}: {e}")
            return np.array([])
    
    def create_sequences(self, features, file_name):
        """Создание последовательностей для обучения"""
        sequences = []
        targets = []
        
        if len(features) > self.seq_length + 1:
            for i in range(len(features) - self.seq_length - 1):
                seq = features[i:i + self.seq_length]
                target = features[i + self.seq_length][0]  
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def preprocess(self):
        """Основной метод препроцессинга с прогресс-барами"""
        print("=" * 60)
        print("НАЧАЛО ПРЕДОБРАБОТКИ ДАННЫХ")
        print("=" * 60)
        
       
        print("\n1. Загрузка MIDI файлов...")
        midi_files = self.load_midi_files()
        
        if not midi_files:
            print("Ошибка: не удалось загрузить MIDI файлы!")
            return np.array([]), np.array([])
        
        all_sequences = []
        all_targets = []
        
        
        print("\n2. Извлечение признаков и создание последовательностей...")
        with tqdm(total=len(midi_files), desc="Обработка файлов", unit="файл") as pbar:
            for file_name, midi in midi_files:
                
                features = self.extract_features(midi, file_name)
                
                if len(features) > self.seq_length + 1:
                    sequences, targets = self.create_sequences(features, file_name)
                    
                    if len(sequences) > 0:
                        all_sequences.append(sequences)
                        all_targets.append(targets)
                        pbar.set_postfix({
                            'послед.': len(sequences),
                            'всего': sum([len(s) for s in all_sequences])
                        })
                
                pbar.update(1)
        
        if all_sequences:
            X = np.vstack(all_sequences)
            y = np.hstack(all_targets)
        else:
            print("Ошибка: не удалось создать последовательности!")
            return np.array([]), np.array([])
        
        print(f"\nФинальная форма данных: X={X.shape}, y={y.shape}")
        
        os.makedirs('../data/processed', exist_ok=True)
        np.save('../data/processed/X.npy', X)
        np.save('../data/processed/y.npy', y)
        
        print("\nДанные успешно сохранены в data/processed/")
        
        return X, y
    
    def analyze_dataset(self, X, y):
        """Анализ датасета с красивым выводом"""
        print("\n" + "=" * 60)
        print("АНАЛИЗ ДАТАСЕТА")
        print("=" * 60)
        
        if len(X) == 0:
            print("Данные отсутствуют!")
            return
        
        print(f"\n📊 Количество последовательностей: {len(X):,}")
        print(f"📏 Длина каждой последовательности: {X.shape[1]}")
        print(f"🔢 Количество признаков: {X.shape[2]}")
        print(f"🎵 Уникальных нот (целевая переменная): {len(np.unique(y))}")
        
        print("\n📈 Статистика по признакам:")
        feature_names = ['Высота тона (pitch)', 'Громкость (velocity)', 
                        'Время начала (start_time)', 'Длительность (duration)']
        
        stats_data = []
        for i, name in enumerate(feature_names):
            stats = {
                'Признак': name,
                'Min': f"{X[:,:,i].min():.3f}",
                'Max': f"{X[:,:,i].max():.3f}",
                'Mean': f"{X[:,:,i].mean():.3f}",
                'Std': f"{X[:,:,i].std():.3f}"
            }
            stats_data.append(stats)
        
        from tabulate import tabulate
        print(tabulate(stats_data, headers="keys", tablefmt="grid"))
        
        print(f"\n🎹 Распределение нот:")
        unique_notes, counts = np.unique(y, return_counts=True)
        print(f"   Самая частая нота: {int(unique_notes[np.argmax(counts)])} (встречается {np.max(counts)} раз)")
        print(f"   Самая редкая нота: {int(unique_notes[np.argmin(counts)])} (встречается {np.min(counts)} раз)")