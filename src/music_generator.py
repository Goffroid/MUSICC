import numpy as np
import os
import joblib
import pretty_midi
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MusicGenerator:
    """Класс для генерации музыки с использованием обученной модели"""
    
    def __init__(self, model_path, scaler_path=None, project_root='.'):
        """
        Инициализация генератора музыки
        
        Args:
            model_path: путь к обученной модели
            scaler_path: путь к scaler'у для нормализации (опционально)
            project_root: корневая директория проекта
        """
        self.project_root = project_root
        
        print(f"🎵 Загрузка модели из {model_path}...")
        self.model = joblib.load(model_path)
        
        if scaler_path and os.path.exists(scaler_path):
            print(f"📊 Загрузка scaler из {scaler_path}...")
            self.scaler = joblib.load(scaler_path)
            
            if hasattr(self.scaler, 'n_features_in_'):
                print(f"   • Scaler ожидает {self.scaler.n_features_in_} признаков")
            elif hasattr(self.scaler, 'mean_'):
                print(f"   • Scaler имеет {len(self.scaler.mean_)} признаков")
        else:
            print("⚠️  Scaler не указан или не найден. Использую ручную нормализацию.")
            self.scaler = None
        
        print("✅ Генератор музыки инициализирован")
    
    def normalize_sequence(self, sequence):
        """
        Нормализация последовательности нот
        
        Args:
            sequence: массив формы (seq_length, 4) - (pitch, velocity, start, duration)
        
        Returns:
            np.array: нормализованная последовательность
        """
        if self.scaler is not None:
            if sequence.shape[1] != self.scaler.n_features_in_:
                print(f"⚠️  Несоответствие признаков: данные {sequence.shape[1]}, scaler {self.scaler.n_features_in_}")
                print("   Использую ручную нормализацию...")
                return self._manual_normalize(sequence)
            return self.scaler.transform(sequence)
        else:
            return self._manual_normalize(sequence)
    
    def _manual_normalize(self, sequence):
        """Ручная нормализация признаков"""
        normalized = sequence.copy().astype(float)
        
        normalized[:, 0] = sequence[:, 0] / 127.0
        
        
        if sequence[:, 2].max() > 0:
            normalized[:, 2] = sequence[:, 2] / sequence[:, 2].max()
        
        if sequence[:, 3].max() > 0:
            normalized[:, 3] = sequence[:, 3] / sequence[:, 3].max()
        
        return normalized
    
    def extract_features_from_notes(self, notes):
        """
        Извлечение признаков из списка нот
        
        Args:
            notes: список нот в формате [(pitch, velocity, start, duration), ...]
        
        Returns:
            np.array: массив признаков (seq_length, 4)
        """
        features = []
        for note in notes:
            pitch, velocity, start, duration = note
            feature_vector = [
                pitch,
                velocity,
                start,
                duration
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_temporal_features(self, sequence):
        """
        Извлечение временных признаков из нормализованной последовательности
        (такое же как при обучении)
        """
        mean_features = np.mean(sequence, axis=0)
        std_features = np.std(sequence, axis=0)
        
        if len(sequence) > 1:
            diff_features = np.diff(sequence, axis=0).mean(axis=0)
        else:
            diff_features = np.zeros(sequence.shape[1])
        
       
        min_features = np.min(sequence, axis=0)
        max_features = np.max(sequence, axis=0)
        
        combined_features = np.concatenate([
            mean_features, 
            std_features, 
            diff_features,
            min_features,
            max_features
        ])
        
        return combined_features.reshape(1, -1)
    
    def prepare_sequence_for_prediction(self, sequence):
        """
        Подготовка последовательности для предсказания
        
        Args:
            sequence: последовательность нот (seq_length, 4)
        
        Returns:
            np.array: подготовленные признаки для модели
        """
        sequence_normalized = self.normalize_sequence(sequence)
        
        features = self.extract_temporal_features(sequence_normalized)
        
        if hasattr(self.model, 'n_features_in_'):
            if features.shape[1] != self.model.n_features_in_:
                print(f"⚠️  Модель ожидает {self.model.n_features_in_} признаков, а получено {features.shape[1]}")
                if features.shape[1] > self.model.n_features_in_:
                    features = features[:, :self.model.n_features_in_]
                else:
                    padding = np.zeros((1, self.model.n_features_in_ - features.shape[1]))
                    features = np.hstack([features, padding])
        
        return features
    
    def generate_from_seed(self, seed_notes, num_notes=100, temperature=1.0, seq_length=25):
        """
        Генерация музыки из начальной последовательности
        
        Args:
            seed_notes: начальная последовательность нот
            num_notes: количество нот для генерации
            temperature: параметр "творчества" (0.0-2.0)
            seq_length: длина последовательности для модели
        
        Returns:
            list: сгенерированные ноты
        """
        print(f"🎹 Генерация {num_notes} нот из seed длиной {len(seed_notes)}...")
        
        generated_notes = []
        current_sequence = seed_notes.copy()
        
        with tqdm(total=num_notes, desc="Генерация музыки", unit="нот") as pbar:
            for i in range(num_notes):
                if len(current_sequence) > seq_length:
                    current_seq = current_sequence[-seq_length:]
                else:
                    current_seq = current_sequence.copy()
                
                sequence_array = self.extract_features_from_notes(current_seq)
                
                try:
                    X = self.prepare_sequence_for_prediction(sequence_array)
                    
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(X)[0]
                        
                        if temperature != 1.0:
                            probabilities = np.power(probabilities, 1.0/temperature)
                            probabilities = probabilities / probabilities.sum()
                        
                        predicted_pitch = np.random.choice(
                            len(probabilities), 
                            p=probabilities
                        )
                    else:
                        predicted_pitch = self.model.predict(X)[0]
                    
                    pitches = [note[0] for note in current_seq]
                    velocities = [note[1] for note in current_seq]
                    durations = [note[3] for note in current_seq]
                    
                    new_velocity = np.mean(velocities) if velocities else 0.5
                    new_duration = np.mean(durations) if durations else 0.5
                    
                    last_note = current_sequence[-1]
                    new_start = last_note[2] + last_note[3]
                    
                    new_note = (
                        int(predicted_pitch),
                        float(new_velocity),
                        float(new_start),
                        float(new_duration)
                    )
                    
                    generated_notes.append(new_note)
                    current_sequence.append(new_note)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'нота': predicted_pitch,
                        'громкость': f'{new_velocity:.2f}',
                        'длит.': f'{new_duration:.2f}'
                    })
                    
                except Exception as e:
                    print(f"⚠️ Ошибка при генерации ноты {i}: {e}")
                    last_note = current_sequence[-1] if current_sequence else (60, 0.5, 0.0, 0.5)
                    new_note = (
                        np.random.randint(60, 72),
                        0.5,
                        last_note[2] + last_note[3],
                        0.5
                    )
                    generated_notes.append(new_note)
                    current_sequence.append(new_note)
                    pbar.update(1)
        
        print(f"✅ Сгенерировано {len(generated_notes)} нот")
        return generated_notes
    
    def create_midi_from_notes(self, notes, instrument_name="Acoustic Grand Piano", 
                               tempo=120, output_path=None):
        """
        Создание MIDI файла из списка нот
        """
        print(f"🎼 Создание MIDI файла ({len(notes)} нот)...")
        
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
        instrument = pretty_midi.Instrument(program=instrument_program)
        
        for pitch, velocity, start, duration in notes:
            midi_velocity = int(velocity * 127)
            note = pretty_midi.Note(
                velocity=midi_velocity,
                pitch=int(pitch),
                start=float(start),
                end=float(start + duration)
            )
            instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            midi.write(output_path)
            print(f"💾 MIDI файл сохранен: {output_path}")
        
        return midi
    
    def generate_random_seed(self, seq_length=25, pitch_range=(48, 84)):
        """
        Генерация случайной начальной последовательности
        """
        print(f"🎲 Генерация случайного seed ({seq_length} нот)...")
        
        seed_notes = []
        current_time = 0.0
        
        for i in range(seq_length):
            pitch = np.random.randint(pitch_range[0], pitch_range[1])
            velocity = np.random.uniform(0.3, 0.9)
            duration = np.random.uniform(0.25, 1.0)
            
            note = (pitch, velocity, current_time, duration)
            seed_notes.append(note)
            
            current_time += duration
        
        print(f"✅ Случайный seed создан: {len(seed_notes)} нот")
        return seed_notes
    
    def load_seed_from_midi(self, midi_path, seq_length=25):
        """
        Загрузка seed последовательности из MIDI файла
        """
        print(f"📂 Загрузка seed из {midi_path}...")
        
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            all_notes = []
            
            for instrument in midi.instruments:
                for note in instrument.notes:
                    normalized_note = (
                        note.pitch,
                        note.velocity / 127.0,
                        note.start,
                        note.end - note.start
                    )
                    all_notes.append(normalized_note)
            
            all_notes.sort(key=lambda x: x[2])
            
            if len(all_notes) >= seq_length:
                seed_notes = all_notes[:seq_length]
                print(f"✅ Загружено {len(seed_notes)} нот из MIDI файла")
            else:
                print(f"⚠️  В файле только {len(all_notes)} нот, дополняем случайными...")
                seed_notes = all_notes.copy()
                while len(seed_notes) < seq_length:
                    last_note = seed_notes[-1] if seed_notes else (60, 0.5, 0.0, 0.5)
                    new_note = (
                        np.random.randint(48, 84),
                        0.5,
                        last_note[2] + last_note[3],
                        0.5
                    )
                    seed_notes.append(new_note)
            
            return seed_notes
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке MIDI файла: {e}")
            print("Генерирую случайный seed...")
            return self.generate_random_seed(seq_length)