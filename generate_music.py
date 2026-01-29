# generate_music.py - обновленная версия
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

import argparse
from music_generator import MusicGenerator

def main():
    """Основная функция для генерации музыки"""
    print("=" * 70)
    print("🎵 ГЕНЕРАЦИЯ МУЗЫКИ С ПОМОЩЬЮ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 70)
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Генерация музыки с помощью ИИ')
    parser.add_argument('--model', type=str, default='models/voting_ensemble_windows.pkl',
                       help='Путь к обученной модели')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl',
                       help='Путь к scaler (опционально)')
    parser.add_argument('--output', type=str, default='results/generated_music',
                       help='Папка для сохранения результатов')
    parser.add_argument('--num_notes', type=int, default=50,
                       help='Количество нот для генерации')
    parser.add_argument('--seq_length', type=int, default=25,
                       help='Длина последовательности для модели')
    parser.add_argument('--tempo', type=int, default=120,
                       help='Темп музыки (BPM)')
    parser.add_argument('--instrument', type=str, default='Acoustic Grand Piano',
                       help='Название инструмента')
    parser.add_argument('--seed_type', type=str, default='random',
                       choices=['random', 'file'],
                       help='Тип seed: random или file')
    parser.add_argument('--seed_file', type=str, 
                       default=None,
                       help='Путь к MIDI файлу для seed')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Параметр творчества (0.1-2.0)')
    
    args = parser.parse_args()
    
    # Проверка наличия файлов
    if not os.path.exists(args.model):
        print(f"❌ Модель не найдена: {args.model}")
        print("Сначала обучите модель с помощью main_windows.py")
        return
    
    # Проверка scaler (опционально)
    if args.scaler and not os.path.exists(args.scaler):
        print(f"⚠️ Scaler не найден: {args.scaler}")
        print("Будет использована ручная нормализация")
        args.scaler = None
    
    # Создаем папку для результатов
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # 1. Инициализация генератора
        print("\n1. Инициализация генератора музыки...")
        generator = MusicGenerator(
            model_path=args.model,
            scaler_path=args.scaler,
            project_root=current_dir
        )
        
        # 2. Подготовка seed последовательности
        print("\n2. Подготовка seed последовательности...")
        if args.seed_type == 'random':
            seed_notes = generator.generate_random_seed(seq_length=args.seq_length)
            print(f"✅ Используется случайный seed ({len(seed_notes)} нот)")
        else:
            if args.seed_file and os.path.exists(args.seed_file):
                seed_notes = generator.load_seed_from_midi(args.seed_file, seq_length=args.seq_length)
                print(f"✅ Seed загружен из файла: {args.seed_file}")
            else:
                print(f"⚠️  Файл не указан или не найден, использую случайный seed")
                seed_notes = generator.generate_random_seed(seq_length=args.seq_length)
        
        # 3. Генерация музыки
        print(f"\n3. Генерация {args.num_notes} нот...")
        print(f"   • Длина последовательности: {args.seq_length}")
        print(f"   • Температура (творчество): {args.temperature}")
        print(f"   • Темп: {args.tempo} BPM")
        print(f"   • Инструмент: {args.instrument}")
        
        generated_notes = generator.generate_from_seed(
            seed_notes=seed_notes,
            num_notes=args.num_notes,
            temperature=args.temperature,
            seq_length=args.seq_length
        )
        
        # 4. Сохранение результатов
        print("\n4. Сохранение результатов...")
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # MIDI файл
        midi_filename = f"generated_music_{timestamp}.mid"
        midi_path = os.path.join(args.output, midi_filename)
        
        generator.create_midi_from_notes(
            notes=generated_notes,
            instrument_name=args.instrument,
            tempo=args.tempo,
            output_path=midi_path
        )
        
        # Текстовый файл с нотами
        txt_filename = f"generated_music_{timestamp}.txt"
        txt_path = os.path.join(args.output, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("🎵 СГЕНЕРИРОВАННАЯ МУЗЫКАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ 🎵\n")
            f.write("=" * 50 + "\n")
            f.write(f"Нот: {len(generated_notes)}\n")
            f.write(f"Темп: {args.tempo} BPM\n")
            f.write(f"Инструмент: {args.instrument}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, (pitch, velocity, start, duration) in enumerate(generated_notes):
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                octave = pitch // 12 - 1
                note = pitch % 12
                note_name = f"{note_names[note]}{octave}"
                f.write(f"[{i:3d}] {note_name:3s} | vel={velocity:.2f} | start={start:.2f} | dur={duration:.2f}\n")
        
        print(f"📝 Текстовый файл сохранен: {txt_filename}")
        
        print("\n" + "=" * 70)
        print("🎉 ГЕНЕРАЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print("=" * 70)
        print(f"📁 Результаты сохранены в папке: {args.output}")
        print(f"🎵 MIDI файл: {midi_filename}")
        print(f"📝 Текстовый файл: {txt_filename}")
        print("\nЧтобы прослушать результат:")
        print(f"1. Откройте файл: {midi_path}")
        print("2. Используйте программу для воспроизведения MIDI (VLC, Windows Media Player)")
        print("3. Или импортируйте в цифровую звуковую рабочую станцию (DAW)")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Ошибка при генерации музыки: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()