import os
import pandas as pd
import numpy as np
import cv2

# Параметры


def get_video_and_csv_pairs(data_dir):
    """
    Возвращает список пар (видео, CSV, track_id) для всех матчей в указанной директории.

    Args:
        data_dir (strconstexpr: Путь к директории.

    Returns:
        list: Список кортежей (video_path, csv_path, track_id).
    """
    pairs = []
    match_dirs = [d for d in os.listdir(data_dir) if d.startswith('match') and os.path.isdir(os.path.join(data_dir, d))]

    for match_dir in match_dirs:
        video_dir = os.path.join(data_dir, match_dir, 'video')
        csv_dir = os.path.join(data_dir, match_dir, 'csv')
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])

        for video_file in video_files:
            track_id = video_file.split('.')[0]  # Например, 'st_lelina_20250430_00001_odd0_30fps'
            csv_file = f"{track_id}_ball.csv"
            video_path = os.path.join(video_dir, video_file)
            csv_path = os.path.join(csv_dir, csv_file)

            if os.path.exists(csv_path):
                pairs.append((video_path, csv_path, track_id))

    return pairs

def rescale_coordinates(x, y, orig_width, orig_height, new_width=512, new_height=288):
    """
    Пересчитывает координаты (x, y) под новое разрешение.

    Args:
        x (float): Исходная координата X.
        y (float): Исходная координата Y.
        orig_width (int): Исходная ширина видео.
        orig_height (int): Исходная высота видео.
        new_width (int): Новая ширина (512).
        new_height (int): Новая высота (288).

    Returns:
        tuple: (new_x, new_y) — пересчитанные координаты.
    """
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    new_x = int(x * scale_x)
    new_y = int(y * scale_y)
    if new_x >= 512:
        new_x = 511
    if new_y >= 288:
        new_y = 287
    return new_x, new_y

def preprocess_video_and_csv(video_path, csv_path, track_id, output_dir):
    """
    Извлекает кадры из видео, масштабирует их до 512x288, пересчитывает координаты из CSV
    и сохраняет результаты в output_dir/frames/track_id.

    Args:
        video_path (str): Путь к видео.
        csv_path (str): Путь к CSV.
        track_id (str): Идентификатор видео (например, 'st_lelina_20250430_00001_odd0_30fps').
        output_dir (str): Корневая директория для сохранения кадров и CSV.
    """
    # Создаем директорию для кадров
    output_frame_dir = os.path.join(output_dir, track_id)
    os.makedirs(output_frame_dir, exist_ok=True)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Не удалось открыть видео {video_path}")
        return

    # Получаем параметры видео
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Читаем CSV
    df = pd.read_csv(csv_path)
    new_rows = []

    # Получаем количество кадров
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(num_frames, len(df))
    print(f"Processing {num_frames} frames for video: {track_id}")

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx} from {video_path}")
            break

        # Конвертируем BGR в RGB и масштабируем
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

        # Сохраняем кадр
        frame_path = os.path.join(output_frame_dir, f"{frame_idx}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Пересчитываем координаты
        row = df.iloc[frame_idx]
        x, y, visibility = row['X'], row['Y'], row['Visibility']
        if visibility == 1:
            new_x, new_y = rescale_coordinates(x, y, orig_width, orig_height, IMG_WIDTH, IMG_HEIGHT)
        else:
            new_x, new_y = -1, -1

        if new_x == 0 or new_y == 0:
            visibility = 0

        new_rows.append({'Frame': frame_idx, 'Visibility': int(visibility), 'X': int(new_x), 'Y': int(new_y)})

        frame_idx += 1

    # Освобождаем видео
    cap.release()

    # Сохраняем новый CSV
    new_df = pd.DataFrame(new_rows)
    new_csv_path = os.path.join(output_dir, f"{track_id}_ball.csv")
    new_df.to_csv(new_csv_path, index=False)
    print(f"Saved processed CSV: {new_csv_path}")

def main():
    """
    Основная функция для предобработки всех видео и CSV в датасете.
    """
    # Создаем корневую директорию
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Получаем пары видео и CSV
    pairs = get_video_and_csv_pairs(TRAIN_DIR)
    for video_path, csv_path, track_id in pairs:
        print(f"Processing video: {track_id}")
        preprocess_video_and_csv(video_path, csv_path, track_id, TRAIN_OUTPUT_DIR)

    pairs = get_video_and_csv_pairs(TEST_DIR)
    for video_path, csv_path, track_id in pairs:
        print(f"Processing video: {track_id}")
        preprocess_video_and_csv(video_path, csv_path, track_id, TEST_OUTPUT_DIR)

if __name__ == "__main__":
    main()