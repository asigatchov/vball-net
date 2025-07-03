import os
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
import cv2

# Параметры
from constants import IMG_FORMAT, IMG_HEIGHT, IMG_WIDTH, TRAIN_DIR, TEST_DIR, TRAIN_OUTPUT_DIR, TEST_OUTPUT_DIR

# IMG_HEIGHT = 288
# IMG_WIDTH = 512
# TRAIN_DIR = 'data/train'
# TEST_DIR = '    data/test'

# TRAIN_OUTPUT_DIR = '/home/gled/frames/train'
# TEST_OUTPUT_DIR = '/home/gled/frames/test'


def get_video_and_csv_pairs(data_dir):
    """
    Возвращает список пар (видео, CSV, track_id) для всех видео в поддиректориях указанной директории.

    Args:
        data_dir (str): Путь к директории (train).

    Returns:
        list: Список кортежей (video_path, csv_path, track_id).
    """
    pairs = []
    # Получаем все поддиректории в data_dir
    subdirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    for subdir in subdirs:
        video_dir = os.path.join(data_dir, subdir, "video")
        csv_dir = os.path.join(data_dir, subdir, "csv")

        # Проверяем, существует ли папка video
        if os.path.exists(video_dir):
            video_files = sorted(
                [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
            )

            for video_file in video_files:
                track_id = video_file.split(".")[
                    0
                ]  # Например, 'st_lelina_20250430_00001_odd0_30fps'
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
    return new_x, new_y

def preprocess_video_and_csv(video_path, csv_path, track_id, output_dir):
    """
    Извлекает кадры из видео, масштабирует их до 512x288, пересчитывает координаты из CSV
    и сохраняет результаты в output_dir/frames/track_id/.

    Args:
        video_path (str): Путь к видео.
        csv_path (str): Путь к CSV.
        track_id (str): Идентификатор видео (например, 'st_lelina_20250430_00001_odd0_30fps').
        output_dir (str): Корневая директория для сохранения кадров и CSV.
    """
    # Создаём директорию для кадров
    output_frame_dir = os.path.join(output_dir, track_id)
    os.makedirs(output_frame_dir, exist_ok=True)

    # Читаем видео
    vr = VideoReader(video_path, ctx=cpu(0))
    orig_height, orig_width = vr[0].shape[:2]  # Получаем исходное разрешение (высота, ширина)

    # Читаем CSV
    df = pd.read_csv(csv_path)
    new_rows = []

    # Проверяем, что количество кадров совпадает
    num_frames = min(len(vr), len(df))
    print(f"Processing {num_frames} frames for video: {track_id}")

    # Обрабатываем каждый кадр
    for frame_idx in range(num_frames):
        # Извлекаем кадр
        frame = vr[frame_idx].asnumpy()  # (orig_height, orig_width, 3)
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))  # (288, 512, 3)

        # Сохраняем кадр
        frame_path = os.path.join(output_frame_dir, f"{frame_idx}{IMG_FORMAT}")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Пересчитываем координаты
        row = df.iloc[frame_idx]
        x, y, visibility = row['X'], row['Y'], row['Visibility']
        if visibility == 1:
            new_x, new_y = x,y
            new_x, new_y = rescale_coordinates(x, y, orig_width, orig_height, IMG_WIDTH, IMG_HEIGHT)
        else:
            new_x, new_y = -1, -1  # Для невидимых кадров ставим -1

        if new_x == 0 or new_y == 0:
            visibility = 0

        new_rows.append({'Frame': frame_idx, 'Visibility': int(visibility), 'X': int(new_x), 'Y': int(new_y)})

    # Сохраняем новый CSV
    new_df = pd.DataFrame(new_rows)
    new_csv_path = os.path.join(output_dir, f"{track_id}_ball.csv")
    new_df.to_csv(new_csv_path, index=False)
    print(f"Saved processed CSV: {new_csv_path}")

def main():
    """
    Основная функция для предобработки всех видео и CSV в датасете.
    """
    # Создаём корневую директорию frames
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    # Получаем пары видео и CSV
    pairs = get_video_and_csv_pairs(TRAIN_DIR)

    # Обрабатываем каждую пару
    for video_path, csv_path, track_id in pairs:
        print(f"Processing video: {track_id}")
        preprocess_video_and_csv(video_path, csv_path, track_id, TRAIN_OUTPUT_DIR)

    pairs = get_video_and_csv_pairs(TEST_DIR)

    # Обрабатываем каждую пару
    for video_path, csv_path, track_id in pairs:
        print(f"Processing video: {track_id}")
        preprocess_video_and_csv(video_path, csv_path, track_id, TEST_OUTPUT_DIR)

if __name__ == "__main__":
    main()
