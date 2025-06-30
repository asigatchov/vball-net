import pandas as pd
import numpy as np
import cv2
import os
import argparse
from typing import List, Tuple

def read_ball_log(csv_path: str) -> pd.DataFrame:
    """Читает CSV-файл с логом детекций мяча."""
    return pd.read_csv(csv_path)

def interpolate_missing_frames(df: pd.DataFrame, max_gap: int = 60) -> pd.DataFrame:
    """Интерполирует координаты для пропущенных кадров с Visibility = 0."""
    df_interpolated = df.copy()
    df_interpolated['Interpolated'] = 0

    subframes = df['Subframe'].unique() if 'Subframe' in df.columns else [None]

    for subframe in subframes:
        if subframe is not None:
            df_sub = df_interpolated[df_interpolated['Subframe'] == subframe]
        else:
            df_sub = df_interpolated

        for i in range(1, len(df_sub)):
            if df_sub.iloc[i]['Visibility'] == 0 and df_sub.iloc[i-1]['Visibility'] == 1:
                start_idx = df_sub.index[i-1]
                start_frame = df_sub.iloc[i-1]['Frame']
                start_x, start_y = df_sub.iloc[i-1]['X'], df_sub.iloc[i-1]['Y']

                for j in range(i, min(i + max_gap + 1, len(df_sub))):
                    if df_sub.iloc[j]['Visibility'] == 1:
                        end_idx = df_sub.index[j]
                        end_frame = df_sub.iloc[j]['Frame']
                        end_x, end_y = df_sub.iloc[j]['X'], df_sub.iloc[j]['Y']
                        gap_size = end_frame - start_frame

                        if gap_size <= max_gap:
                            for k in range(df_sub.index[i], end_idx):
                                t = (df_interpolated.at[k, 'Frame'] - start_frame) / gap_size
                                df_interpolated.at[k, 'X'] = int(start_x + t * (end_x - start_x))
                                df_interpolated.at[k, 'Y'] = int(start_y + t * (end_y - start_y))
                                df_interpolated.at[k, 'Visibility'] = 2
                                df_interpolated.at[k, 'Interpolated'] = 1
                        break

    return df_interpolated

def detect_rallies(df: pd.DataFrame, min_rally_length: int = 30) -> List[Tuple[int, int]]:
    """Определяет начало и конец розыгрышей (Visibility = 1 или 2) с минимальной длиной 30 кадров."""
    rallies = []
    in_rally = False
    start_frame = None

    if 'Subframe' in df.columns:
        df_sub = df[df['Subframe'] == 2]
    else:
        df_sub = df

    for i in range(len(df_sub)):
        if df_sub.iloc[i]['Visibility'] in [1, 2] and not in_rally:
            in_rally = True
            start_frame = df_sub.iloc[i]['Frame']
        elif df_sub.iloc[i]['Visibility'] == 0 and in_rally:
            end_frame = df_sub.iloc[i-1]['Frame']
            if end_frame - start_frame + 1 >= min_rally_length:
                rallies.append((start_frame, end_frame))
            in_rally = False
            start_frame = None

    if in_rally and start_frame is not None:
        end_frame = df_sub.iloc[-1]['Frame']
        if end_frame - start_frame + 1 >= min_rally_length:
            rallies.append((start_frame, end_frame))

    return rallies

def filter_short_rallies(df: pd.DataFrame, rallies: List[Tuple[int, int]]) -> pd.DataFrame:
    """Присваивает Visibility=0, X=0, Y=0 для кадров вне розыгрышей длиной ≥30 кадров и добавляет столбец IsRally."""
    df_filtered = df.copy()
    df_filtered['Visibility'] = 0
    df_filtered['X'] = 0.0
    df_filtered['Y'] = 0.0
    df_filtered['IsRally'] = False

    for start, end in rallies:
        mask = (df_filtered['Frame'] >= start) & (df_filtered['Frame'] <= end)
        df_filtered.loc[mask, 'Visibility'] = df.loc[mask, 'Visibility']
        df_filtered.loc[mask, 'X'] = df.loc[mask, 'X']
        df_filtered.loc[mask, 'Y'] = df.loc[mask, 'Y']
        df_filtered.loc[mask, 'IsRally'] = True

    return df_filtered

def visualize_heatmaps(df: pd.DataFrame, frame_idx: int, input_height: int = 288, input_width: int = 512):
    """Визуализирует тепловые карты для всех трех подкадров текущего кадра, если они доступны."""
    if 'Subframe' not in df.columns:
        return

    df_frame = df[df['Frame'] == frame_idx]
    for subframe in range(1, 4):
        df_sub = df_frame[df_frame['Subframe'] == subframe]
        if not df_sub.empty:
            visibility, x, y = df_sub.iloc[0]['Visibility'], df_sub.iloc[0]['X'], df_sub.iloc[0]['Y']
            if visibility in [1, 2]:
                heatmap = np.zeros((input_height, input_width), dtype=np.float32)
                if x > 0 and y > 0:
                    scale_x = input_width / df_sub.iloc[0]['X'] if df_sub.iloc[0]['X'] != 0 else 1
                    scale_y = input_height / df_sub.iloc[0]['Y'] if df_sub.iloc[0]['Y'] != 0 else 1
                    cv2.circle(heatmap, (int(x * scale_x), int(y * scale_y)), 5, 1.0, -1)
                heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_uint8 = heatmap_norm.astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                cv2.imshow(f'Heatmap Subframe {subframe} (Frame {frame_idx})', heatmap_color)
    cv2.waitKey(1)

def visualize_rallies(video_path: str, df: pd.DataFrame, rallies: List[Tuple[int, int]], write_output: bool = False, output_path: str = 'output_rally.mp4', skip_non_rally: bool = True):
    """Визуализирует розыгрыши на видео с аннотациями и тепловыми картами, пропуская неигровые моменты, если skip_non_rally=True."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if write_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        in_rally = any(start <= frame_idx <= end for start, end in rallies)
        if skip_non_rally and not in_rally:
            frame_idx += 1
            continue  # Пропускаем неигровые моменты

        # Аннотация для игровых и неигровых моментов
        if in_rally:
            cv2.putText(frame, "Rally", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Non-Rally", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Отображение точек для второго подкадра (если есть)
        if frame_idx < len(df[df['Subframe'] == 2]) if 'Subframe' in df.columns else frame_idx < len(df):
            row = df[df['Subframe'] == 2].iloc[frame_idx] if 'Subframe' in df.columns else df.iloc[frame_idx]
            if row['Visibility'] in [1, 2]:
                x, y = int(row['X']), int(row['Y'])
                color = (0, 0, 255) if row['Visibility'] == 1 else (255, 0, 0)
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.putText(frame, f"Frame: {frame_idx}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Визуализация тепловых карт только для игровых моментов
        if in_rally:
            visualize_heatmaps(df, frame_idx)

        cv2.namedWindow("Rally Visualization", cv2.WINDOW_NORMAL)
        cv2.imshow('Rally Visualization', frame)

        if write_output and out is not None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if write_output and out is not None:
        out.release()
    cv2.destroyAllWindows()

def analyze_rally(csv_path: str, video_path: str, write_output: bool = False, output_csv: str = 'interpolated_log.csv', output_video: str = 'output_rally.mp4', skip_non_rally: bool = True):
    """Основная функция для анализа и визуализации розыгрышей."""
    df = read_ball_log(csv_path)
    df_interpolated = interpolate_missing_frames(df)
    rallies = detect_rallies(df_interpolated, min_rally_length=30)
    df_filtered = filter_short_rallies(df_interpolated, rallies)
    df_filtered.to_csv(output_csv, index=False)

    print("Обнаруженные розыгрыши (длина ≥ 30 кадров):")
    for start, end in rallies:
        print(f"Розыгрыш: с кадра {start} по кадр {end} (длина: {end - start + 1})")

    if os.path.exists(video_path):
        visualize_rallies(video_path, df_filtered, rallies, write_output, output_video, skip_non_rally)
        if write_output:
            print(f"Видео сохранено как {output_video}")
    else:
        print(f"Видео не найдено: {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Анализ лога детекций мяча и визуализация розыгрышей.")
    parser.add_argument('--csv_path', type=str, required=True, help="Путь к CSV-файлу с логом детекций")
    parser.add_argument('--video_path', type=str, required=True, help="Путь к видеофайлу")
    parser.add_argument('--write_output', action='store_true', default=False, help="Записывать выходное видео")
    parser.add_argument('--skip_non_rally', action='store_true', default=True, help="Пропускать неигровые моменты при визуализации")
    args = parser.parse_args()

    analyze_rally(args.csv_path, args.video_path, args.write_output, skip_non_rally=args.skip_non_rally)

if __name__ == "__main__":
    main()