import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import argparse
import os
import sys


def smooth_trajectory(input_csv, output_csv=None, show_plot=True):
    """Сглаживает траекторию мяча с помощью фильтра Калмана."""
    # Загрузка данных
    try:
        data = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Ошибка: файл {input_csv} не найден", file=sys.stderr)
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}", file=sys.stderr)
        return

    # Проверка необходимых колонок
    required_columns = ["Frame", "X", "Y"]
    if not all(col in data.columns for col in required_columns):
        print(
            f"Ошибка: файл должен содержать колонки: {required_columns}",
            file=sys.stderr,
        )
        return

    # Извлечение координат
    measurements = data[["X", "Y"]].values

    # Инициализация фильтра Калмана
    initial_state_mean = [measurements[0, 0], 0, measurements[0, 1], 0]

    transition_matrix = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0], [0, 0, 1, 0]]

    observation_covariance = np.eye(2) * 10  # Шум измерений
    transition_covariance = np.eye(4) * 5  # Шум процесса

    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        initial_state_covariance=np.eye(4),
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
    )

    # Применение фильтра
    smoothed_state_means, _ = kf.smooth(measurements)

    # Извлечение сглаженных координат
    data["X_smoothed"] = smoothed_state_means[:, 0]
    data["Y_smoothed"] = smoothed_state_means[:, 2]

    # Сохранение результатов
    if output_csv:
        try:
            data.to_csv(output_csv, index=False)
            print(f"Сглаженные данные сохранены в {output_csv}")
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}", file=sys.stderr)

    # Визуализация
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(data["X"], data["Y"], "b-", label="Исходная траектория", alpha=0.5)
        plt.plot(
            data["X_smoothed"],
            data["Y_smoothed"],
            "r-",
            label="Сглаженная траектория",
        )
        plt.scatter(data["X"], data["Y"], c="blue", s=10, alpha=0.5)
        plt.scatter(data["X_smoothed"], data["Y_smoothed"], c="red", s=10)
        plt.xlabel("X координата")
        plt.ylabel("Y координата")
        plt.title("Сглаживание траектории мяча")
        plt.legend()
        plt.grid(True)

        # Проверяем, есть ли дисплей
        if not show_plot :
            print(
                "График не может быть отображен (нет DISPLAY), сохраняю в файл plot.png"
            )
            plt.savefig("plot.png")
        else:
            plt.show()
    except Exception as e:
        print(f"Ошибка при построении графика: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Сглаживание траектории мяча на видео")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Путь к CSV файлу с координатами мяча",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения сглаженных данных (опционально)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Не отображать график (полезно для серверного использования)",
    )

    args = parser.parse_args()

    smooth_trajectory(args.csv_path, args.output, show_plot=not args.no_plot)


if __name__ == "__main__":
    main()
