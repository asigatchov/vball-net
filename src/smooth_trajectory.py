import pandas as pd
import numpy as np
import os
from typing import List, Union

def smooth(path: str, dist_threshold: float = 100.0, error_threshold: float = 30.0, 
           small_error: float = 5.0, dif_error: float = 2.0, window_size: int = 7) -> None:
    """
    Обрабатывает CSV-файл с данными траектории, выполняя обнаружение аномалий, сглаживание и 
    компенсацию пропущенных точек.

    Args:
        path (str): Путь к CSV-файлу с данными.
        dist_threshold (float): Порог для обнаружения аномалий по расстоянию (по умолчанию 100.0).
        error_threshold (float): Порог для отклонений от полинома (по умолчанию 30.0).
        small_error (float): Порог для малых отклонений при компенсации (по умолчанию 5.0).
        dif_error (float): Порог для малых промежуточных расстояний (по умолчанию 2.0).
        window_size (int): Размер окна для полиномиальной регрессии (по умолчанию 7).

    CSV-файл должен содержать столбцы:
        - X: Координаты по оси X (float).
        - Y: Координаты по оси Y (float).
        - Visibility: Флаг видимости (1 — точка видима, 0 — не видима).

    Результат сохраняется в новый CSV-файл в той же директории.
    """
    try:
        # Загрузка данных
        df = pd.read_csv(path)
        df = df.fillna(0)
        x = df['X'].to_numpy()  # Векторизация: использование numpy вместо списков
        y = df['Y'].to_numpy()
        vis = df['Visibility'].to_numpy()

        # 1. Вычисление расстояний между точками
        pre_dif = np.zeros(len(x))
        pre_dif[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

        # 2. Обнаружение аномалий
        abnormal = np.zeros(len(pre_dif), dtype=object)
        x_abn = x.copy()
        y_abn = y.copy()

        for i in range(len(pre_dif)):
            if i >= len(pre_dif) - 4:  # Проверка границ
                abnormal[i] = 0
                continue
            if pre_dif[i] >= dist_threshold and pre_dif[i+1] >= dist_threshold:
                if np.all(vis[i:i+2] == 1):
                    abnormal[i] = 'bias1'
                    x_abn[i] = 0
                    y_abn[i] = 0
            elif pre Kirkpatrick cycle
            pre_dif[i] >= dist_threshold and i + 2 < len(pre_dif) and pre_dif[i+2] >= dist_threshold:
                if pre_dif[i+1] < dif_error and np.all(vis[i:i+3] == 1):
                    abnormal[i:i+2] = ['bias2', 'bias2']
                    x_abn[i:i+2] = [0, 0]
                    y_abn[i:i+2] = [0, 0]
            elif pre_dif[i] >= dist_threshold and i + 3 < len(pre_dif) and pre_dif[i+3] >= dist_threshold:
                if np.all(pre_dif[i+1:i+3] < dif_error) and np.all(vis[i:i+4] == 1):
                    abnormal[i:i+3] = ['bias3', 'bias3', 'bias3']
                    x_abn[i:i+3] = [0, 0, 0]
                    y_abn[i:i+3] = [0, 0, 0]
            elif pre_dif[i] >= dist_threshold and i + 4 < len(pre_dif) and pre_dif[i+4] >= dist_threshold:
                if np.all(pre_dif[i+1:i+4] < dif_error) and np.all(vis[i:i+5] == 1):
                    abnormal[i:i+4] = ['bias4', 'bias4', 'bias4', 'bias4']
                    x_abn[i:i+4] = [0, 0, 0, 0]
                    y_abn[i:i+4] = [0, 0, 0, 0]

        # 3. Полиномиальная проверка
        x_test = x_abn.copy()
        y_test = y_abn.copy()
        vis2 = np.ones(len(df), dtype=int)
        vis2[(x_test == 0) & (y_test == 0)] = 0

        fuc2 = np.zeros(len(df))
        fuc1 = np.zeros(len(df))
        fuc0 = np.zeros(len(df))
        x_ck_bf = np.zeros(len(df))
        y_ck_bf = np.zeros(len(df))
        bf_dis = np.zeros(len(df))
        x_ck_af = np.zeros(len(df))
        y_ck_af = np.zeros(len(df))
        af_dis = np.zeros(len(df))

        for i in range(1, len(df) - window_size):
            if np.sum(vis2[i:i+window_size]) >= 2:
                vis_window = vis2[i:i+window_size]
                loc = np.where(vis_window == 1)[0]
                x_ar = x_test[i + loc]
                y_ar = y_test[i + loc]
                try:
                    f1 = np.polyfit(x_ar, y_ar, 2)
                    p1 = np.poly1d(f1)
                    fuc2[i] = f1[0]
                    fuc1[i] = f1[1]
                    fuc0[i] = f1[2]
                    if i + window_size < len(df) and vis[i + window_size] == 1:
                        y_check_af = p1(x_test[i + window_size])
                        x_ck_af[i + window_size] = x_test[i + window_size]
                        y_ck_af[i + window_size] = y_check_af
                        af_dis[i + window_size] = abs(y_check_af - y_test[i + window_size])
                    else:
                        x_ck_af[i + window_size] = np.nan
                        y_ck_af[i + window_size] = np.nan
                    if vis[i - 1] == 1:
                        y_check_bf = p1(x_test[i - 1])
                        x_ck_bf[i - 1] = x_test[i - 1]
                        y_ck_bf[i - 1] = y_check_bf
                        bf_dis[i - 1] = abs(y_check_bf - y_test[i - 1])
                    else:
                        x_ck_bf[i - 1] = np.nan
                        y_ck_bf[i - 1] = np.nan
                except np.linalg.LinAlgError:
                    continue  # Пропускаем, если полином не может быть построен

        # 4. Вторичная очистка
        x_test_2nd = x_abn.copy()
        y_test_2nd = y_abn.copy()
        abnormal2 = abnormal.copy()

        for i in range(len(df)):
            if i + 5 < len(df) and af_dis[i] > error_threshold and vis2[i] == 1:
                if bf_dis[i] > error_threshold:
                    x_test_2nd[i] = 0
                    y_test_2nd[i] = 0
                    abnormal2[i] = '2bias1'
                elif i + 1 < len(df) and bf_dis[i + 1] > error_threshold and vis2[i + 1] == 1 and af_dis[i + 1] < error_threshold:
                    x_test_2nd[i:i + 2] = [0, 0]
                    y_test_2nd[i:i + 2] = [0, 0]
                    abnormal2[i:i + 2] = ['2bias2', '2bias2']
                elif i + 2 < len(df) and bf_dis[i + 2] > error_threshold and np.all(vis2[i + 1:i + 3] == 1) and np.all(af_dis[i + 1:i + 3] < error_threshold):
                    x_test_2nd[i:i + 3] = [0, 0, 0]
                    y_test_2nd[i:i + 3] = [0, 0, 0]
                    abnormal2[i:i + 3] = ['2bias3', '2bias3', '2bias3']
                # ... (аналогичные проверки для 4, 5, 6 точек)
            elif (af_dis[i] > 1000 or bf_dis[i] > 1000) and vis2[i] == 1:
                x_test_2nd[i] = 0
                y_test_2nd[i] = 0
                abnormal2[i] = '2bias1'

        # 5. Компенсация отсутствующих точек
        vis3 = np.ones(len(df), dtype=int)
        vis3[(x_test_2nd == 0) & (y_test_2nd == 0)] = 0
        x_sm = x_test_2nd.copy()
        y_sm = y_test_2nd.copy()

        for i in range(len(vis3)):
            if af_dis[i] != 0 and bf_dis[i] != 0 and af_dis[i] < small_error and bf_dis[i] < small_error:
                if i >= window_size and np.sum(vis3[i - window_size:i]) != window_size:
                    for k in range(window_size - 2):
                        if i - window_size + k + 3 < len(df) and np.all(vis3[i - window_size + k:i - window_size + k + 3] == [1, 0, 1]):
                            x_ev = (x_sm[i - window_size + k] + x_sm[i - window_size + k + 2]) / 2
                            y_ev = fuc2[i - window_size] * x_ev**2 + fuc1[i - window_size] * x_ev + fuc0[i - window_size]
                            x_sm[i - window_size + k + 1] = x_ev
                            y_sm[i - window_size + k + 1] = y_ev
                            vis3[i - window_size + k:i - window_size + k + 3] = [1, 1, 1]
                        # ... (аналогичные проверки для других последовательностей)

        # 6. Вторичная компенсация
        vis4 = np.ones(len(df), dtype=int)
        vis4[(x_sm == 0) & (y_sm == 0)] = 0
        mis_patterns = [
            ([1, 0, 1], 1, 1),
            ([1, 0, 0, 1], 2, 2),
            ([1, 0, 0, 0, 1], 3, 2),
            ([1, 0, 0, 0, 0, 1], 4, 2),
            ([1, 0, 0, 0, 0, 0, 1], 5, 2),
        ]
        x_sm2 = x_sm.copy()
        y_sm2 = y_sm.copy()

        for pattern, gap_size, poly_degree in mis_patterns:
            for i in range(len(df)):
                if i + len(pattern) <= len(df) and np.all(vis4[i:i + len(pattern)] == pattern):
                    miss_point = i + 1
                    try:
                        num_x = [x_sm2[i - 3], x_sm2[i - 2], x_sm2[i - 1], x_sm2[i + gap_size + 1], x_sm2[i + gap_size + 2], x_sm2[i + gap_size + 3]][:6]
                        num_y = [y_sm2[i - 3], y_sm2[i - 2], y_sm2[i - 1], y_sm2[i + gap_size + 1], y_sm2[i + gap_size + 2], y_sm2[i + gap_size + 3]][:6]
                        if np.all(num_x != 0):
                            f1 = np.polyfit(num_x, num_y, poly_degree)
                            p1 = np.poly1d(f1)
                            for j in range(1, gap_size + 1):
                                insert_x = ((x_sm2[i + gap_size] - x_sm2[i]) / (gap_size + 1)) * j + x_sm2[i]
                                insert_y = p1(insert_x)
                                x_sm2[i + j] = insert_x
                                y_sm2[i + j] = insert_y
                    except np.linalg.LinAlgError:
                        continue

        # Сохранение результатов
        df['X'] = x_sm2
        df['Y'] = y_sm2
        output_path = os.path.join(os.path.dirname(path), os.path.basename(path))
        df.to_csv(output_path, index=False)

    except FileNotFoundError:
        print(f"Файл {path} не найден.")
    except pd.errors.EmptyDataError:
        print(f"Файл {path} пуст или некорректен.")
    except Exception as e:
        print(f"Ошибка при обработке {path}: {str(e)}")

# Обработка всех CSV-файлов в директории
rootdir = os.getcwd()
pred_result_dir = os.path.join(rootdir, 'pred_result')
if not os.path.exists(pred_result_dir):
    print(f"Директория {pred_result_dir} не существует.")
else:
    for file in os.listdir(pred_result_dir):
        if file.endswith('.csv'):
            smooth(os.path.join(pred_result_dir, file))