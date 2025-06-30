import csv
import math
import sys
import argparse

# Парсер аргументов командной строки
parser = argparse.ArgumentParser(description="Analyze ball trajectory from CSV file")
parser.add_argument("csv_file", help="Path to the CSV file")
args = parser.parse_args()

# Параметры
FPS = 30  # Частота кадров
PAUSE_THRESHOLD = 40  # Порог для пауз (в кадрах)

# Чтение CSV
data = []
try:
    with open(args.csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                "Frame": int(row["Frame"]),
                "Visibility": int(row["Visibility"]),
                "X": int(row["X"]),
                "Y": int(row["Y"])
            })
except FileNotFoundError:
    print(f"Error: File {args.csv_file} not found")
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Выделение игровых моментов и пауз
game_moments = []
pauses = []
errors = []
current_moment = []
current_pause = []
for i, row in enumerate(data):
    if row["Visibility"] == 1:
        if current_pause:
            pause_length = current_pause[-1]["Frame"] - current_pause[0]["Frame"] + 1
            if pause_length <= PAUSE_THRESHOLD:
                errors.append(current_pause)
            else:
                pauses.append(current_pause)
            current_pause = []
        current_moment.append(row)
    else:
        if current_moment:
            game_moments.append(current_moment)
            current_moment = []
        current_pause.append(row)
if current_moment:
    game_moments.append(current_moment)
if current_pause:
    pause_length = current_pause[-1]["Frame"] - current_pause[0]["Frame"] + 1
    if pause_length <= PAUSE_THRESHOLD:
        errors.append(current_pause)
    else:
        pauses.append(current_pause)

# Расчет скоростей
velocities = []
for moment in game_moments:
    for i in range(len(moment) - 1):
        frame1, frame2 = moment[i], moment[i + 1]
        dx = frame2["X"] - frame1["X"]
        dy = frame2["Y"] - frame1["Y"]
        dt = (frame2["Frame"] - frame1["Frame"]) / FPS
        distance = math.sqrt(dx**2 + dy**2)
        velocity = distance / dt if dt != 0 else 0
        velocities.append({
            "start_frame": frame1["Frame"],
            "end_frame": frame2["Frame"],
            "velocity": velocity
        })

# Расчет ускорений
accelerations = []
for i in range(len(velocities) - 1):
    v1, v2 = velocities[i], velocities[i + 1]
    dv = v2["velocity"] - v1["velocity"]
    t1_mid = (v1["start_frame"] + v1["end_frame"]) / 2 / FPS
    t2_mid = (v2["start_frame"] + v2["end_frame"]) / 2 / FPS
    dt = t2_mid - t1_mid
    acceleration = dv / dt if dt != 0 else 0
    accelerations.append({
        "interval": f"({v1['start_frame']}→{v1['end_frame']}) to ({v2['start_frame']}→{v2['end_frame']})",
        "acceleration": acceleration
    })

# Вывод результатов
print("Игровые моменты:")
for i, moment in enumerate(game_moments, 1):
    frames = [row["Frame"] for row in moment]
    coords = [(row["X"], row["Y"]) for row in moment]
    print(f"Момент {i}: Кадры {frames}, Координаты {coords}")

print("\nОшибки детекции (≤40 кадров):")
for i, error in enumerate(errors, 1):
    frames = [row["Frame"] for row in error]
    print(f"Ошибка {i}: Кадры {frames}")

print("\nПаузы (>40 кадров):")
for i, pause in enumerate(pauses, 1):
    frames = [row["Frame"] for row in pause]
    print(f"Пауза {i}: Кадры {frames}")

print("\nСкорости:")
for v in velocities:
    print(f"Кадры {v['start_frame']}→{v['end_frame']}: {v['velocity']:.1f} пикселей/сек")

print("\nУскорения:")
for a in accelerations:
    print(f"Между интервалами {a['interval']}: {a['acceleration']:.1f} пикселей/сек²")