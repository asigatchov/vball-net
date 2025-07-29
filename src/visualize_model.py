import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import uuid
from model.VballNetV2 import VballNetV2  # Предполагается, что модель VballNetV2 доступна

def extract_frames(video_path, num_frames=9, frame_skip=0, target_size=(512, 288)):
    """
    Извлекает указанное количество кадров из видео, пропуская заданное количество кадров.
    
    Args:
        video_path: Путь к видеофайлу.
        num_frames: Количество кадров для извлечения.
        frame_skip: Количество кадров для пропуска между извлеченными кадрами.
        target_size: Целевой размер кадров (ширина, высота).
    
    Returns:
        numpy массив формы (1, num_frames, height, width) в grayscale.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    frames = []
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_skip)
    while len(frames) < num_frames:
        # Устанавливаем позицию на нужный кадр
        
        ret, frame = cap.read()
        if not ret:
            print(f"Достигнут конец видео или ошибка чтения на кадре {frame_count + frame_skip + 1}")
            break

        # Преобразуем в grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Изменяем размер до 512x288
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame)
        frame_count += 1

    cap.release()

    if len(frames) < num_frames:
        print(f"Извлечено только {len(frames)} кадров вместо {num_frames}")
        # Дополняем черными кадрами, если не хватает
        while len(frames) < num_frames:
            frames.append(np.zeros(target_size, dtype=np.uint8))

    # Преобразуем в тензор формы (1, num_frames, height, width)
    frames = np.array(frames).astype(np.float32) / 255.0  # Нормализация
    frames = frames[np.newaxis, ...]  # Добавляем batch
    return frames

def save_input_frames(frames, output_dir="layer_visualizations"):
    """
    Сохраняет входные grayscale кадры как отдельные изображения.
    
    Args:
        frames: Тензор формы (1, num_frames, height, width).
        output_dir: Папка для сохранения изображений.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frames = frames.shape[1]
    for i in range(num_frames):
        plt.figure(figsize=(6, 4))
        plt.imshow(frames[0, i, :, :], cmap='gray')
        plt.title(f'Входной кадр {i+1}')
        plt.axis('off')
        plt.savefig(f'{output_dir}/input_frame_{i+1}_{uuid.uuid4()}.png')
        plt.close()

def save_heatmaps(heatmaps, output_dir="layer_visualizations"):
    """
    Сохраняет финальные тепловые карты как отдельные изображения.
    
    Args:
        heatmaps: Тензор формы (batch, num_frames, height, width).
        output_dir: Папка для сохранения изображений.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frames = heatmaps.shape[1]
    for i in range(num_frames):
        plt.figure(figsize=(6, 4))
        plt.imshow(heatmaps[0, i, :, :], cmap='viridis')
        plt.title(f'Тепловая карта {i+1}')
        plt.axis('off')
        plt.savefig(f'{output_dir}/heatmap_{i+1}_{uuid.uuid4()}.png')
        plt.close()

def visualize_feature_maps(model, sample_input, layer_names, output_dir="layer_visualizations"):
    """
    Визуализирует карты признаков (feature maps) из указанных слоев модели VballNetV2.
    
    Args:
        model: Модель VballNetV2.
        sample_input: Входной тензор формы (1, 9, 512, 288) для grayscale.
        layer_names: Список имен слоев для визуализации.
        output_dir: Папка для сохранения изображений.
    """
    # Создаем папку для сохранения изображений, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Сохраняем входные кадры
    save_input_frames(sample_input, output_dir)

    # Создаем модель для получения промежуточных выходов
    try:
        intermediate_outputs = [model.get_layer(name).output for name in layer_names]
    except ValueError as e:
        print(f"Ошибка: {e}")
        print("Проверьте имена слоев с помощью model.summary()")
        return

    visualization_model = tf.keras.Model(inputs=model.input, outputs=intermediate_outputs)

    # Получаем карты признаков
    feature_maps = visualization_model.predict(sample_input)

    # Визуализация для каждого слоя
    for layer_name, feature_map in zip(layer_names, feature_maps):
        # Для motion_prompt_layer берем первый элемент кортежа (attention_map)
        if layer_name == 'motion_prompt_layer' and isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[0]  # Извлекаем attention_map, игнорируя loss

        # Проверяем форму тензора
        if len(feature_map.shape) == 4:  # Форма (batch, channels, height, width)
            channels = feature_map.shape[1]
            height = feature_map.shape[2]
            width = feature_map.shape[3]
            cols = min(channels, 8)  # Ограничиваем до 8 каналов для визуализации
            rows = (channels + cols - 1) // cols

            plt.figure(figsize=(cols * 3, rows * 3))
            for i in range(min(channels, cols * rows)):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(feature_map[0, i, :, :], cmap='viridis')
                plt.title(f'Канал {i+1}')
                plt.axis('off')
            plt.suptitle(f'Слой: {layer_name}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{layer_name}_{uuid.uuid4()}.png')
            plt.close()

        elif len(feature_map.shape) == 5:  # Форма (batch, frames, channels, height, width)
            frames = feature_map.shape[1]
            channels = feature_map.shape[2] if feature_map.shape[2] != 1 else 1
            height = feature_map.shape[-2]
            width = feature_map.shape[-1]
            cols = min(frames, 4)  # Ограничиваем до 4 кадров
            rows = (frames + cols - 1) // cols

            plt.figure(figsize=(cols * 3, rows * 3))
            for i in range(min(frames, cols * rows)):
                plt.subplot(rows, cols, i + 1)
                # Для карт внимания channels=1, иначе берем первый канал
                img = feature_map[0, i, 0, :, :] if channels == 1 else feature_map[0, i, 0, :, :]
                plt.imshow(img, cmap='viridis')
                plt.title(f'Кадр {i+1}')
                plt.axis('off')
            plt.suptitle(f'Слой: {layer_name}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{layer_name}_{uuid.uuid4()}.png')
            plt.close()

        # Сохраняем тепловые карты отдельно для activation_6
        if layer_name == 'activation_6':
            save_heatmaps(feature_map, output_dir)

def main():
    # Параметры модели
    height = 288  # Высота кадра
    width = 512   # Ширина кадра
    in_dim = 9    # 9 grayscale кадров
    out_dim = 9   # 9 тепловых карт на выходе
    fusion_layer_type = "TypeA"

    # Путь к видеофайлу
    video_path = "data/train/match_gtu_20250316_backline/video/gtu_20250316_001.mp4"

    # Создаем модель
    model = VballNetV2(height, width, in_dim, out_dim, fusion_layer_type)

    # Извлекаем 9 кадров, пропуская 150 кадров
    try:
        sample_input = extract_frames(video_path, num_frames=9, frame_skip=50, target_size=(width, height))
    except Exception as e:
        print(f"Ошибка при извлечении кадров: {e}")
        return

    # Список слоев для визуализации
    layer_names = [
        'motion_prompt_layer',  # Карты внимания из MotionPromptLayer
        'separable_conv2d',    # Первый сверточный слой энкодера
        'separable_conv2d_2',  # Второй сверточный слой энкодера
        'separable_conv2d_3',  # Третий сверточный слой энкодера
        'permute_1',           # Выход механизма пространственного внимания
        'conv2d_1',            # Финальный сверточный слой перед fusion
        'fusion_layer_type_a', # Выход слоя fusion
        'activation_6'         # Финальный выход после сигмоиды (тепловые карты)
    ]

    # layer_names = [
    #     'input_layer', 
    #     'separable_conv2d', 
    #     'activation', 
    #     'batch_normalization', 
    #     'separable_conv2d_1', 
    #     'activation_1', 
    #     'batch_normalization_1',
    #     'max_pooling2d',
    #     'separable_conv2d_2',
    #     'activation_2',
    #     'batch_normalization_2',
    #     'max_pooling2d_1',
    #     'separable_conv2d_3', 
    #     'activation_3', 
    #     'batch_normalization_3', 
    #     'permute', 
    #     'global_average_pooling2d', 
    #     'global_max_pooling2d', 
    #     'reshape_1', 
    #     'reshape_2', 
    #     'concatenate', 
    #     'up_sampling2d', 
    #     'conv2d', 
    #     'permute_1', 
    #     'up_sampling2d_1', 
    #     'concatenate_1', 
    #     'separable_conv2d_4', 
    #     'activation_4', 
    #     'batch_normalization_4', 
    #     'up_sampling2d_2', 
    #     'concatenate_2', 
    #     'separable_conv2d_5', 
    #     'activation_5', 
    #     'batch_normalization_5', 
    #     'reshape', 
    #     'conv2d_1', 
    #     'motion_prompt_layer', 
    #     'fusion_layer_type_a', 
    #     'activation_6'
    # ]

    # Визуализируем карты признаков
    visualize_feature_maps(model, sample_input, layer_names)

if __name__ == '__main__':
    main()