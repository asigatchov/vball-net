import tensorflow as tf
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import cv2
import os
import argparse
import logging
import json
from datetime import datetime
import glob
from pathlib import Path
from constants import HEIGHT, WIDTH, SIGMA, DATASET_DIR, IMG_FORMAT
from utils import (
    custom_loss,
    limit_gpu_memory,
    OutcomeMetricsCallback,
    VisualizationCallback,
)

# Параметры
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH  # 512
BATCH_SIZE = 4  # Уменьшено для стабильности
MAG = 1.0  # Magnitude для тепловых карт
RATIO = 1.0  # Коэффициент масштабирования координат
MODEL_DIR = "models"  # Директория для сохранения модели


def setup_logging(debug=False):
    """
    Настройка логирования с указанным уровнем в зависимости от флага debug.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()


def get_model(model_name, height, width, seq, grayscale=False):
    """
    Получить экземпляр модели по указанному имени.
    """
    in_dim = seq if grayscale else seq * 3
    out_dim = seq
    from model.VballNetFastV1 import VballNetFastV1
    from model.VballNetV1 import VballNetV1

    if model_name == "PlayerNetFastV1":
        from model.PlayerNetFastV1 import PlayerNetFastV1
        return PlayerNetFastV1(input_shape=(9, height, width), output_channels=3)
    if model_name == "VballNetFastV1":
        return VballNetFastV1(height, width, in_dim=in_dim, out_dim=out_dim)
    if model_name == "TrackNetV4":
        from model.TrackNetV4 import TrackNetV4
        return TrackNetV4(height, width, "TypeB")
        
    logger.info(
        f"Creating model {model_name} with height={height}, width={width}, in_dim={in_dim}, out_dim={out_dim}, seq={seq}, grayscale={grayscale}"
    )
    
    if model_name == "VballNetV2b":
        from model.VballNetV2b import VballNetV2b
        return VballNetV2b(height, width, in_dim=in_dim, out_dim=out_dim)
    if model_name == "VballNetV2":
        from model.VballNetV2 import VballNetV2
        return VballNetV2(height, width, in_dim=in_dim, out_dim=out_dim)
    if model_name == "VballNetV3":
        from model.VballNetV3 import VballNetV3
        return VballNetV3(height, width, in_dim=in_dim, out_dim=out_dim)
    
    return VballNetV1(height, width, in_dim=in_dim, out_dim=out_dim)


def get_preprocessed_data_pairs(dataset_root, mode, seq):
    """
    Получить пары (sequence_name, inputs_dir, heatmaps_dir, frame_indices) 
    для предварительно обработанных данных.
    
    Ожидаемая структура:
    dataset_preprocessed/
    ├── match1/
    │   ├── inputs/rally1/0.jpg,1.jpg... (512×288)
    │   └── heatmaps/rally1/0.jpg,1.jpg... (Gaussian heatmaps)
    └── match2/...
    """
    logger = logging.getLogger(__name__)
    pairs = []
    
    if not os.path.exists(dataset_root):
        logger.warning("Dataset root %s does not exist", dataset_root)
        return pairs
    
    # Поиск match директорий
    match_dirs = [
        d for d in os.listdir(dataset_root) 
        if d.startswith("match") and os.path.isdir(os.path.join(dataset_root, d))
    ]
    
    for match_dir in match_dirs:
        match_path = os.path.join(dataset_root, match_dir)
        inputs_base_dir = os.path.join(match_path, "inputs")
        heatmaps_base_dir = os.path.join(match_path, "heatmaps")
        
        if not os.path.exists(inputs_base_dir) or not os.path.exists(heatmaps_base_dir):
            logger.warning("Missing inputs or heatmaps directory in %s", match_path)
            continue
            
        # Поиск rally директорий
        rally_dirs = [
            d for d in os.listdir(inputs_base_dir)
            if os.path.isdir(os.path.join(inputs_base_dir, d))
        ]
        
        for rally_dir in rally_dirs:
            inputs_dir = os.path.join(inputs_base_dir, rally_dir)
            heatmaps_dir = os.path.join(heatmaps_base_dir, rally_dir)
            
            if not os.path.exists(heatmaps_dir):
                logger.warning("Missing heatmaps directory %s", heatmaps_dir)
                continue
                
            # Получить список доступных кадров
            input_files = [
                f for f in os.listdir(inputs_dir) 
                if f.endswith('.jpg') and f.replace('.jpg', '').isdigit()
            ]
            heatmap_files = [
                f for f in os.listdir(heatmaps_dir)
                if f.endswith('.jpg') and f.replace('.jpg', '').isdigit()
            ]
            
            # Получить номера кадров
            input_frames = set(int(f.replace('.jpg', '')) for f in input_files)
            heatmap_frames = set(int(f.replace('.jpg', '')) for f in heatmap_files)
            
            # Пересечение доступных кадров
            available_frames = sorted(input_frames.intersection(heatmap_frames))
            
            if len(available_frames) < seq:
                logger.warning(
                    "Sequence %s has only %d frames, need at least %d",
                    rally_dir, len(available_frames), seq
                )
                continue
                
            # Создать последовательности кадров
            for i in range(len(available_frames) - seq + 1):
                frame_indices = available_frames[i:i + seq]
                
                # Проверить, что индексы последовательны (optional, зависит от требований)
                if max(frame_indices) - min(frame_indices) == seq - 1:
                    sequence_name = f"{match_dir}_{rally_dir}_{frame_indices[0]}"
                    pairs.append((sequence_name, inputs_dir, heatmaps_dir, frame_indices))
    
    logger.info("Found %d valid preprocessed sequences for mode %s", len(pairs), mode)
    return pairs


def load_preprocessed_frames(inputs_dir, frame_indices, grayscale=False):
    """
    Загрузить последовательность кадров из предварительно обработанных изображений.
    """
    logger = logging.getLogger(__name__)
    frames = []
    
    for frame_idx in frame_indices:
        frame_path = os.path.join(inputs_dir, f"{frame_idx}.jpg")
        
        if not os.path.exists(frame_path):
            logger.error("Frame file does not exist: %s", frame_path)
            raise FileNotFoundError(f"Frame file does not exist: {frame_path}")
            
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.error("Failed to load frame: %s", frame_path)
            raise ValueError(f"Failed to load frame: {frame_path}")
            
        # Конвертировать BGR в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.expand_dims(frame, axis=-1)
            
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        
        logger.debug("Loaded frame %s with shape %s", frame_path, frame.shape)
    
    # Объединить кадры по каналам
    try:
        concatenated = tf.concat(frames, axis=2)
        logger.debug("Concatenated frames shape: %s", concatenated.shape)
        return concatenated
    except Exception as e:
        logger.error("Failed to concatenate frames: %s", str(e))
        raise ValueError(f"Failed to concatenate frames: {e}")


def load_preprocessed_heatmaps(heatmaps_dir, frame_indices):
    """
    Загрузить последовательность тепловых карт из предварительно обработанных изображений.
    """
    logger = logging.getLogger(__name__)
    heatmaps = []
    
    for frame_idx in frame_indices:
        heatmap_path = os.path.join(heatmaps_dir, f"{frame_idx}.jpg")
        
        if not os.path.exists(heatmap_path):
            logger.error("Heatmap file does not exist: %s", heatmap_path)
            raise FileNotFoundError(f"Heatmap file does not exist: {heatmap_path}")
            
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        if heatmap is None:
            logger.error("Failed to load heatmap: %s", heatmap_path)
            raise ValueError(f"Failed to load heatmap: {heatmap_path}")
            
        heatmap = heatmap.astype(np.float32) / 255.0
        heatmap = np.expand_dims(heatmap, axis=-1)
        heatmaps.append(heatmap)
        
        logger.debug("Loaded heatmap %s with shape %s", heatmap_path, heatmap.shape)
    
    # Объединить тепловые карты по каналам
    try:
        concatenated = tf.concat(heatmaps, axis=2)
        logger.debug("Concatenated heatmaps shape: %s", concatenated.shape)
        return concatenated
    except Exception as e:
        logger.error("Failed to concatenate heatmaps: %s", str(e))
        raise ValueError(f"Failed to concatenate heatmaps: {e}")


def load_preprocessed_data(sequence_name, inputs_dir, heatmaps_dir, frame_indices, mode, seq, grayscale=False):
    """
    Загрузить данные из предварительно обработанного датасета.
    """
    logger = logging.getLogger(__name__)
    
    # Обработка tensor inputs для tf.py_function
    if isinstance(sequence_name, tf.Tensor):
        sequence_name = sequence_name.numpy().decode('utf-8')
    if isinstance(inputs_dir, tf.Tensor):
        inputs_dir = inputs_dir.numpy().decode('utf-8')
    if isinstance(heatmaps_dir, tf.Tensor):
        heatmaps_dir = heatmaps_dir.numpy().decode('utf-8')
    if isinstance(frame_indices, tf.Tensor):
        frame_indices = frame_indices.numpy().tolist()
        
    frames = load_preprocessed_frames(inputs_dir, frame_indices, grayscale)
    heatmaps = load_preprocessed_heatmaps(heatmaps_dir, frame_indices)
    
    # Установить форму тензоров
    frames.set_shape([IMG_HEIGHT, IMG_WIDTH, seq * (1 if grayscale else 3)])
    heatmaps.set_shape([IMG_HEIGHT, IMG_WIDTH, seq])
    
    logger.debug(
        "Loaded preprocessed data for %s: frames shape %s, heatmaps shape %s",
        sequence_name, frames.shape, heatmaps.shape
    )
    
    return frames, heatmaps


def preload_dataset_to_memory(pairs, seq, grayscale=False):
    """
    Предварительная загрузка всего датасета в память для ускорения обучения.
    Возвращает списки предзагруженных frames и heatmaps.
    """
    logger = logging.getLogger(__name__)
    logger.info("Preloading dataset to memory for faster training...")
    
    preloaded_frames = []
    preloaded_heatmaps = []
    
    for i, (sequence_name, inputs_dir, heatmaps_dir, frame_indices) in enumerate(pairs):
        if i % 100 == 0:
            logger.info(f"Preloaded {i}/{len(pairs)} sequences")
            
        try:
            frames = load_preprocessed_frames(inputs_dir, frame_indices, grayscale)
            heatmaps = load_preprocessed_heatmaps(heatmaps_dir, frame_indices)
            
            # Преобразовать в numpy для экономии памяти
            frames_np = frames.numpy()
            heatmaps_np = heatmaps.numpy()
            
            preloaded_frames.append(frames_np)
            preloaded_heatmaps.append(heatmaps_np)
            
        except Exception as e:
            logger.error(f"Failed to preload sequence {sequence_name}: {e}")
            continue
    
    logger.info(f"Successfully preloaded {len(preloaded_frames)} sequences to memory")
    
    # Подсчет использования памяти
    total_memory_mb = 0
    if preloaded_frames:
        frames_memory = len(preloaded_frames) * preloaded_frames[0].nbytes / (1024 * 1024)
        heatmaps_memory = len(preloaded_heatmaps) * preloaded_heatmaps[0].nbytes / (1024 * 1024)
        total_memory_mb = frames_memory + heatmaps_memory
        
    logger.info(f"Dataset memory usage: {total_memory_mb:.2f} MB")
    
    return preloaded_frames, preloaded_heatmaps


def load_preloaded_data(index, preloaded_frames, preloaded_heatmaps, seq, grayscale=False):
    """
    Загрузить данные из предзагруженного в память датасета по индексу.
    """
    if isinstance(index, tf.Tensor):
        index = index.numpy()
        
    frames = tf.convert_to_tensor(preloaded_frames[index], dtype=tf.float32)
    heatmaps = tf.convert_to_tensor(preloaded_heatmaps[index], dtype=tf.float32)
    
    # Установить форму тензоров
    frames.set_shape([IMG_HEIGHT, IMG_WIDTH, seq * (1 if grayscale else 3)])
    heatmaps.set_shape([IMG_HEIGHT, IMG_WIDTH, seq])
    
    return frames, heatmaps


def reshape_tensors(frames, heatmaps, seq, grayscale=False):
    """
    Изменить форму тензоров для соответствия ожидаемому формату модели.
    """
    logger = logging.getLogger(__name__)
    frames = tf.ensure_shape(
        frames, [IMG_HEIGHT, IMG_WIDTH, seq * (1 if grayscale else 3)]
    )
    heatmaps = tf.ensure_shape(heatmaps, [IMG_HEIGHT, IMG_WIDTH, seq])
    
    # Транспонировать к каналы-первые (N, C, H, W)
    frames = tf.transpose(frames, [2, 0, 1])
    heatmaps = tf.transpose(heatmaps, [2, 0, 1])
    
    logger.debug(
        "Reshaped tensors: frames %s, heatmaps %s", frames.shape, heatmaps.shape
    )
    return frames, heatmaps


def mixup(frames, heatmaps, alpha=0.5):
    """
    Применить mixup аугментацию к батчу данных.
    """
    logger = logging.getLogger(__name__)
    batch_size = tf.shape(frames)[0]
    gamma1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lamb = gamma1 / (gamma1 + gamma2)
    lamb = tf.maximum(lamb, 1.0 - lamb)
    lamb = tf.reshape(lamb, [batch_size, 1, 1, 1])
    
    indices = tf.random.shuffle(tf.range(batch_size))
    frames_mixed = frames * lamb + tf.gather(frames, indices) * (1.0 - lamb)
    heatmaps_mixed = heatmaps * lamb + tf.gather(heatmaps, indices) * (1.0 - lamb)
    
    logger.debug(
        "Applied mixup: frames_mixed shape %s, heatmaps_mixed shape %s",
        frames_mixed.shape, heatmaps_mixed.shape
    )
    return frames_mixed, heatmaps_mixed


def augment_sequence(frames, heatmaps, seq, grayscale=False, alpha=-1.0):
    """
    Применить аугментации к последовательности кадров и тепловых карт.
    """
    logger = logging.getLogger(__name__)
    try:
        tf.debugging.assert_shapes([
            (frames, (seq * (1 if grayscale else 3), 288, 512)),
            (heatmaps, (seq, 288, 512)),
        ])
        tf.debugging.assert_non_negative(
            frames, message="Frames contain negative values"
        )
        tf.debugging.assert_less_equal(frames, 1.0, message="Frames contain values > 1")
        tf.debugging.assert_non_negative(
            heatmaps, message="Heatmaps contain negative values"
        )
        tf.debugging.assert_less_equal(
            heatmaps, 1.0, message="Heatmaps contain values > 1"
        )
        
        logger.debug(
            "Input shapes: frames %s, heatmaps %s", frames.shape, heatmaps.shape
        )
        
        # Транспонировать для применения аугментаций
        frames = tf.transpose(frames, [1, 2, 0])
        heatmaps = tf.transpose(heatmaps, [1, 2, 0])
        
        # Объединить для синхронных аугментаций
        combined = tf.concat([frames, heatmaps], axis=2)
        
        # Применить случайное отражение
        combined = tf.image.random_flip_left_right(combined, seed=None)
        
        logger.debug("After flip: combined shape %s", combined.shape)
        
        # Разделить обратно
        frames = combined[:, :, :seq * (1 if grayscale else 3)]
        heatmaps = combined[:, :, seq * (1 if grayscale else 3):]
        
        # Транспонировать обратно
        frames = tf.transpose(frames, [2, 0, 1])
        heatmaps = tf.transpose(heatmaps, [2, 0, 1])
        
        frames = tf.ensure_shape(frames, [seq * (1 if grayscale else 3), 288, 512])
        heatmaps = tf.ensure_shape(heatmaps, [seq, 288, 512])
        
        logger.debug(
            "After geometric augmentations: frames %s, heatmaps %s",
            frames.shape, heatmaps.shape
        )
        return frames, heatmaps
        
    except Exception as e:
        logger.error("Error in augment_sequence: %s", str(e))
        logger.error("Frames shape: %s", frames.shape if frames is not None else "None")
        logger.error("Heatmaps shape: %s", heatmaps.shape if heatmaps is not None else "None")
        raise


def save_lr_schedule_params(save_dir, initial_learning_rate, decay_steps, decay_rate):
    """
    Сохранить параметры графика скорости обучения в JSON-файл.
    """
    lr_params = {
        "initial_learning_rate": float(initial_learning_rate),
        "decay_steps": int(decay_steps),
        "decay_rate": float(decay_rate),
    }
    with open(os.path.join(save_dir, "lr_schedule_params.json"), "w") as f:
        json.dump(lr_params, f, indent=4)
    logger.info(
        "Сохранены параметры графика скорости обучения в %s",
        os.path.join(save_dir, "lr_schedule_params.json"),
    )


def load_lr_schedule_params(save_dir):
    """
    Загрузить параметры графика скорости обучения из JSON-файла.
    """
    params_file = os.path.join(save_dir, "lr_schedule_params.json")
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            lr_params = json.load(f)
        logger.info("Загружены параметры графика скорости обучения из %s", params_file)
        return (
            lr_params["initial_learning_rate"],
            lr_params["decay_steps"],
            lr_params["decay_rate"],
        )
    logger.warning(
        "Файл параметров графика скорости обучения не найден: %s", params_file
    )
    return None


def parser_args():
    parser = argparse.ArgumentParser(description="Train VballNet model on preprocessed data.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset_preprocessed",
        help="Root directory of preprocessed dataset (default: dataset_preprocessed).",
    )
    parser.add_argument(
        "--preload_memory",
        action="store_true",
        help="Preload entire dataset to memory for faster training (uses more RAM).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--seq", type=int, default=3, help="Number of frames in sequence."
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use grayscale frames with seq input/output channels.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="VballNetFastV1",
        help="Model name to train (VballNetFastV1, VballNetV1, VballNetV2, etc.).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train (default: 50).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=-1.0,
        help="Alpha for mixup augmentation, -1 means no mixup.",
    )
    parser.add_argument(
        "--gpu_memory_limit",
        type=int,
        default=-1,
        help="Limit GPU memory usage in MB, -1 means no limit.",
    )
    return parser.parse_args()


def main():
    args = parser_args()
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting training script for preprocessed data with seq=%d, grayscale=%s, debug=%s, resume=%s, model_name=%s, alpha=%s gpu_memory_limit=%d preload_memory=%s",
        args.seq, args.grayscale, args.debug, args.resume, args.model_name, args.alpha, args.gpu_memory_limit, args.preload_memory
    )
    
    limit_gpu_memory(args.gpu_memory_limit)
    
    if args.seq < 1:
        logger.error("Sequence length must be at least 1, got %d", args.seq)
        raise ValueError(f"Invalid sequence length: {args.seq}")
        
    if args.model_name not in [
        "VballNetFastV1", "VballNetV1", "VballNetV2", "VballNetV2b", 
        "VballNetV3", "PlayerNetFastV1", "TrackNetV4"
    ]:
        logger.error(
            "Invalid model name: %s. Must be one of supported models", args.model_name
        )
        raise ValueError(f"Invalid model name: {args.model_name}")
    
    # Настроить имена и директории модели
    model_name_suffix = (
        f"_seq{args.seq}_grayscale" if args.grayscale and args.seq == 9 else ""
    )
    model_save_name = f"{args.model_name}{model_name_suffix}_preprocessed"
    model_save_dir = os.path.join(MODEL_DIR, model_save_name)
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info("Created model save directory: %s", model_save_dir)
    
    # Получить пары данных
    train_pairs = get_preprocessed_data_pairs(args.dataset_root, "train_preprocessed", args.seq)
    test_pairs = get_preprocessed_data_pairs(args.dataset_root, "test", args.seq)
    
    logger.info("Number of training pairs: %d", len(train_pairs))
    logger.info("Number of test pairs: %d", len(test_pairs))
    
    if len(train_pairs) == 0:
        logger.error("No training data found. Check dataset_root and preprocessed structure.")
        raise ValueError("No training data found. Check dataset_root and preprocessed structure.")
        
    if len(test_pairs) == 0:
        logger.warning("No test data found. Will use training data for validation.")
        test_pairs = train_pairs[:max(1, len(train_pairs) // 10)]  # Use 10% of training data
    
    # Предзагрузка датасета в память (опционально)
    if args.preload_memory:
        logger.info("Preloading datasets to memory...")
        train_frames_mem, train_heatmaps_mem = preload_dataset_to_memory(train_pairs, args.seq, args.grayscale)
        test_frames_mem, test_heatmaps_mem = preload_dataset_to_memory(test_pairs, args.seq, args.grayscale)
        
        # Создать датасеты из предзагруженных данных
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(list(range(len(train_pairs))))
            .map(
                lambda idx: tf.py_function(
                    func=lambda i: load_preloaded_data(
                        i, train_frames_mem, train_heatmaps_mem, args.seq, args.grayscale
                    ),
                    inp=[idx],
                    Tout=[tf.float32, tf.float32],
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda frames, heatmaps: reshape_tensors(
                    frames, heatmaps, args.seq, args.grayscale
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda frames, heatmaps: augment_sequence(
                    frames, heatmaps, args.seq, args.grayscale, args.alpha
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(BATCH_SIZE)
        )
        
        test_dataset = (
            tf.data.Dataset.from_tensor_slices(list(range(len(test_pairs))))
            .map(
                lambda idx: tf.py_function(
                    func=lambda i: load_preloaded_data(
                        i, test_frames_mem, test_heatmaps_mem, args.seq, args.grayscale
                    ),
                    inp=[idx],
                    Tout=[tf.float32, tf.float32],
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda frames, heatmaps: reshape_tensors(
                    frames, heatmaps, args.seq, args.grayscale
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        
    else:
        # Обычная загрузка с диска
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((
                [p[0] for p in train_pairs],  # sequence_name
                [p[1] for p in train_pairs],  # inputs_dir
                [p[2] for p in train_pairs],  # heatmaps_dir
                [p[3] for p in train_pairs],  # frame_indices
            ))
            .map(
                lambda sn, id, hd, fi: tf.py_function(
                    func=lambda w, x, y, z: load_preprocessed_data(
                        w, x, y, z, "train", args.seq, args.grayscale
                    ),
                    inp=[sn, id, hd, fi],
                    Tout=[tf.float32, tf.float32],
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda frames, heatmaps: reshape_tensors(
                    frames, heatmaps, args.seq, args.grayscale
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda frames, heatmaps: augment_sequence(
                    frames, heatmaps, args.seq, args.grayscale, args.alpha
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(BATCH_SIZE)
        )
        
        test_dataset = (
            tf.data.Dataset.from_tensor_slices((
                [p[0] for p in test_pairs],
                [p[1] for p in test_pairs],
                [p[2] for p in test_pairs],
                [p[3] for p in test_pairs],
            ))
            .map(
                lambda sn, id, hd, fi: tf.py_function(
                    func=lambda w, x, y, z: load_preprocessed_data(
                        w, x, y, z, "test", args.seq, args.grayscale
                    ),
                    inp=[sn, id, hd, fi],
                    Tout=[tf.float32, tf.float32],
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .map(
                lambda frames, heatmaps: reshape_tensors(
                    frames, heatmaps, args.seq, args.grayscale
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
    
    # Применить mixup аугментацию к training dataset
    if args.alpha > 0:
        train_dataset = train_dataset.map(
            lambda frames, heatmaps: mixup(frames, heatmaps, args.alpha),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    
    # Добавить prefetch для training dataset
    if args.preload_memory:
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    else:
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    train_size = tf.data.experimental.cardinality(train_dataset).numpy()
    test_size = tf.data.experimental.cardinality(test_dataset).numpy()
    logger.info("Number of training batches: %d", train_size)
    logger.info("Number of test batches: %d", test_size)
    
    # Создать модель
    model = get_model(
        args.model_name,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        seq=args.seq,
        grayscale=args.grayscale,
    )
    model.summary(print_fn=lambda x: logger.info(x))
    
    # Настройки обучения
    initial_epoch = 0
    initial_learning_rate = 1e-3
    decay_steps = train_size * 2
    decay_rate = 0.9
    
    if args.resume:
        # Проверить наличие последней контрольной точки
        latest_checkpoint = os.path.join(model_save_dir, f"{model_save_name}_latest.keras")
        if os.path.exists(latest_checkpoint):
            logger.info("Resuming training from latest checkpoint: %s", latest_checkpoint)
            model = tf.keras.models.load_model(
                latest_checkpoint, custom_objects={"custom_loss": custom_loss}
            )
            
            # Извлечь эпоху из файлов контрольных точек
            checkpoint_files = glob.glob(
                os.path.join(model_save_dir, f"{model_save_name}/{model_save_name}_*.keras")
            )
            if checkpoint_files:
                latest_epoch_file = max(checkpoint_files, key=os.path.getmtime)
                epoch_str = (
                    os.path.basename(latest_epoch_file)
                    .split("_")[-1]
                    .replace(".keras", "")
                )
                initial_epoch = int(epoch_str) if epoch_str.isdigit() else 0
            else:
                initial_epoch = 0
                
            # Загрузить параметры графика скорости обучения
            lr_params = load_lr_schedule_params(model_save_dir)
            if lr_params:
                initial_learning_rate, decay_steps, decay_rate = lr_params
                logger.info(
                    "Восстановлены параметры графика: initial_learning_rate=%f, decay_steps=%d, decay_rate=%f",
                    initial_learning_rate, decay_steps, decay_rate
                )
        else:
            logger.warning(
                "No latest checkpoint found for %s in %s, starting training from scratch.",
                model_save_name, model_save_dir
            )
    
    # Настроить оптимизатор и компиляцию
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["mae"])
    
    logger.info(
        "Model compiled with optimizer=Adam(lr_schedule), loss=custom_loss, metrics=['mae']"
    )
    
    # Пути для сохранения контрольных точек
    latest_checkpoint_path = os.path.join(model_save_dir, f"{model_save_name}_latest.keras")
    best_checkpoint_path = os.path.join(model_save_dir, f"{model_save_name}_best.keras")
    epoch_checkpoint_path = os.path.join(
        model_save_dir, f"{model_save_name}/{model_save_name}_{{epoch:02d}}.keras"
    )
    
    # Сохранить параметры графика скорости обучения
    save_lr_schedule_params(model_save_dir, initial_learning_rate, decay_steps, decay_rate)
    
    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self, lr_schedule, train_size):
            super().__init__()
            self.lr_schedule = lr_schedule
            self.train_size = train_size

        def on_epoch_begin(self, epoch, logs=None):
            current_step = epoch * self.train_size
            current_lr = self.lr_schedule(current_step).numpy()
            logger.info(f"Epoch {epoch + 1}: Learning rate = {current_lr}")

    # Настроить callbacks
    callbacks = [
        # Сохранение последней контрольной точки
        tf.keras.callbacks.ModelCheckpoint(
            filepath=latest_checkpoint_path,
            save_best_only=False,
            save_weights_only=False,
            monitor="val_loss",
            verbose=1,
        ),
        # Сохранение лучшей контрольной точки по val_loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        # Сохранение контрольной точки для каждой эпохи
        tf.keras.callbacks.ModelCheckpoint(
            filepath=epoch_checkpoint_path,
            save_best_only=False,
            save_weights_only=False,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs", model_save_name)
        ),
        tf.keras.callbacks.EarlyStopping(patience=30, monitor="val_loss"),
        OutcomeMetricsCallback(
            validation_data=test_dataset,
            tol=10,
            log_dir=os.path.join(MODEL_DIR, "logs", f"{model_save_name}/outcome"),
        ),
        LearningRateLogger(lr_schedule, train_size),
    ]
    
    logger.info(
        "Callbacks configured: ModelCheckpoint (latest, best, epoch), TensorBoard, EarlyStopping, OutcomeMetricsCallback, LearningRateLogger"
    )
    
    # Сохранить пример аугментированных данных
    for frames, heatmaps in train_dataset.take(1):
        logger.info("Sample batch - Frames shape: %s", frames.shape)
        logger.info("Sample batch - Heatmaps shape: %s", heatmaps.shape)
        
        frames_sample = tf.transpose(frames[0], [1, 2, 0])
        heatmaps_sample = tf.transpose(heatmaps[0], [1, 2, 0])
        
        if args.grayscale:
            tf.io.write_file(
                "augmented_frame_preprocessed.png",
                tf.image.encode_png(
                    tf.cast(tf.image.grayscale_to_rgb(frames_sample[:, :, :1]) * 255, tf.uint8)
                ),
            )
        else:
            tf.io.write_file(
                "augmented_frame_preprocessed.png",
                tf.image.encode_png(tf.cast(frames_sample[:, :, :3] * 255, tf.uint8)),
            )
        tf.io.write_file(
            "augmented_heatmap_preprocessed.png",
            tf.image.encode_png(tf.cast(heatmaps_sample[:, :, 0:1] * 255, tf.uint8)),
        )
        break
    
    logger.info("Starting training...")
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )
    logger.info("Training completed")


if __name__ == "__main__":
    main()