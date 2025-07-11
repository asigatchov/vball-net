import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime
from utils import get_video_and_csv_pairs, load_data
from train_v1 import reshape_tensors, augment_sequence, parser_args

# Настройка логирования
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = setup_logging(debug=True)

# Функция mixup (взята из исходного кода)
def mixup(frames, heatmaps, alpha=0.5):
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
        frames_mixed.shape,
        heatmaps_mixed.shape,
    )
    return frames_mixed, heatmaps_mixed

# Функция визуализации
def visualize_mixup(train_dataset, seq, grayscale=False, alpha=0.5, save_dir="mixup_visualizations"):
    """
    Visualize the effect of mixup augmentation on frames and heatmaps.

    Args:
        train_dataset: TensorFlow dataset with batched frames and heatmaps.
        seq: Number of frames in sequence.
        grayscale: Whether the frames are grayscale (True) or RGB (False).
        alpha: Alpha parameter for mixup augmentation.
        save_dir: Directory to save visualization images.
    """
    logger.info("Starting mixup visualization...")

    # Создаем директорию для сохранения визуализаций
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Берем один батч из датасета
    for frames, heatmaps in train_dataset.take(1):
        logger.info("Original frames shape: %s, heatmaps shape: %s", frames.shape, heatmaps.shape)

        # Применяем mixup, если alpha > 0
        if alpha > 0:
            frames_mixed, heatmaps_mixed = mixup(frames, heatmaps, alpha)
        else:
            frames_mixed, heatmaps_mixed = frames, heatmaps
            logger.info("Alpha <= 0, skipping mixup augmentation.")

        # Преобразуем тензоры в numpy для визуализации
        frames_np = frames.numpy()
        heatmaps_np = heatmaps.numpy()
        frames_mixed_np = frames_mixed.numpy()
        heatmaps_mixed_np = heatmaps_mixed.numpy()

        # Количество примеров в батче
        batch_size = frames_np.shape[0]
        channels = 1 if grayscale else 3

        # Визуализация для каждого примера в батче
        for i in range(min(batch_size, 2)):  # Ограничиваемся 2 примерами для экономии места
            plt.figure(figsize=(15, 10))

            # Исходные кадры
            for t in range(seq):
                plt.subplot(4, seq, t + 1)
                frame = frames_np[i, t * channels:(t + 1) * channels, :, :].transpose(1, 2, 0)
                if grayscale:
                    frame = frame[:, :, 0]
                    plt.imshow(frame, cmap='gray')
                else:
                    frame = np.clip(frame, 0, 1)
                    plt.imshow(frame)
                plt.title(f"Original Frame {t + 1}")
                plt.axis('off')

            # Исходные тепловые карты
            for t in range(seq):
                plt.subplot(4, seq, seq + t + 1)
                heatmap = heatmaps_np[i, t, :, :]
                plt.imshow(heatmap, cmap='hot')
                plt.title(f"Original Heatmap {t + 1}")
                plt.axis('off')

            # Смешанные кадры
            for t in range(seq):
                plt.subplot(4, seq, 2 * seq + t + 1)
                frame_mixed = frames_mixed_np[i, t * channels:(t + 1) * channels, :, :].transpose(1, 2, 0)
                if grayscale:
                    frame_mixed = frame_mixed[:, :, 0]
                    plt.imshow(frame_mixed, cmap='gray')
                else:
                    frame_mixed = np.clip(frame_mixed, 0, 1)
                    plt.imshow(frame_mixed)
                plt.title(f"Mixed Frame {t + 1}")
                plt.axis('off')

            # Смешанные тепловые карты
            for t in range(seq):
                plt.subplot(4, seq, 3 * seq + t + 1)
                heatmap_mixed = heatmaps_mixed_np[i, t, :, :]
                plt.imshow(heatmap_mixed, cmap='hot')
                plt.title(f"Mixed Heatmap {t + 1}")
                plt.axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"mixup_example_{i}_{timestamp}.png")
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved visualization to {save_path}")

    logger.info("Mixup visualization completed.")

# Пример использования
if __name__ == "__main__":
    # Предполагается, что train_dataset уже создан, как в исходном коде
    # Для примера можно использовать параметры из исходного скрипта
    SEQ = 3
    GRAYSCALE = False
    ALPHA = 0.5
    BATCH_SIZE = 8
    SAVE_DIR = "mixup_visualizations"

    # Загружаем датасет (пример, нужно заменить на реальный train_dataset)
    # Здесь предполагается, что train_dataset уже создан, как в main()
    # Для демонстрации можно использовать заглушку или реальный датасет

    args = parser_args()

    train_pairs = get_video_and_csv_pairs("train", args.seq)


    


    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in train_pairs],
                [p[1] for p in train_pairs],
                [p[2] for p in train_pairs],
            )
        )
        .map(
            lambda t, c, f: tf.py_function(
                func=lambda x, y, z: load_data(
                    x, y, z, "train", args.seq, args.grayscale
                ),
                inp=[t, c, f],
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


    visualize_mixup(train_dataset, args.seq, args.grayscale , args.alpha, SAVE_DIR)
