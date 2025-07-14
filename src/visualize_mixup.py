import tensorflow as tf
import cv2
import numpy as np
import os
import logging
from datetime import datetime

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = setup_logging(debug=True)

def mixup(frames, heatmaps, alpha=0.5):
    logger = logging.getLogger(__name__)
    batch_size = tf.shape(frames)[0]
    gamma1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lamb = gamma1 / (gamma1 + gamma2)
    # Убрано ограничение lamb >= 0.5 для большего разнообразия
    lamb = tf.reshape(lamb, [batch_size, 1, 1, 1])
    indices = tf.random.shuffle(tf.range(batch_size))
    frames_mixed = frames * lamb + tf.gather(frames, indices) * (1.0 - lamb)
    heatmaps_mixed = heatmaps * lamb + tf.gather(heatmaps, indices) * (1.0 - lamb)
    logger.debug("Mixup lambda values: %s", lamb.numpy())
    logger.debug(
        "Applied mixup: frames_mixed shape %s, heatmaps_mixed shape %s",
        frames_mixed.shape,
        heatmaps_mixed.shape,
    )
    return frames_mixed, heatmaps_mixed, lamb

def visualize_mixup(train_dataset, seq, grayscale=False, alpha=0.5, window_name="Mixup Visualization", save_dir="mixup_visualizations"):
    """
    Visualize the effect of mixup augmentation on frames and heatmaps using OpenCV and save to MP4.

    Args:
        train_dataset: TensorFlow dataset with batched frames and heatmaps.
        seq: Number of frames in sequence.
        grayscale: Whether the frames are grayscale (True) or RGB (False).
        alpha: Alpha parameter for mixup augmentation.
        window_name: Name of the OpenCV window.
        save_dir: Directory to save the MP4 video.
    """
    logger.info("Starting mixup visualization with OpenCV and MP4 saving...")

    # Создаем директорию для сохранения видео
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(save_dir, f"mixup_visualization_{timestamp}.mp4")

    # Берем один батч из датасета
    for frames, heatmaps in train_dataset.take(1):
        logger.info("Original frames shape: %s, heatmaps shape: %s", frames.shape, heatmaps.shape)
        if alpha > 0:
            frames_mixed, heatmaps_mixed, lamb = mixup(frames, heatmaps, alpha)
            logger.info("Mixup applied with lambda: %s", lamb.numpy())
        else:
            frames_mixed, heatmaps_mixed = frames, heatmaps
            lamb = tf.ones([tf.shape(frames)[0], 1, 1, 1], dtype=tf.float32)
            logger.info("Alpha <= 0, skipping mixup augmentation.")

        # Преобразуем тензоры в numpy для визуализации
        frames_np = frames.numpy()
        heatmaps_np = heatmaps.numpy()
        frames_mixed_np = frames_mixed.numpy()
        heatmaps_mixed_np = heatmaps_mixed.numpy()

        # Количество примеров в батче
        batch_size = frames_np.shape[0]
        channels = 1 if grayscale else 3

        # Инициализация VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (512 * 2, 288 * 2))  # 1024x576

        # Берем первый пример из батча
        i = 0
        current_frame = 0

        # Размеры изображений
        height, width = frames_np.shape[2], frames_np.shape[3]  # 288, 512

        while True:
            # Подготавливаем 4 изображения для текущего кадра
            t = current_frame

            # 1. Оригинальный кадр
            frame = frames_np[i, t * channels:(t + 1) * channels, :, :].transpose(1, 2, 0)
            if grayscale:
                frame = frame[:, :, 0:1]
                frame = np.repeat(frame, 3, axis=2)
            else:
                frame = np.clip(frame, 0, 1)
            frame_display = (frame * 255).astype(np.uint8)
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_display, f"Original Frame {t + 1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 2. Оригинальная тепловая карта
            heatmap = heatmaps_np[i, t, :, :]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_display = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.putText(heatmap_display, f"Original Heatmap {t + 1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 3. Смешанный кадр
            frame_mixed = frames_mixed_np[i, t * channels:(t + 1) * channels, :, :].transpose(1, 2, 0)
            if grayscale:
                frame_mixed = frame_mixed[:, :, 0:1]
                frame_mixed = np.repeat(frame_mixed, 3, axis=2)
            else:
                frame_mixed = np.clip(frame_mixed, 0, 1)
            frame_mixed_display = (frame_mixed * 255).astype(np.uint8)
            frame_mixed_display = cv2.cvtColor(frame_mixed_display, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_mixed_display, f"Mixed Frame {t + 1} (lambda={lamb[i,0,0,0]:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 4. Смешанная тепловая карта
            heatmap_mixed = heatmaps_mixed_np[i, t, :, :]
            heatmap_mixed = (heatmap_mixed - heatmap_mixed.min()) / (heatmap_mixed.max() - heatmap_mixed.min() + 1e-8)
            heatmap_mixed_display = cv2.applyColorMap((heatmap_mixed * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.putText(heatmap_mixed_display, f"Mixed Heatmap {t + 1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Объединяем изображения в сетку 2x2
            top_row = np.hstack((frame_display, heatmap_display))
            bottom_row = np.hstack((frame_mixed_display, heatmap_mixed_display))
            combined = np.vstack((top_row, bottom_row))

            # Записываем кадр в видео
            video_writer.write(combined)

            # Отображаем окно
            cv2.imshow(window_name, combined)

            # Обработка клавиш
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):  # Следующий кадр
                current_frame = (current_frame + 1) % seq
            elif key == ord('p'):  # Предыдущий кадр
                current_frame = (current_frame - 1) % seq
            elif key in [ord('q'), 27]:  # q или Esc для выхода
                break

        # Освобождаем ресурсы
        video_writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Saved video to {video_path}")
        break  # Выходим после обработки одного батча

    logger.info("Mixup visualization completed.")

if __name__ == "__main__":
    from utils import get_video_and_csv_pairs, load_data
    from train_v1 import reshape_tensors, augment_sequence, parser_args

    args = parser_args()
    SEQ = args.seq
    GRAYSCALE = args.grayscale
    ALPHA = args.alpha
    BATCH_SIZE = 8
    WINDOW_NAME = "Mixup Visualization"
    SAVE_DIR = "mixup_visualizations"

    train_pairs = get_video_and_csv_pairs("train", args.seq)
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in train_pairs],
                [p[1] for p in train_pairs],
                [p[2] for p in train_pairs],
            )
        )
        .shuffle(buffer_size=1000)
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

    visualize_mixup(train_dataset, SEQ, GRAYSCALE, ALPHA, WINDOW_NAME, SAVE_DIR)