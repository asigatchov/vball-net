import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
import argparse
import logging
from datetime import datetime
import glob
from model.InpaintNet import InpaintNet
import pandas as pd

# Параметры
DATASET_DIR = "data/frames"
BATCH_SIZE = 32
MODEL_DIR = "models"
SEQ_LEN = 18  # Длина последовательности из 18 точек


def setup_logging(debug=False):
    """Настройка логирования с указанным уровнем в зависимости от флага debug."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def custom_loss(y_true, y_pred):
    """Пользовательская функция потерь MSE для координат."""
    mse = K.mean(K.square(y_true - y_pred), axis=[1, 2])
    return mse


def get_model(model_name, seq_len):
    """Создание экземпляра модели InpaintNet."""
    if model_name == "InpaintNet":
        logger.info(f"Создание InpaintNet с seq_len={seq_len}")
        return InpaintNet()
    else:
        logger.error(f"Недопустимое имя модели: {model_name}. Должно быть 'InpaintNet'")
        raise ValueError(f"Недопустимое имя модели: {model_name}")


def get_csv_files(split):
    """Получение списка CSV-файлов для указанного сплита (train или test)."""
    csv_files = glob.glob(os.path.join(DATASET_DIR, split, "*.csv"))
    valid_files = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf8")
            if len(df) >= SEQ_LEN:
                valid_files.append(csv_file)
            else:
                logger.warning(
                    f"Skipping {csv_file}: fewer than {SEQ_LEN} frames ({len(df)} frames)"
                )
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
    return valid_files


def load_coordinate_data(csv_path, split, seq_len):
    """
    Загрузка данных координат из CSV-файла для InpaintNet.
    Генерация недостающих столбцов X_GT, Y_GT, Inpaint_Mask на основе Visibility.
    """
    logger.debug(
        f"Загрузка данных: csv_path={csv_path}, split={split}, seq_len={seq_len}"
    )

    # Чтение CSV-файла
    try:
        df = pd.read_csv(csv_path, encoding="utf8").sort_values(by="Frame").fillna(0)
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return [], [], []

    # Проверка наличия необходимых столбцов
    required_columns = ["Frame", "Visibility", "X", "Y"]
    if not all(col in df.columns for col in required_columns):
        logger.error(
            f"CSV-файл {csv_path} не содержит всех необходимых столбцов: {required_columns}"
        )
        return [], [], []

    # Извлечение данных
    frames = np.array(df["Frame"])
    x_pred = np.array(df["X"], dtype=np.float32)
    y_pred = np.array(df["Y"], dtype=np.float32)
    vis_pred = np.array(df["Visibility"], dtype=np.float32)

    # Генерация истинных координат и маски
    x_true = np.where(vis_pred[:, None], x_pred, 0).astype(
        np.float32
    )  # Истинные X = X для видимых, 0 для невидимых
    y_true = np.where(vis_pred[:, None], y_pred, 0).astype(
        np.float32
    )  # Истинные Y = Y для видимых, 0 для невидимых
    inpaint_mask = (1 - vis_pred).astype(
        np.float32
    )  # Маска: 1 для невидимых, 0 для видимых

    # Формирование последовательностей
    coor_pred = []
    coor_true = []
    inpaint = []
    for i in range(0, len(frames) - seq_len + 1):
        if i + seq_len <= len(frames):
            coor_pred.append(
                np.stack([x_pred[i : i + seq_len], y_pred[i : i + seq_len]], axis=-1)
            )  # [seq_len, 2]
            coor_true.append(
                np.stack([x_true[i : i + seq_len], y_true[i : i + seq_len]], axis=-1)
            )  # [seq_len, 2]
            inpaint.append(inpaint_mask[i : i + seq_len].reshape(-1, 1))  # [seq_len, 1]

    if not coor_pred:  # Handle case where no sequences are generated
        logger.warning(
            f"CSV-файл {csv_path} не содержит достаточно кадров для seq_len={seq_len}"
        )
        return [], [], []

    coor_pred = np.array(coor_pred, dtype=np.float32)  # [N, seq_len, 2]
    coor_true = np.array(coor_true, dtype=np.float32)  # [N, seq_len, 2]
    inpaint = np.array(inpaint, dtype=np.float32)  # [N, seq_len, 1]

    # Нормализация координат (предполагается размер изображения 1920x1080)
    coor_pred[:, :, 0] /= 512.0  # Нормализация X
    coor_pred[:, :, 1] /= 288.0  # Нормализация Y
    coor_true[:, :, 0] /= 512.0  # Нормализация X
    coor_true[:, :, 1] /= 288.0  # Нормализация Y

    logger.debug(f"Loaded {len(coor_pred)} sequences from {csv_path}")
    return coor_pred, inpaint, coor_true


def sequence_generator(csv_files, split, seq_len):
    """Генератор для последовательного чтения последовательностей из CSV-файлов."""
    for csv_path in csv_files:
        coor_pred, inpaint, coor_true = load_coordinate_data(csv_path, split, seq_len)
        if len(coor_pred) == 0:
            continue
        for i in range(len(coor_pred)):
            logger.debug(
                f"Yielding sequence {i} from {csv_path}: coor_pred={coor_pred[i].shape}, inpaint={inpaint[i].shape}, coor_true={coor_true[i].shape}"
            )
            yield coor_pred[i], inpaint[i], coor_true[i]


def create_dataset(csv_files, split, seq_len):
    """Создание датасета из CSV-файлов с использованием генератора."""
    if not csv_files:
        logger.error(f"No valid CSV files found for {split} split")
        return tf.data.Dataset.from_tensor_slices(([], [], []))  # Return empty dataset
    return (
        tf.data.Dataset.from_generator(
            lambda: sequence_generator(csv_files, split, seq_len),
            output_signature=(
                tf.TensorSpec(shape=(seq_len, 2), dtype=tf.float32),  # coor_pred
                tf.TensorSpec(shape=(seq_len, 1), dtype=tf.float32),  # inpaint
                tf.TensorSpec(shape=(seq_len, 2), dtype=tf.float32),  # coor_true
            ),
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def parser_args():
    parser = argparse.ArgumentParser(description="Обучение модели InpaintNet.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Возобновить обучение с последней контрольной точки.",
    )
    parser.add_argument("--debug", action="store_true", help="Включить режим отладки.")
    parser.add_argument(
        "--seq",
        type=int,
        default=SEQ_LEN,
        help=f"Длина последовательности (по умолчанию: {SEQ_LEN}).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="InpaintNet",
        help="Имя модели для обучения (InpaintNet).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох для обучения (по умолчанию: 50).",
    )
    parser.add_argument(
        "--gpu_memory_limit",
        type=int,
        default=-1,
        help="Ограничение памяти GPU в МБ, -1 означает без ограничения.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Начальная скорость обучения (по умолчанию: 0.001).",
    )
    return parser.parse_args()


def main():
    args = parser_args()
    logger.info(
        "Запуск скрипта обучения с seq=%d, debug=%s, resume=%s, model_name=%s, gpu_memory_limit=%d, learning_rate=%f",
        args.seq,
        args.debug,
        args.resume,
        args.model_name,
        args.gpu_memory_limit,
        args.learning_rate,
    )

    # Ограничение памяти GPU
    if args.gpu_memory_limit > 0:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=args.gpu_memory_limit
                        )
                    ],
                )
                logger.info(
                    f"Установлено ограничение памяти GPU: {args.gpu_memory_limit} МБ"
                )
            except RuntimeError as e:
                logger.error(f"Ошибка при установке ограничения памяти GPU: {e}")

    if args.seq < 1:
        logger.error(
            "Длина последовательности должна быть не менее 1, получено %d", args.seq
        )
        raise ValueError(f"Недопустимая длина последовательности: {args.seq}")

    # Создание директории для сохранения модели
    model_save_dir = os.path.join(MODEL_DIR, args.model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info("Создана директория для сохранения модели: %s", model_save_dir)

    # Загрузка пар CSV-файлов
    train_pairs = get_csv_files("train")
    test_pairs = get_csv_files("test")

    logger.info("Количество тренировочных пар: %d", len(train_pairs))
    logger.info("Количество тестовых пар: %d", len(test_pairs))
    if len(train_pairs) == 0:
        logger.error(
            "Не найдены тренировочные данные. Проверьте %s/train и файлы CSV.",
            DATASET_DIR,
        )
        raise ValueError(
            "Не найдены тренировочные данные. Проверьте %s/train и файлы CSV."
            % DATASET_DIR
        )
    if len(test_pairs) == 0:
        logger.warning(
            "Не найдены тестовые данные. Проверьте %s/test и файлы CSV.", DATASET_DIR
        )

    # Создание тренировочного и тестового датасетов
    train_dataset = create_dataset(train_pairs, "train", args.seq)
    test_dataset = create_dataset(test_pairs, "test", args.seq)

    train_size = tf.data.experimental.cardinality(train_dataset).numpy()
    test_size = tf.data.experimental.cardinality(test_dataset).numpy()
    logger.info("Количество тренировочных батчей: %d", train_size)
    logger.info("Количество тестовых батчей: %d", test_size)

    if train_size <= 0:
        logger.error(
            "Тренировочный датасет пуст или имеет недопустимую размерность. Проверьте CSV-файлы в %s/train.",
            DATASET_DIR,
        )
        raise ValueError(
            "Empty or invalid training dataset. Check CSV files in %s/train."
            % DATASET_DIR
        )

    # Инициализация модели
    model = get_model(args.model_name, seq_len=args.seq)
    model.summary(print_fn=lambda x: logger.info(x))

    # Возобновление обучения
    initial_epoch = 0
    if args.resume:
        checkpoint_files = glob.glob(
            os.path.join(model_save_dir, f"{args.model_name}/{args.model_name}_*.keras")
        )
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            logger.info("Возобновление обучения с %s", latest_checkpoint)
            model = tf.keras.models.load_model(
                latest_checkpoint, custom_objects={"custom_loss": custom_loss}
            )
            epoch_str = (
                os.path.basename(latest_checkpoint).split("_")[-1].replace(".keras", "")
            )
            initial_epoch = int(epoch_str) if epoch_str.isdigit() else 0
        else:
            logger.warning(
                "Контрольные точки для %s в %s не найдены, обучение начинается с нуля.",
                args.model_name,
                model_save_dir,
            )

    # Настройка расписания скорости обучения
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=max(train_size * 2, 1),  # Ensure decay_steps is positive
        decay_rate=0.9,
    )

    # Компиляция модели
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["mae"])
    logger.info(
        "Модель скомпилирована с optimizer=Adam(lr_schedule), loss=custom_loss, metrics=['mae']"
    )

    # Настройка пути сохранения модели
    filepath = os.path.join(
        model_save_dir, f"{args.model_name}/{args.model_name}_{{epoch:02d}}.keras"
    )

    # Коллбэк для логирования скорости обучения
    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self, lr_schedule, train_size):
            super().__init__()
            self.lr_schedule = lr_schedule
            self.train_size = train_size

        def on_epoch_begin(self, epoch, logs=None):
            current_step = epoch * self.train_size
            current_lr = self.lr_schedule(current_step).numpy()
            logger.info(f"Эпоха {epoch + 1}: Скорость обучения = {current_lr}")

    # Настройка коллбэков
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_best_only=True, monitor="val_loss", mode="min"
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs", args.model_name)
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, monitor="val_loss", restore_best_weights=True
        ),
        LearningRateLogger(lr_schedule, train_size),
    ]
    logger.info(
        "Коллбэки настроены: ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateLogger"
    )

    # Логирование примера данных
    try:
        for coor_pred, inpaint, coor in train_dataset.take(1):
            logger.info("coor_pred shape: %s", coor_pred.shape)
            logger.info("inpaint shape: %s", inpaint.shape)
            logger.info("coor shape: %s", coor.shape)
    except Exception as e:
        logger.error(f"Failed to iterate over train_dataset: {e}")
        raise ValueError(
            "Cannot iterate over train_dataset. Check data pipeline and CSV files."
        )

    # Запуск обучения
    logger.info("Запуск обучения...")
    model.fit(
        train_dataset.map(lambda x, y, z: ((x, y), z)),  # (inputs, targets)
        validation_data=(
            test_dataset.map(lambda x, y, z: ((x, y), z)) if test_size > 0 else None
        ),
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )
    logger.info("Обучение завершено")

    # Сохранение финальной модели
    final_model_path = os.path.join(model_save_dir, f"{args.model_name}_final.keras")
    model.save(final_model_path)
    logger.info(f"Финальная модель сохранена в {final_model_path}")


if __name__ == "__main__":
    main()
