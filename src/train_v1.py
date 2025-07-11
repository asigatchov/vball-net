import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import cv2
import os
import argparse
import logging
from datetime import datetime
import glob
from constants import HEIGHT, WIDTH, SIGMA, DATASET_DIR , IMG_FORMAT
from utils import create_heatmap, custom_loss, limit_gpu_memory, OutcomeMetricsCallback, VisualizationCallback

from utils import get_video_and_csv_pairs, load_data

# Parameters
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH  # 512
BATCH_SIZE = 4  # Reduced for stability
MAG = 1.0  # Magnitude for heatmap
RATIO = 1.0  # Scaling factor for coordinates
MODEL_DIR = "models"  # Directory for model saving


def setup_logging(debug=False):
    """
    Configure logging with specified level based on debug flag.
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

# Import get_model function
def get_model(model_name, height, width, seq, grayscale=False):
    """
    Retrieve an instance of a TrackNet model based on the specified model name.
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
        return TrackNetV4(height, width, 'TypeB')
    return VballNetV1(height, width, in_dim=in_dim, out_dim=out_dim)

def reshape_tensors(frames, heatmaps, seq, grayscale=False):
    """
    Reshape tensors to (channels, height, width) for channels_first.
    """
    logger = logging.getLogger(__name__)
    frames = tf.ensure_shape(
        frames, [IMG_HEIGHT, IMG_WIDTH, seq * (1 if grayscale else 3)]
    )
    heatmaps = tf.ensure_shape(heatmaps, [IMG_HEIGHT, IMG_WIDTH, seq])
    frames = tf.transpose(frames, [2, 0, 1])  # (seq*(1 or 3), 288, 512)
    heatmaps = tf.transpose(heatmaps, [2, 0, 1])  # (seq, 288, 512)
    logger.debug(
        "Reshaped tensors: frames %s, heatmaps %s", frames.shape, heatmaps.shape
    )
    return frames, heatmaps


def mixup(frames, heatmaps, alpha=0.5):
    """
    Apply mixup augmentation to frames and heatmaps.

    Args:
        frames (tf.Tensor): Input frames tensor (batch_size, channels, height, width)
        heatmaps (tf.Tensor): Target heatmaps tensor (batch_size, seq, height, width)
        alpha (float): Alpha parameter for beta distribution

    Returns:
        frames_mixed (tf.Tensor): Mixed frames tensor
        heatmaps_mixed (tf.Tensor): Mixed heatmaps tensor
    """
    logger = logging.getLogger(__name__)
    batch_size = tf.shape(frames)[0]

    # Use tf.random.gamma to sample from Beta(alpha, alpha)
    gamma1 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    gamma2 = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lamb = gamma1 / (gamma1 + gamma2)
    lamb = tf.maximum(lamb, 1.0 - lamb)  # Ensure lambda >= 0.5
    lamb = tf.reshape(lamb, [batch_size, 1, 1, 1])  # Shape: (batch_size, 1, 1, 1)

    # Randomly permute batch indices
    indices = tf.random.shuffle(tf.range(batch_size))

    # Mix frames and heatmaps
    frames_mixed = frames * lamb + tf.gather(frames, indices) * (1.0 - lamb)
    heatmaps_mixed = heatmaps * lamb + tf.gather(heatmaps, indices) * (1.0 - lamb)

    logger.debug(
        "Applied mixup: frames_mixed shape %s, heatmaps_mixed shape %s",
        frames_mixed.shape,
        heatmaps_mixed.shape,
    )
    return frames_mixed, heatmaps_mixed


def augment_sequence(frames, heatmaps, seq, grayscale=False, alpha=-1.0):
    """
    Apply data augmentation to frames and heatmaps using TensorFlow, including mixup if alpha > 0.
    """
    logger = logging.getLogger(__name__)
    try:
        tf.debugging.assert_shapes(
            [
                (frames, (seq * (1 if grayscale else 3), 288, 512)),
                (heatmaps, (seq, 288, 512)),
            ]
        )
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

        # Transpose to (height, width, channels)
        frames = tf.transpose(frames, [1, 2, 0])  # (288, 512, seq*(1 or 3))
        heatmaps = tf.transpose(heatmaps, [1, 2, 0])  # (288, 512, seq)

        # Concatenate along channel axis
        combined = tf.concat([frames, heatmaps], axis=2)  # (288, 512, seq*(1 or 3)+seq)

        # Apply left-right flip with 50% probability
        combined = tf.image.random_flip_left_right(combined, seed=None)
        logger.debug("After flip: combined shape %s", combined.shape)

        # Split back to framesfera and heatmaps
        frames = combined[
            :, :, : seq * (1 if grayscale else 3)
        ]  # (288, 512, seq*(1 or 3))
        heatmaps = combined[:, :, seq * (1 if grayscale else 3) :]  # (288, 512, seq)

        # Transpose back to (channels, height, width)
        frames = tf.transpose(frames, [2, 0, 1])  # (seq*(1 or 3), 288, 512)
        heatmaps = tf.transpose(heatmaps, [2, 0, 1])  # (seq, 288, 512)

        frames = tf.ensure_shape(frames, [seq * (1 if grayscale else 3), 288, 512])
        heatmaps = tf.ensure_shape(heatmaps, [seq, 288, 512])
        logger.debug(
            "After geometric augmentations: frames %s, heatmaps %s",
            frames.shape,
            heatmaps.shape,
        )

        # Apply mixup if alpha > 0 (batch-level operation, handled outside this function)
        return frames, heatmaps

    except Exception as e:
        logger.error("Error in augment_sequence: %s", str(e))
        logger.error("Frames shape: %s", frames.shape if frames is not None else "None")
        logger.error(
            "Heatmaps shape: %s", heatmaps.shape if heatmaps is not None else "None"
        )
        raise


def main():


    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train VballNet model.")
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
        help="Model name to train (VballNetFastV1 or VballNetV1).",
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

    args = parser.parse_args()

    logger.info(
        "Starting training script with seq=%d, grayscale=%s, debug=%s, resume=%s, model_name=%s, alpha=%s gpu_memory_limit=%d",
        args.seq,
        args.grayscale,
        args.debug,
        args.resume,
        args.model_name,
        args.alpha,
        args.gpu_memory_limit,
    )


    limit_gpu_memory(args.gpu_memory_limit)
    if args.seq < 1:
        logger.error("Sequence length must be at least 1, got %d", args.seq)
        raise ValueError(f"Invalid sequence length: {args.seq}")

    if args.model_name not in ["VballNetFastV1", "VballNetV1", "PlayerNetFastV1", "TrackNetV4"]:
        logger.error(
            "Invalid model name: %s. Must be 'VballNetFastV1' or 'VballNetV1'",
            args.model_name,
        )
        raise ValueError(f"Invalid model name: {args.model_name}")

    model_name_suffix = (
        f"_seq{args.seq}_grayscale" if args.grayscale and args.seq == 9 else ""
    )
    model_save_name = f"{args.model_name}{model_name_suffix}"
    model_save_dir = os.path.join(MODEL_DIR, model_save_name)
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info("Created model save directory: %s", model_save_dir)

    train_pairs = get_video_and_csv_pairs("train", args.seq)
    test_pairs = get_video_and_csv_pairs("test", args.seq)

    logger.info("Number of training pairs: %d", len(train_pairs))
    logger.info("Number of test pairs: %d", len(test_pairs))
    if len(train_pairs) == 0:
        logger.error(
            "No training data found. Check DATASET_DIR/train and frame/CSV files."
        )
        raise ValueError(
            "No training data found. Check DATASET_DIR/train and frame/CSV files."
        )
    if len(test_pairs) == 0:
        logger.warning(
            "No test data found. Check DATASET_DIR/test and frame/CSV files."
        )

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

    # Apply mixup after batching if alpha > 0
    if args.alpha > 0:
        train_dataset = train_dataset.map(
            lambda frames, heatmaps: mixup(frames, heatmaps, args.alpha),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in test_pairs],
                [p[1] for p in test_pairs],
                [p[2] for p in test_pairs],
            )
        )
        .map(
            lambda t, c, f: tf.py_function(
                func=lambda x, y, z: load_data(
                    x, y, z, "test", args.seq, args.grayscale
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
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_size = tf.data.experimental.cardinality(train_dataset).numpy()
    test_size = tf.data.experimental.cardinality(test_dataset).numpy()
    logger.info("Number of training batches: %d", train_size)
    logger.info("Number of test batches: %d", test_size)

    model = get_model(
        args.model_name,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        seq=args.seq,
        grayscale=args.grayscale,
    )
    model.summary(print_fn=lambda x: logger.info(x))

    initial_epoch = 0
    if args.resume:
        checkpoint_files = glob.glob(
            os.path.join(model_save_dir, f"{model_save_name}/{model_save_name}_*.keras")
        )
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            logger.info("Resuming training from %s", latest_checkpoint)
            model = tf.keras.models.load_model(
                latest_checkpoint, custom_objects={"custom_loss": custom_loss}
            )
            epoch_str = (
                os.path.basename(latest_checkpoint).split("_")[-1].replace(".keras", "")
            )
            initial_epoch = int(epoch_str) if epoch_str.isdigit() else 0
        else:
            logger.warning(
                "No checkpoints found for %s in %s, starting training from scratch.",
                model_save_name,
                model_save_dir,
            )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=train_size * 2, decay_rate=0.9
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["mae"])
    logger.info(
        "Model compiled with optimizer=Adam(lr_schedule), loss=custom_loss, metrics=['mae']"
    )

    filepath = os.path.join(
        model_save_dir, f"{model_save_name}/{model_save_name}_{{epoch:02d}}.keras"
    )

    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self, lr_schedule, train_size):
            super().__init__()
            self.lr_schedule = lr_schedule
            self.train_size = train_size

        def on_epoch_begin(self, epoch, logs=None):
            current_step = epoch * self.train_size
            current_lr = self.lr_schedule(current_step).numpy()
            logger.info(f"Epoch {epoch + 1}: Learning rate = {current_lr}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(filepath), save_best_only=False, monitor="val_loss"
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs", model_save_name)
        ),
        tf.keras.callbacks.EarlyStopping(patience=30, monitor="val_loss"),
        # OutcomeMetricsCallback(
        #     validation_data=test_dataset,
        #     tol=10,
        #     log_dir=os.path.join(MODEL_DIR, "logs", f"{model_save_name}/outcome"),
        # ),
        # LearningRateLogger(lr_schedule, train_size),
        # VisualizationCallback(
        #     test_dataset=test_dataset,
        #     save_dir=os.path.join(MODEL_DIR, "visualizations"),
        #     seq=args.seq,
        #     grayscale=args.grayscale,
        # ),
    ]

    logger.info(
        "Callbacks configured: ModelCheckpoint, TensorBoard, EarlyStopping, OutcomeMetricsCallback, LearningRateLogger, VisualizationCallback"
    )

    for frames, heatmaps in train_dataset.take(1):
        logger.info("Frames shape: %s", frames.shape)
        logger.info("Heatmaps shape: %s", heatmaps.shape)
        frames = tf.transpose(frames[0], [1, 2, 0])
        heatmaps = tf.transpose(heatmaps[0], [1, 2, 0])
        if args.grayscale:
            tf.io.write_file(
                "augmented_frame.png",
                tf.image.encode_png(
                    tf.cast(tf.image.grayscale_to_rgb(frames[:, :, :1]) * 255, tf.uint8)
                ),
            )
        else:
            tf.io.write_file(
                "augmented_frame.png",
                tf.image.encode_png(tf.cast(frames[:, :, :3] * 255, tf.uint8)),
            )
        tf.io.write_file(
            "augmented_heatmap.png",
            tf.image.encode_png(tf.cast(heatmaps[:, :, 0:1] * 255, tf.uint8)),
        )

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
