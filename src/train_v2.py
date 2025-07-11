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
from constants import HEIGHT, WIDTH, SIGMA, DATASET_DIR, IMG_FORMAT
from utils import (
    create_heatmap,
    custom_loss,
    limit_gpu_memory,
    OutcomeMetricsCallback,
    VisualizationCallback,
)
from utils import get_video_and_csv_pairs, load_data
import multiprocessing as mp
from multiprocessing import Manager, Barrier

# Parameters
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH  # 512
BATCH_SIZE = 4  # Reduced for stability
MAG = 1.0  # Magnitude for heatmap
RATIO = 1.0  # Scaling factor for coordinates
MODEL_DIR = "models"  # Directory for model saving


def setup_logging(debug=False, process_id=0):
    """
    Configure logging with specified level based on debug flag and process ID.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format=f"[Process {process_id}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return logger


def get_model(model_name, height, width, seq, grayscale=False):
    """
    Retrieve an instance of a TrackNet model based on the specified model name.
    For grayscale mode with seq=9, set out_dim=9.
    """
    in_dim = seq if grayscale else seq * 3
    out_dim = (
        seq  # For grayscale, out_dim=seq (e.g., 9 for seq=9); for RGB, out_dim=seq
    )
    from model.VballNetV1 import VballNetV1

    logger = logging.getLogger(__name__)
    logger.info(
        f"Creating model {model_name} with in_dim={in_dim}, out_dim={out_dim}, grayscale={grayscale}"
    )
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
        frames_mixed.shape,
        heatmaps_mixed.shape,
    )
    return frames_mixed, heatmaps_mixed


def augment_sequence(frames, heatmaps, seq, grayscale=False, alpha=-1.0):
    """
    Apply data augmentation to frames and heatmaps using TensorFlow.
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
        frames = tf.transpose(frames, [1, 2, 0])
        heatmaps = tf.transpose(heatmaps, [1, 2, 0])
        combined = tf.concat([frames, heatmaps], axis=2)
        combined = tf.image.random_flip_left_right(combined, seed=None)
        logger.debug("After flip: combined shape %s", combined.shape)
        frames = combined[:, :, : seq * (1 if grayscale else 3)]
        heatmaps = combined[:, :, seq * (1 if grayscale else 3) :]
        frames = tf.transpose(frames, [2, 0, 1])
        heatmaps = tf.transpose(heatmaps, [2, 0, 1])
        frames = tf.ensure_shape(frames, [seq * (1 if grayscale else 3), 288, 512])
        heatmaps = tf.ensure_shape(heatmaps, [seq, 288, 512])
        logger.debug(
            "After geometric augmentations: frames %s, heatmaps %s",
            frames.shape,
            heatmaps.shape,
        )
        return frames, heatmaps
    except Exception as e:
        logger.error("Error in augment_sequence: %s", str(e))
        raise


def synchronize_weights(models, shared_weights, process_id, barrier):
    """
    Synchronize weights between two models by averaging.
    """
    logger = logging.getLogger(__name__)
    weights = models[process_id].get_weights()
    shared_weights[process_id] = weights
    barrier.wait()  # Wait for both processes to save weights
    if process_id == 0:
        avg_weights = []
        for w1, w2 in zip(shared_weights[0], shared_weights[1]):
            avg_w = (w1 + w2) / 2.0
            avg_weights.append(avg_w)
        shared_weights[0] = avg_weights
        shared_weights[1] = avg_weights
    barrier.wait()  # Ensure both processes wait until averaging is done
    models[process_id].set_weights(shared_weights[process_id])
    logger.info(f"Process {process_id}: Weights synchronized")


def train_process(process_id, args, train_pairs, test_pairs, shared_weights, barrier):
    """
    Training function for a single process.
    """
    logger = setup_logging(args.debug, process_id)
    limit_gpu_memory(args.gpu_memory_limit)
    if args.seq < 1:
        logger.error("Sequence length must be at least 1, got %d", args.seq)
        raise ValueError(f"Invalid sequence length: {args.seq}")

    # Split training pairs for this process
    split_idx = len(train_pairs) // 2
    if process_id == 0:
        process_train_pairs = train_pairs[:split_idx]
    else:
        process_train_pairs = train_pairs[split_idx:]
    logger.info(
        f"Process {process_id}: Number of training pairs: %d", len(process_train_pairs)
    )

    # Create datasets
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in process_train_pairs],
                [p[1] for p in process_train_pairs],
                [p[2] for p in process_train_pairs],
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
    logger.info(f"Process {process_id}: Number of training batches: %d", train_size)

    # Initialize model
    model = get_model(
        args.model_name,
        height=IMG_HEIGHT,
        width=IMG_WIDTH,
        seq=args.seq,
        grayscale=args.grayscale,
    )
    models = {process_id: model}  # Store model for this process

    # Load checkpoint if resuming
    model_name_suffix = f"_seq{args.seq}_grayscale" if args.grayscale else ""
    model_save_name = f"{args.model_name}{model_name_suffix}"
    model_save_dir = os.path.join(MODEL_DIR, model_save_name)
    os.makedirs(model_save_dir, exist_ok=True)

    initial_epoch = 0
    if args.resume:
        checkpoint_files = glob.glob(
            os.path.join(model_save_dir, f"{model_save_name}_*.keras")
        )
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            logger.info(f"Process {process_id}: Resuming from %s", latest_checkpoint)
            model = tf.keras.models.load_model(
                latest_checkpoint, custom_objects={"custom_loss": custom_loss}
            )
            epoch_str = (
                os.path.basename(latest_checkpoint).split("_")[-1].replace(".keras", "")
            )
            initial_epoch = int(epoch_str) if epoch_str.isdigit() else 0
        else:
            logger.warning(
                f"Process {process_id}: No checkpoints found, starting from scratch."
            )

    # Compile model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=train_size * 2, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["mae"])
    logger.info(f"Process {process_id}: Model compiled with Adam, custom_loss, mae")

    # Callbacks
    filepath = os.path.join(
        model_save_dir, f"{model_save_name}_{process_id}_{{epoch:02d}}.keras"
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_best_only=False, monitor="val_loss"
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                MODEL_DIR, "logs", f"{model_save_name}_process_{process_id}"
            )
        ),
        tf.keras.callbacks.EarlyStopping(patience=30, monitor="val_loss"),
    ]

    # Training loop with synchronization
    for epoch in range(initial_epoch, args.epochs):
        logger.info(f"Process {process_id}: Starting epoch {epoch + 1}")
        model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=1,
            initial_epoch=0,
            callbacks=callbacks,
            verbose=1 if process_id == 0 else 0,
        )
        logger.info(f"Process {process_id}: Epoch {epoch + 1} completed")
        synchronize_weights(models, shared_weights, process_id, barrier)

    logger.info(f"Process {process_id}: Training completed")


def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(
        description="Train VballNet model with parallel processes."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--seq", type=int, default=9, help="Number of frames in sequence."
    )
    parser.add_argument(
        "--grayscale", action="store_true", help="Use grayscale frames."
    )
    parser.add_argument(
        "--model_name", type=str, default="VballNetV1", help="Model name to train."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train."
    )
    parser.add_argument(
        "--alpha", type=float, default=-1.0, help="Alpha for mixup augmentation."
    )
    parser.add_argument(
        "--gpu_memory_limit", type=int, default=-1, help="Limit GPU memory usage in MB."
    )
    args = parser.parse_args()

    logger.info(
        "Starting parallel training with seq=%d, grayscale=%s, debug=%s, resume=%s, model_name=%s, alpha=%s, gpu_memory_limit=%d",
        args.seq,
        args.grayscale,
        args.debug,
        args.resume,
        args.model_name,
        args.alpha,
        args.gpu_memory_limit,
    )

    if args.model_name not in ["VballNetV1"]:
        logger.error("Invalid model name: %s. Must be 'VballNetV1'", args.model_name)
        raise ValueError(f"Invalid model name: {args.model_name}")

    # Load data
    train_pairs = get_video_and_csv_pairs("train", args.seq)
    test_pairs = get_video_and_csv_pairs("test", args.seq)
    logger.info("Total training pairs: %d", len(train_pairs))
    logger.info("Total test pairs: %d", len(test_pairs))

    if len(train_pairs) == 0:
        logger.error("No training data found.")
        raise ValueError("No training data found.")

    # Initialize multiprocessing
    manager = Manager()
    shared_weights = manager.dict()
    barrier = Barrier(2)  # Barrier for 2 processes
    processes = []

    # Start two processes
    for process_id in range(2):
        p = mp.Process(
            target=train_process,
            args=(process_id, args, train_pairs, test_pairs, shared_weights, barrier),
        )
        processes.append(p)
        p.start()

    # Wait for processes to complete
    for p in processes:
        p.join()

    logger.info("Parallel training completed")


if __name__ == "__main__":
    main()
