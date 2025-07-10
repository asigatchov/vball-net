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
from constants import HEIGHT, WIDTH, SIGMA
from utils import create_heatmap, custom_loss, OutcomeMetricsCallback

def limit_gpu_memory(memory_limit_mb):
    """Limit GPU memory usage for the current TensorFlow process."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
            print(f"Set GPU memory limit to {memory_limit_mb} MB")
        except RuntimeError as e:
            print(f"Error setting GPU memory limit: {e}")

def get_model(model_name, height, width, seq, grayscale=False):
    """
    Retrieve an instance of a TrackNet model based on the specified model name.
    """
    from model.VballNetFastV1 import VballNetFastV1
    from model.VballNetV1 import VballNetV1

    if model_name == "PlayerNetFastV1":
        from model.PlayerNetFastV1 import PlayerNetFastV1
        return PlayerNetFastV1(input_shape=(9, height, width), output_channels=3)

    if model_name == "VballNetFastV1":
        in_dim = seq if grayscale else 9
        out_dim = seq if grayscale else 3
        return VballNetFastV1(height, width, in_dim=in_dim, out_dim=out_dim)

    if model_name == "TrackNetV4":
        from model.TrackNetV4 import TrackNetV4 
        in_dim = seq if grayscale else 9
        out_dim = seq if grayscale else 3
        return TrackNetV4(height, width, 'TypeB')

    in_dim = seq if grayscale else 9
    out_dim = seq if grayscale else 3
    return VballNetV1(height, width, in_dim=in_dim, out_dim=out_dim)

# Parameters
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH  # 512
BATCH_SIZE = 32  # Use 16 for grayscale
DATASET_DIR = "./data/frames"
TFRECORD_DIR = "./data/tfrecords"
MAG = 1.0
RATIO = 1.0
MODEL_DIR = "models"

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

def parse_tfrecord(example_proto, grayscale=False):
    """
    Parse a single TFRecord example.
    """
    feature_description = {
        'track_id': tf.io.FixedLenFeature([], tf.string),
        'frame_idx': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'x': tf.io.FixedLenFeature([], tf.int64),
        'y': tf.io.FixedLenFeature([], tf.int64),
        'visibility': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    frame = tf.image.decode_png(example['image'], channels=1 if grayscale else 3)
    frame = tf.cast(frame, tf.float32) / 255.0
    x = tf.cast(example['x'], tf.float32)
    y = tf.cast(example['y'], tf.float32)
    heatmap = create_heatmap(
        x / RATIO,
        y / RATIO,
        example['visibility'],
        IMG_HEIGHT,
        IMG_WIDTH,
        SIGMA,
        MAG
    )
    return frame, heatmap, example['track_id'], example['frame_idx']

def load_sequence(track_id, frame_indices, mode, seq, grayscale=False):
    """
    Load a sequence of frames and heatmaps from TFRecord using native TensorFlow operations.
    """
    logger = logging.getLogger(__name__)
    dataset = tf.data.TFRecordDataset(os.path.join(TFRECORD_DIR, f"{mode}.tfrecord"))
    frame_indices = tf.cast(frame_indices, tf.int64)

    # Map to parse TFRecord
    dataset = dataset.map(
        lambda x: parse_tfrecord(x, grayscale),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Filter by track_id and frame_indices
    dataset = dataset.filter(
        lambda frame, heatmap, tid, fid: tf.logical_and(
            tf.equal(tid, track_id),
            tf.reduce_any(tf.equal(fid, frame_indices))
        )
    )

    # Ensure we get exactly 'seq' frames
    dataset = dataset.take(seq)

    # Collect frames and heatmaps
    def reduce_fn(state, element):
        frames, heatmaps = state
        frame, heatmap, _, _ = element
        frames = frames.write([frame])
        heatmaps = heatmaps.write([heatmap])
        return frames, heatmaps

    # Initialize accumulators
    frames_acc = tf.TensorArray(dtype=tf.float32, size=seq, dynamic_size=False)
    heatmaps_acc = tf.TensorArray(dtype=tf.float32, size=seq, dynamic_size=False)

    # Reduce dataset to collect frames and heatmaps
    frames, heatmaps = dataset.reduce(
        (frames_acc, heatmaps_acc),
        reduce_fn
    )

    # Concatenate frames and heatmaps
    try:
        frames = frames.concat()
        heatmaps = heatmaps.concat()
        frames = tf.reshape(frames, [seq, IMG_HEIGHT, IMG_WIDTH, 1 if grayscale else 3])
        heatmaps = tf.reshape(heatmaps, [seq, IMG_HEIGHT, IMG_WIDTH, 1])
        frames = tf.transpose(frames, [1, 2, 3, 0])  # [H, W, C, seq] -> [H, W, seq*C]
        heatmaps = tf.transpose(heatmaps, [1, 2, 3, 0])  # [H, W, 1, seq] -> [H, W, seq]
        frames = tf.reshape(frames, [IMG_HEIGHT, IMG_WIDTH, seq * (1 if grayscale else 3)])
        heatmaps = tf.reshape(heatmaps, [IMG_HEIGHT, IMG_WIDTH, seq])
        logger.debug(
            "Loaded sequence for track_id %s, frames shape %s, heatmaps shape %s",
            track_id,
            frames.shape,
            heatmaps.shape
        )
        return frames, heatmaps
    except Exception as e:
        logger.error("Failed to load sequence for track_id %s: %s", track_id, str(e))
        raise

def get_tfrecord_pairs(mode, seq):
    """
    Returns a list of (track_id, csv_path, frame_indices) tuples for TFRecord data.
    """
    logger = logging.getLogger(__name__)
    pairs = []
    mode_dir = os.path.join(DATASET_DIR, mode)
    if not os.path.exists(mode_dir):
        logger.warning("Directory %s does not exist", mode_dir)
        return pairs

    video_dirs = [
        d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))
    ]
    for track_id in video_dirs:
        csv_path = os.path.join(mode_dir, f"{track_id}_ball.csv")
        if not os.path.exists(csv_path):
            logger.warning(
                "CSV not found for track_id %s at %s, skipping", track_id, csv_path
            )
            continue
        df = pd.read_csv(
            csv_path,
            dtype={
                "Frame": np.int64,
                "X": np.int64,
                "Y": np.int64,
                "Visibility": np.int64,
            },
        )
        df = df.fillna({"X": 0, "Y": 0, "Visibility": 0})

        if not np.all(df["Frame"].values == np.arange(len(df))):
            logger.warning("Non-sequential frame indices in %s, skipping", csv_path)
            continue

        num_frames = len(df)
        if num_frames < seq:
            logger.warning(
                "%s has %d frames, need at least %d, skipping",
                csv_path,
                num_frames,
                seq,
            )
            continue
        for t in range(seq - 1, num_frames):
            frame_indices = list(range(t - seq + 1, t + 1))
            if max(frame_indices) >= len(df):
                logger.warning(
                    "Frame indices %s exceed CSV length %d for %s, skipping",
                    frame_indices,
                    len(df),
                    csv_path,
                )
                continue
            pairs.append((track_id, csv_path, frame_indices))
    logger.debug("Found %d valid pairs for mode %s", len(pairs), mode)
    return pairs

def reshape_tensors(frames, heatmaps, seq, grayscale=False):
    """
    Reshape tensors to (channels, height, width) for channels_first.
    """
    logger = logging.getLogger(__name__)
    frames = tf.ensure_shape(
        frames, [IMG_HEIGHT, IMG_WIDTH, seq * (1 if grayscale else 3)]
    )
    heatmaps = tf.ensure_shape(heatmaps, [IMG_HEIGHT, IMG_WIDTH, seq])
    frames = tf.transpose(frames, [2, 0, 1])
    heatmaps = tf.transpose(heatmaps, [2, 0, 1])
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
    lamb = tf.random.stateless_beta([alpha, alpha], shape=[batch_size], seed=(123, 456))
    lamb = tf.maximum(lamb, 1.0 - lamb)
    lamb = tf.expand_dims(
        tf.expand_dims(tf.expand_dims(lamb, -1), -1), -1
    )
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

        frames = tf.transpose(frames, [1, 2, 0])
        heatmaps = tf.transpose(heatmaps, [1, 2, 0])
        combined = tf.concat([frames, heatmaps], axis=2)
        combined = tf.image.random_flip_left_right(combined, seed=None)
        angle = tf.random.uniform([], minval=-20, maxval=20, dtype=tf.float32) * (
            3.14159 / 180
        )
        combined = tf.image.rot90(
            combined, k=tf.cast(tf.round(angle / (3.14159 / 2)), tf.int32)
        )
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

def main():
    from tensorflow.keras.mixed_precision import Policy, set_global_policy

    # Enable mixed precision
    policy = Policy('mixed_float16')
    set_global_policy(policy)
    logger = setup_logging()
    logger.info("Mixed precision enabled with policy: mixed_float16")

    # Enable XLA
    tf.config.optimizer.set_jit(True)
    logger.info("XLA enabled")

    limit_gpu_memory(20480)
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
    args = parser.parse_args()

    setup_logging(args.debug)
    logger.info(
        "Starting training script with seq=%d, grayscale=%s, debug=%s, resume=%s, model_name=%s, alpha=%s",
        args.seq,
        args.grayscale,
        args.debug,
        args.resume,
        args.model_name,
        args.alpha,
    )

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
    os.makedirs(TFRECORD_DIR, exist_ok=True)
    logger.info("Created model save directory: %s", model_save_dir)

    train_pairs = get_tfrecord_pairs("train", args.seq)
    test_pairs = get_tfrecord_pairs("test", args.seq)

    logger.info("Number of training pairs: %d", len(train_pairs))
    logger.info("Number of test pairs: %d", len(test_pairs))
    if len(train_pairs) == 0:
        logger.error(
            "No training data found. Check DATASET_DIR/train and TFRecord files."
        )
        raise ValueError(
            "No training data found. Check DATASET_DIR/train and TFRecord files."
        )
    if len(test_pairs) == 0:
        logger.warning(
            "No test data found. Check DATASET_DIR/test and TFRecord files."
        )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in train_pairs],
                [p[1] for p in train_pairs],
                [p[2] for p in train_pairs],
            )
        )
        .shuffle(1000)
        .map(
            lambda t, c, f: load_sequence(t, f, "train", args.seq, args.grayscale),
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
        .cache()
        .batch(BATCH_SIZE)
        .map(
            lambda frames, heatmaps: mixup(frames, heatmaps, args.alpha) if args.alpha > 0 else (frames, heatmaps),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in test_pairs],
                [p[1] for p in test_pairs],
                [p[2] for p in test_pairs],
            )
        )
        .map(
            lambda t, c, f: load_sequence(t, f, "test", args.seq, args.grayscale),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            lambda frames, heatmaps: reshape_tensors(
                frames, heatmaps, args.seq, args.grayscale
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .cache()
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

    class VisualizationCallback(tf.keras.callbacks.Callback):
        def __init__(
            self,
            test_dataset,
            save_dir="visualizations",
            seq=3,
            buffer_size=1,
            grayscale=False,
        ):
            super().__init__()
            self.test_dataset = test_dataset
            self.save_dir = save_dir
            self.seq = seq
            self.buffer_size = buffer_size
            self.grayscale = grayscale
            os.makedirs(self.save_dir, exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            logger = logging.getLogger(__name__)
            for frames, heatmaps in self.test_dataset.shuffle(self.buffer_size):
                frames = tf.transpose(frames, [0, 2, 3, 1])
                pred_heatmaps = self.model.predict(frames, verbose=0)
                frames_np = frames[0].numpy()
                heatmaps_np = heatmaps[0].numpy()
                pred_heatmaps_np = pred_heatmaps[0]

                for i in range(self.seq):
                    if self.grayscale:
                        frame = frames_np[:, :, i:i+1]
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        frame = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
                        frame = tf.ensure_shape(frame, [288, 512, 3])
                    else:
                        frame = frames_np[:, :, i*3:(i+1)*3]
                        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
                        frame = tf.ensure_shape(frame, [288, 512, 3])

                    try:
                        true_heatmap = heatmaps_np[i, :, :]
                        pred_heatmap = pred_heatmaps_np[i, :, :]
                        frame = tf.cast(frame * 255, tf.uint8)
                        true_heatmap = tf.cast(true_heatmap * 255, tf.uint8)
                        pred_heatmap = tf.cast(pred_heatmap * 255, tf.uint8)
                        true_heatmap = tf.expand_dims(true_heatmap, axis=-1)
                        pred_heatmap = tf.expand_dims(pred_heatmap, axis=-1)
                        true_heatmap = tf.image.grayscale_to_rgb(true_heatmap)
                        pred_heatmap = tf.image.grayscale_to_rgb(pred_heatmap)
                        combined = tf.concat([frame, true_heatmap, pred_heatmap], axis=1)
                        tf.io.write_file(
                            os.path.join(
                                self.save_dir, f"vis_epoch_{epoch:03d}_frame_{i}.png"
                            ),
                            tf.image.encode_png(combined),
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to save visualization for epoch %d, frame %d: %s",
                            epoch,
                            i,
                            str(e),
                        )
                        continue

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(filepath), save_best_only=False, monitor="val_loss"
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs", model_save_name)
        ),
        tf.keras.callbacks.EarlyStopping(patience=30, monitor="val_loss"),
        LearningRateLogger(lr_schedule, train_size),
        VisualizationCallback(
            test_dataset=test_dataset,
            save_dir=os.path.join(MODEL_DIR, "visualizations"),
            seq=args.seq,
            grayscale=args.grayscale,
        ),
    ]

    logger.info(
        "Callbacks configured: ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateLogger, VisualizationCallback"
    )

    tf.profiler.experimental.start(os.path.join(MODEL_DIR, "logs", model_save_name, "profiler"))

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

    tf.profiler.experimental.stop()

if __name__ == "__main__":
    main()