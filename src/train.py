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
from constants import HEIGHT, WIDTH
from utils import create_heatmap, custom_loss, OutcomeMetricsCallback

# Import get_model function
def get_model(model_name, height, width):
    """
    Retrieve an instance of a TrackNet model based on the specified model name.
    """
    from model.VballNetFastV1 import VballNetFastV1
    from model.VballNetV1 import VballNetV1

    if model_name == 'PlayerNetFastV1':
        from model.PlayerNetFastV1 import PlayerNetFastV1
        return PlayerNetFastV1(input_shape=(9, height, width), output_channels=3)

    if model_name == "VballNetFastV1":
        return VballNetFastV1(height, width, in_dim=9, out_dim=3)
    return VballNetV1(height, width, in_dim=9, out_dim=3)

# Parameters
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH  # 512
IMG_FORMAT = ".png"
BATCH_SIZE = 1  # Reduced for stability
DATASET_DIR = "/home/gled/frames"  # Base directory for frames
SIGMA = 5  # Radius for circular heatmap
MAG = 1.0  # Magnitude for heatmap
RATIO = 1.0  # Scaling factor for coordinates
MODEL_DIR = "models"  # Directory for model saving

# Configure logging
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
def load_image_frames(track_id, frame_indices, mode, height=288, width=512):
    """
    Loads seq preprocessed image frames from DATASET_DIR/mode/track_id/.
    """
    logger = logging.getLogger(__name__)
    frames = []
    track_dir = os.path.join(DATASET_DIR, mode, str(track_id))
    for idx in frame_indices:
        frame_path = os.path.join(track_dir, f"{idx}{IMG_FORMAT}")
        if not os.path.exists(frame_path):
            logger.error("Frame not found at %s", frame_path)
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        # Load image using TensorFlow
        try:
            image_string = tf.io.read_file(frame_path)
            frame = tf.image.decode_jpeg(image_string, channels=3)
        except Exception as e:
            logger.error("Failed to load or decode frame %s: %s", frame_path, str(e))
            raise ValueError(f"Failed to load frame {frame_path}: {e}")

        # Check image shape
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.error("Frame %s has unexpected shape %s, expected (H, W, 3)", frame_path, frame.shape)
            raise ValueError(f"Frame {frame_path} has unexpected shape {frame.shape}, expected (H, W, 3)")

        logger.debug("Loaded frame %s with shape %s", frame_path, frame.shape)

        # Convert to float32
        frame = tf.cast(frame, tf.float32)

        # Normalize to [0, 1]
        frame = frame / 255.0
        frames.append(frame)

    # Concatenate frames along channel axis
    try:
        concatenated = tf.concat(frames, axis=2)
        logger.debug("Concatenated frames shape %s for track_id %s, indices %s", concatenated.shape, track_id, frame_indices)
        return concatenated
    except Exception as e:
        logger.error("Failed to concatenate frames for track_id %s, indices %s: %s", track_id, frame_indices, str(e))
        raise ValueError(f"Failed to concatenate frames: {e}")



def get_video_and_csv_pairs(mode, seq):
    """
    Returns a list of (track_id, csv_path, frame_indices) tuples for videos in DATASET_DIR/mode.
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

        # Validate Frame column
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
            track_dir = os.path.join(mode_dir, track_id)
            missing = [
                idx
                for idx in frame_indices
                if not os.path.exists(os.path.join(track_dir, f"{idx}{IMG_FORMAT}"))
            ]
            if missing:
                logger.warning(
                    "Missing frames %s for track_id %s, skipping sequence %s",
                    missing,
                    track_id,
                    frame_indices,
                )
                continue
            pairs.append((track_id, csv_path, frame_indices))
    logger.debug("Found %d valid pairs for mode %s", len(pairs), mode)
    return pairs

def load_data(track_id, csv_path, frame_indices, mode, seq):
    """
    Loads seq frames and their heatmaps for a single sequence.
    """
    logger = logging.getLogger(__name__)
    if isinstance(track_id, tf.Tensor):
        track_id = (
            track_id.numpy().decode("utf-8")
            if track_id.dtype == tf.string
            else str(track_id.numpy())
        )
    if isinstance(csv_path, tf.Tensor):
        csv_path = (
            csv_path.numpy().decode("utf-8")
            if csv_path.dtype == tf.string
            else str(csv_path.numpy())
        )
    if isinstance(frame_indices, tf.Tensor):
        frame_indices = frame_indices.numpy().tolist()

    frames = load_image_frames(track_id, frame_indices, mode, IMG_HEIGHT, IMG_WIDTH)

    df = pd.read_csv(
        csv_path,
        dtype={"Frame": np.int64, "X": np.int64, "Y": np.int64, "Visibility": np.int64},
    )
    df = df.fillna({"X": 0, "Y": 0, "Visibility": 0})
    heatmaps = []
    for idx in frame_indices:
        if idx >= len(df):
            logger.error("Frame index %d out of range for CSV %s", idx, csv_path)
            raise IndexError(f"Frame index {idx} out of range for CSV {csv_path}")
        row = df[df["Frame"] == idx]
        if row.empty:
            logger.warning(
                "No data for frame %d in %s, setting heatmap to zero", idx, csv_path
            )
            heatmap = tf.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)
        else:
            x, y, visibility = (
                row["X"].iloc[0],
                row["Y"].iloc[0],
                row["Visibility"].iloc[0],
            )
            if not isinstance(visibility, (int, np.int64)) or visibility not in [0, 1]:
                logger.warning(
                    "Invalid visibility %s at frame %d in %s, setting to 0",
                    visibility,
                    idx,
                    csv_path,
                )
                visibility = 0
            x = x / RATIO
            y = y / RATIO
            if x < 0 or y < 0 or x >= IMG_WIDTH or y >= IMG_HEIGHT:
                if x >= IMG_WIDTH or y >= IMG_HEIGHT:
                    logger.warning(
                        "Coordinates out of bounds (x=%s, y=%s) at frame %d in %s, setting heatmap to zero",
                        x,
                        y,
                        idx,
                        csv_path,
                    )
                heatmap = tf.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)
            else:
                heatmap = create_heatmap(
                    x, y, visibility, IMG_HEIGHT, IMG_WIDTH, SIGMA, MAG
                )
        heatmaps.append(heatmap)

    heatmaps = tf.concat(heatmaps, axis=2)
    frames.set_shape([IMG_HEIGHT, IMG_WIDTH, seq * 3])
    heatmaps.set_shape([IMG_HEIGHT, IMG_WIDTH, seq])
    logger.debug(
        "Loaded data for track_id %s: frames shape %s, heatmaps shape %s",
        track_id,
        frames.shape,
        heatmaps.shape,
    )
    return frames, heatmaps

def reshape_tensors(frames, heatmaps, seq):
    """
    Reshape tensors to (channels, height, width) for channels_first.
    """
    logger = logging.getLogger(__name__)
    frames = tf.ensure_shape(frames, [IMG_HEIGHT, IMG_WIDTH, seq * 3])
    heatmaps = tf.ensure_shape(heatmaps, [IMG_HEIGHT, IMG_WIDTH, seq])
    frames = tf.transpose(frames, [2, 0, 1])  # (seq*3, 288, 512)
    heatmaps = tf.transpose(heatmaps, [2, 0, 1])  # (seq, 288, 512)
    logger.debug(
        "Reshaped tensors: frames %s, heatmaps %s", frames.shape, heatmaps.shape
    )
    return frames, heatmaps

def augment_sequence(frames, heatmaps, seq=3):
    """
    Apply data augmentation to frames and heatmaps using TensorFlow.
    """
    logger = logging.getLogger(__name__)
    try:
        # Validate input shapes
        tf.debugging.assert_shapes([(frames, (9, 288, 512)), (heatmaps, (3, 288, 512))])
        # Check for valid pixel values
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
        frames = tf.transpose(frames, [1, 2, 0])  # (288, 512, 9)
        heatmaps = tf.transpose(heatmaps, [1, 2, 0])  # (288, 512, 3)

        # Concatenate along channel axis
        combined = tf.concat([frames, heatmaps], axis=2)  # (288, 512, 12)

        # Apply left-right flip with 50% probability
        combined = tf.image.random_flip_left_right(combined, seed=None)
        logger.debug("After flip: combined shape %s", combined.shape)

        # Apply random rotation up to 7 degrees
        angle = tf.random.uniform([], minval=-20, maxval=20, dtype=tf.float32) * (3.14159 / 180)
        combined = tf.image.rot90(combined, k=tf.cast(tf.round(angle / (3.14159 / 2)), tf.int32))
        logger.debug("After rotation: combined shape %s", combined.shape)

        # Split back to frames and heatmaps
        frames = combined[:, :, :9]  # (288, 512, 9)
        heatmaps = combined[:, :, 9:]  # (288, 512, 3)

        # Transpose back to (channels, height, width)
        frames = tf.transpose(frames, [2, 0, 1])  # (9, 288, 512)
        heatmaps = tf.transpose(heatmaps, [2, 0, 1])  # (3, 288, 512)

        # Ensure output shapes
        frames = tf.ensure_shape(frames, [9, 288, 512])
        heatmaps = tf.ensure_shape(heatmaps, [3, 288, 512])
        logger.debug(
            "Final shapes: frames %s, heatmaps %s", frames.shape, heatmaps.shape
        )

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

    # Parse command-line arguments
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
        "--model_name",
        type=str,
        default="VballNetFastV1",
        help="Model name to train (VballNetFastV1 or VballNetV1).",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger.info(
        "Starting training script with seq=%d, debug=%s, resume=%s, model_name=%s",
        args.seq,
        args.debug,
        args.resume,
        args.model_name,
    )

    # Validate seq
    if args.seq < 1:
        logger.error("Sequence length must be at least 1, got %d", args.seq)
        raise ValueError(f"Invalid sequence length: {args.seq}")

    # Validate model name
    if args.model_name not in ["VballNetFastV1", "VballNetV1", "PlayerNetFastV1"]:
        logger.error(
            "Invalid model name: %s. Must be 'VballNetFastV1' or 'VballNetV1'",
            args.model_name,
        )
        raise ValueError(f"Invalid model name: {args.model_name}")

    # Create train and test datasets
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
                func=lambda x, y, z: load_data(x, y, z, "train", args.seq),
                inp=[t, c, f],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            lambda frames, heatmaps: reshape_tensors(frames, heatmaps, args.seq),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            lambda frames, heatmaps: augment_sequence(frames, heatmaps, args.seq),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(BATCH_SIZE)
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
            lambda t, c, f: tf.py_function(
                func=lambda x, y, z: load_data(x, y, z, "test", args.seq),
                inp=[t, c, f],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(
            lambda frames, heatmaps: reshape_tensors(frames, heatmaps, args.seq),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Log dataset sizes
    train_size = tf.data.experimental.cardinality(train_dataset).numpy()
    test_size = tf.data.experimental.cardinality(test_dataset).numpy()
    logger.info("Number of training batches: %d", train_size)
    logger.info("Number of test batches: %d", test_size)

    model_save_dir = os.path.join(MODEL_DIR, f"{args.model_name}")
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info("Created model save directory: %s", model_save_dir)

    # Initialize model using get_model
    model = get_model(args.model_name, height=IMG_HEIGHT, width=IMG_WIDTH)
    model.summary(print_fn=lambda x: logger.info(x))  # Redirect summary to logger

    # Load latest checkpoint if resuming
    initial_epoch = 0
    if args.resume:
        checkpoint_files = glob.glob(
            os.path.join(model_save_dir, f"{args.model_name}/{args.model_name}_*.keras")
        )
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            logger.info("Resuming training from %s", latest_checkpoint)
            model = tf.keras.models.load_model(
                latest_checkpoint, custom_objects={"custom_loss": custom_loss}
            )
            # Extract epoch from checkpoint filename (e.g., VballNetFastV1_01.keras)
            epoch_str = (
                os.path.basename(latest_checkpoint).split("_")[-1].replace(".keras", "")
            )
            initial_epoch = int(epoch_str) if epoch_str.isdigit() else 0
        else:
            logger.warning(
                "No checkpoints found for %s in %s, starting training from scratch.",
                args.model_name,
                model_save_dir,
            )

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=train_size * 2,  # Decay every two epochs
        decay_rate=0.9
    )

    # Compile model with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["mae"])
    logger.info(
        "Model compiled with optimizer=Adam(lr_schedule), loss=custom_loss, metrics=['mae']"
    )

    filepath = os.path.join(
        model_save_dir, f"{args.model_name}/{args.model_name}_{{epoch:02d}}.keras"
    )

    # Callback for logging learning rate
    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self, lr_schedule, train_size):
            super().__init__()
            self.lr_schedule = lr_schedule
            self.train_size = train_size

        def on_epoch_begin(self, epoch, logs=None):
            current_step = epoch * self.train_size
            current_lr = self.lr_schedule(current_step).numpy()
            logger.info(f"Epoch {epoch + 1}: Learning rate = {current_lr}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(filepath), save_best_only=False, monitor="val_loss"
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs", f"{args.model_name}")
        ),
        tf.keras.callbacks.EarlyStopping(patience=30, monitor="val_loss"),
        OutcomeMetricsCallback(
            validation_data=test_dataset,
            tol=10,  # Set your desired tolerance
            log_dir=os.path.join(MODEL_DIR, "logs", f"{args.model_name}/outcome"),
        ),
        LearningRateLogger(lr_schedule, train_size)
    ]

    logger.info(
        "Callbacks configured: ModelCheckpoint, TensorBoard, EarlyStopping, OutcomeMetricsCallback, LearningRateLogger"
    )

    # Log shapes for debugging
    for frames, heatmaps in train_dataset.take(1):
        logger.info(
            "Frames shape: %s", frames.shape
        )  # Expected: (batch_size, 9, 288, 512)
        logger.info(
            "Heatmaps shape: %s", heatmaps.shape
        )  # Expected: (batch_size, 3, 288, 512)
        frames = tf.transpose(frames[0], [1, 2, 0])  # (288, 512, 9)
        heatmaps = tf.transpose(heatmaps[0], [1, 2, 0])  # (288, 512, 3)
        tf.io.write_file(
            "augmented_frame.png",
            tf.image.encode_png(tf.cast(frames[:, :, :3] * 255, tf.uint8)),
        )
        tf.io.write_file(
            "augmented_heatmap.png",
            tf.image.encode_png(tf.cast(heatmaps[:, :, 0:1] * 255, tf.uint8)),
        )

    # Train model
    logger.info("Starting training...")
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=155,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )
    logger.info("Training completed")

if __name__ == "__main__":
    main()