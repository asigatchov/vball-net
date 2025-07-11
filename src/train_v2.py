import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import os
import argparse
import logging
from datetime import datetime
import glob
from constants import HEIGHT, WIDTH, SIGMA
from utils import custom_loss, OutcomeMetricsCallback

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

def limit_gpu_memory(memory_limit_mb):
    """Limit GPU memory usage for the current TensorFlow process."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.warning("No GPUs found. Running on CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
        logger.info(f"Set GPU memory limit to {memory_limit_mb} MB")
    except RuntimeError as e:
        logger.error(f"Error setting GPU memory limit: {e}")
        raise

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
        return TrackNetV4(height, width, "TypeB")

    in_dim = seq if grayscale else 9
    out_dim = seq if grayscale else 3
    return VballNetV1(height, width, in_dim=in_dim, out_dim=out_dim)

# Parameters
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH  # 512
BATCH_SIZE = 10  # Use 16 for grayscale
TFRECORD_DIR = "./data/tfrecords"
MODEL_DIR = "models"

def parse_tfrecord(example_proto, grayscale=False):
    """
    Parse a single TFRecord example, including the pre-generated heatmap.
    """
    feature_description = {
        "track_id": tf.io.FixedLenFeature([], tf.string),
        "frame_idx": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "x": tf.io.FixedLenFeature([], tf.int64),
        "y": tf.io.FixedLenFeature([], tf.int64),
        "visibility": tf.io.FixedLenFeature([], tf.int64),
        "heatmap": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    frame = tf.image.decode_png(example["image"], channels=1 if grayscale else 3)
    frame = tf.cast(frame, tf.float32) / 255.0
    heatmap = tf.image.decode_png(example["heatmap"], channels=1)
    heatmap = tf.cast(heatmap, tf.float32) / 255.0
    x = tf.cast(example["x"], tf.float32)
    y = tf.cast(example["y"], tf.float32)
    # Ensure visibility is binary (0 or 1) and cast to int32
    visibility = tf.cast(tf.greater_equal(example["visibility"], 1), tf.int32)
    logger.debug("Visibility value: %s", visibility)
    tf.debugging.assert_shapes([(frame, (IMG_HEIGHT, IMG_WIDTH, 1 if grayscale else 3)),
                               (heatmap, (IMG_HEIGHT, IMG_WIDTH, 1))])
    return frame, heatmap, example["track_id"], example["frame_idx"]

def load_sequence(track_id, frame_indices, mode, seq, grayscale=False):
    """
    Load a sequence of frames and heatmaps from TFRecord using native TensorFlow operations.
    """
    logger = logging.getLogger(__name__)
    dataset = tf.data.TFRecordDataset(os.path.join(TFRECORD_DIR, f"{mode}.tfrecord"))
    frame_indices = tf.cast(frame_indices, tf.int64)

    dataset = dataset.map(
        lambda x: parse_tfrecord(x, grayscale), num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.filter(
        lambda frame, heatmap, tid, fid: tf.logical_and(
            tf.equal(tid, track_id), tf.reduce_any(tf.equal(fid, frame_indices))
        )
    )

    # Count frames to ensure we have enough
    def count_frames(count, _):
        return count + 1

    num_frames = dataset.reduce(tf.constant(0, dtype=tf.int64), count_frames)
    tf.debugging.assert_greater_equal(
        num_frames,
        tf.cast(seq, tf.int64),
        message=f"Track {track_id} has fewer than {seq} frames after filtering"
    )

    # Take exactly 'seq' frames
    dataset = dataset.take(seq)

    def reduce_fn(state, element):
        index, frames, heatmaps = state
        frame, heatmap, _, _ = element
        frames = frames.write(index, frame)  # Write frame at index
        heatmaps = heatmaps.write(index, heatmap)  # Write heatmap at index
        return index + 1, frames, heatmaps

    frames_acc = tf.TensorArray(dtype=tf.float32, size=seq, dynamic_size=False)
    heatmaps_acc = tf.TensorArray(dtype=tf.float32, size=seq, dynamic_size=False)
    initial_state = (tf.constant(0, dtype=tf.int32), frames_acc, heatmaps_acc)

    index, frames, heatmaps = dataset.reduce(initial_state, reduce_fn)

    try:
        frames = frames.concat()
        heatmaps = heatmaps.concat()
        frames = tf.reshape(frames, [seq, IMG_HEIGHT, IMG_WIDTH, 1 if grayscale else 3])
        heatmaps = tf.reshape(heatmaps, [seq, IMG_HEIGHT, IMG_WIDTH, 1])
        frames = tf.transpose(frames, [1, 2, 3, 0])  # [H, W, C, seq]
        heatmaps = tf.transpose(heatmaps, [1, 2, 3, 0])  # [H, W, 1, seq]
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

def get_tfrecord_sequences(mode, seq):
    """
    Returns a list of (track_id, frame_indices) tuples for TFRecord data by parsing the TFRecord file.
    """
    logger = logging.getLogger(__name__)
    sequences = []
    tfrecord_path = os.path.join(TFRECORD_DIR, f"{mode}.tfrecord")

    if not os.path.exists(tfrecord_path):
        logger.warning(f"TFRecord file {tfrecord_path} does not exist")
        return sequences

    # Parse TFRecord to collect track_id and frame_idx
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    track_frames = {}

    feature_description = {
        "track_id": tf.io.FixedLenFeature([], tf.string),
        "frame_idx": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_minimal(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return example["track_id"], example["frame_idx"]

    try:
        for record in dataset.map(parse_minimal):
            track_id, frame_idx = record
            track_id = track_id.numpy().decode('utf-8')  # Convert to string
            frame_idx = frame_idx.numpy()  # Convert to int
            if track_id not in track_frames:
                track_frames[track_id] = []
            track_frames[track_id].append(frame_idx)

        # Process each track to generate sequences
        for track_id, frame_indices in track_frames.items():
            frame_indices = sorted(frame_indices)  # Ensure frames are sorted
            num_frames = len(frame_indices)
            logger.debug("Track %s has %d frames, required %d", track_id, num_frames, seq)
            if num_frames < seq:
                logger.warning(
                    "Track %s has %d frames, need at least %d, skipping",
                    track_id, num_frames, seq
                )
                continue
            # Generate sequences of 'seq' consecutive frames
            for t in range(seq - 1, num_frames):
                seq_indices = frame_indices[t - seq + 1:t + 1]
                if len(seq_indices) == seq and seq_indices == list(range(seq_indices[0], seq_indices[0] + seq)):
                    sequences.append((track_id, seq_indices))
                else:
                    logger.warning(
                        "Non-sequential or incomplete frame indices %s for track %s, skipping",
                        seq_indices, track_id
                    )

        logger.debug("Found %d valid sequences for mode %s", len(sequences), mode)
        return sequences

    except Exception as e:
        logger.error("Error processing TFRecord %s: %s", tfrecord_path, str(e))
        return sequences

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

def main():
    logger.info("Starting training script")

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
        "--gpu_memory_limit",
        type=int,
        default=-1,
        help="Limit GPU memory usage in MB, -1 means no limit.",
    )

    args = parser.parse_args()
    setup_logging(args.debug)
    logger.info(
        "Training with seq=%d, grayscale=%s, debug=%s, resume=%s, model_name=%s",
        args.seq,
        args.grayscale,
        args.debug,
        args.resume,
        args.model_name,
    )

    if args.gpu_memory_limit > 0:
        limit_gpu_memory(args.gpu_memory_limit)

    if args.seq < 1:
        logger.error("Sequence length must be at least 1, got %d", args.seq)
        raise ValueError(f"Invalid sequence length: {args.seq}")

    if args.model_name not in [
        "VballNetFastV1",
        "VballNetV1",
        "PlayerNetFastV1",
        "TrackNetV4",
    ]:
        logger.error(
            "Invalid model name: %s. Must be 'VballNetFastV1', 'VballNetV1', 'PlayerNetFastV1', or 'TrackNetV4'",
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

    train_sequences = get_tfrecord_sequences("train", args.seq)
    test_sequences = get_tfrecord_sequences("test", args.seq)

    logger.info("Number of training sequences: %d", len(train_sequences))
    logger.info("Number of test sequences: %d", len(test_sequences))
    if len(train_sequences) == 0:
        logger.error(
            "No training data found. Check TFRECORD_DIR/train.tfrecord."
        )
        raise ValueError(
            "No training data found. Check TFRECORD_DIR/train.tfrecord."
        )
    if len(test_sequences) == 0:
        logger.warning("No test data found. Check TFRECORD_DIR/test.tfrecord.")

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in train_sequences],  # track_id
                [p[1] for p in train_sequences],  # frame_indices
            )
        )
        .shuffle(10)
        .map(
            lambda t, f: load_sequence(t, f, "train", args.seq, args.grayscale),
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

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                [p[0] for p in test_sequences],  # track_id
                [p[1] for p in test_sequences],  # frame_indices
            )
        )
        .map(
            lambda t, f: load_sequence(t, f, "test", args.seq, args.grayscale),
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
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
    model.summary(print_fn=lambda x: logger.info(x))

    for frames, heatmaps in train_dataset.take(1):
        logger.info(
            f"Train dataset sample - Frames shape: {frames.shape}, Heatmaps shape: {heatmaps.shape}"
        )
    for frames, heatmaps in test_dataset.take(1):
        logger.info(
            f"Test dataset sample - Frames shape: {frames.shape}, Heatmaps shape: {heatmaps.shape}"
        )

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

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=["mae"])
    logger.info(
        "Model compiled with optimizer=Adam(learning_rate=1e-3), loss=custom_loss, metrics=['mae']"
    )

    filepath = os.path.join(
        model_save_dir, f"{model_save_name}/{model_save_name}_{{epoch:02d}}.keras"
    )

    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()

        def on_epoch_begin(self, epoch, logs=None):
            current_lr = self.model.optimizer.learning_rate.numpy()
            logger.info(f"Epoch {epoch + 1}: Learning rate = {current_lr}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_best_only=False, monitor="val_loss"
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs", model_save_name)
        ),
        tf.keras.callbacks.EarlyStopping(patience=30, monitor="val_loss"),
        LearningRateLogger(),
    ]

    logger.info(
        "Callbacks configured: ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateLogger"
    )

    tf.profiler.experimental.start(
        os.path.join(MODEL_DIR, "logs", model_save_name, "profiler")
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

    tf.profiler.experimental.stop()

if __name__ == "__main__":
    main()