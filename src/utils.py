"""
Utility functions for dataset generation, training utilities, and model/dataset management.
"""

import math
import numpy as np
import cv2
import tensorflow.keras.backend as K
import tensorflow as tf
import logging
import os
from constants import CUSTOMER_DATASET_ROOT, WIDTH, HEIGHT, DATASET_DIR, IMG_FORMAT, IMG_HEIGHT, IMG_WIDTH, RATIO, MAG, SIGMA

from model.VballNetV1 import VballNetV1
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import pandas as pd

def limit_gpu_memory(memory_limit_mb):
    """Limit GPU memory usage for the current TensorFlow process."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPUs found. Running on CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )
                ],
            )
        print(f"Set GPU memory limit to {memory_limit_mb} MB")
    except RuntimeError as e:
        print(f"Error setting GPU memory limit: {e}")
        raise


def load_image_frames(
    track_id, frame_indices, mode, height=288, width=512, grayscale=False
):
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
        try:
            #print(frame_path)
            image_string = tf.io.read_file(frame_path)
            frame = tf.image.decode_jpeg(image_string, channels=1 if grayscale else 3)
        except Exception as e:
            logger.error("Failed to load or decode frame %s: %s", frame_path, str(e))
            raise ValueError(f"Failed to load frame {frame_path}: {e}")

        expected_channels = 1 if grayscale else 3
        if len(frame.shape) != 3 or frame.shape[2] != expected_channels:
            logger.error(
                "Frame %s has unexpected shape %s, expected (H, W, %d)",
                frame_path,
                frame.shape,
                expected_channels,
            )
            raise ValueError(
                f"Frame {frame_path} has unexpected shape {frame.shape}, expected (H, W, {expected_channels})"
            )

        logger.debug("Loaded frame %s with shape %s", frame_path, frame.shape)
        frame = tf.cast(frame, tf.float32)
        frame = frame / 255.0
        frames.append(frame)

    try:
        concatenated = tf.concat(frames, axis=2)
        logger.debug(
            "Concatenated frames shape %s for track_id %s, indices %s",
            concatenated.shape,
            track_id,
            frame_indices,
        )
        return concatenated
    except Exception as e:
        logger.error(
            "Failed to concatenate frames for track_id %s, indices %s: %s",
            track_id,
            frame_indices,
            str(e),
        )
        raise ValueError(f"Failed to concatenate frames: {e}")


def load_data(track_id, csv_path, frame_indices, mode, seq, grayscale=False):
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

    frames = load_image_frames(
        track_id, frame_indices, mode, IMG_HEIGHT, IMG_WIDTH, grayscale
    )
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
                heatmap = tf.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)
            else:
                heatmap = create_heatmap(
                    x, y, visibility, IMG_HEIGHT, IMG_WIDTH, SIGMA, MAG
                )
        heatmaps.append(heatmap)

    heatmaps = tf.concat(heatmaps, axis=2)
    frames.set_shape([IMG_HEIGHT, IMG_WIDTH, seq * (1 if grayscale else 3)])
    heatmaps.set_shape([IMG_HEIGHT, IMG_WIDTH, seq])
    logger.debug(
        "Loaded data for track_id %s: frames shape %s, heatmaps shape %s",
        track_id,
        frames.shape,
        heatmaps.shape,
    )
    return frames, heatmaps


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


def create_heatmap(x, y, visibility, height=288, width=512, r=5, mag=1.0):
    """
    Creates a heatmap tensor with a circular region of specified magnitude using cv2.
    """
    logger = logging.getLogger(__name__)
    visibility = tf.cast(visibility, tf.int32)
    if visibility == 0 or x <= 0 or y <= 0:
        # logger.debug("Zero heatmap due to visibility=0 or invalid coordinates (x=%s, y=%s)", x, y)
        return tf.zeros((height, width, 1), dtype=tf.float32)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    if x >= width or y >= height:
        logger.warning("Coordinates out of bounds (x=%s, y=%s)", x, y)
        return tf.zeros((height, width, 1), dtype=tf.float32)

    # Create blank heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Draw filled circle using cv2
    cv2.circle(heatmap, (int(x+1), int(y+1)), r, mag, -1)

    # Convert to tensor and add channel dimension
    heatmap = tf.convert_to_tensor(heatmap, dtype=tf.float32)
    heatmap = tf.expand_dims(heatmap, axis=-1)

    if heatmap.shape != (height, width, 1):
        logger.warning(f"Unexpected heatmap shape {heatmap.shape}, returning zero heatmap {x}, {y}, {visibility}")
        return tf.zeros((height, width, 1), dtype=tf.float32)

    logger.debug("Created heatmap with shape %s", heatmap.shape)
    tf.debugging.assert_shapes([(heatmap, (height, width, 1))])
    return heatmap

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
                    frame = frames_np[:, :, i : i + 1]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    frame = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
                    frame = tf.ensure_shape(frame, [288, 512, 3])
                else:
                    frame = frames_np[:, :, i * 3 : (i + 1) * 3]
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
                    combined = tf.concat(
                        [frame, true_heatmap, pred_heatmap], axis=1
                    )
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


####################################
# Dataset related helper functions #
####################################

def genHeatMap(w, h, cx, cy, r, mag):
    """
    Generate a heatmap with a circular region set to a specified magnitude.
    If the center coordinates (cx, cy) are negative, the function returns a zero-filled heatmap.
    """
    if cx < 0 or cy < 0:
        return np.zeros((h, w))

    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap * mag

##############################
# Training related functions #
##############################

def outcome(y_pred, y_true, tol):
    """
    Calculate the outcomes (TP, TN, FP1, FP2, FN) for a batch of predicted heatmaps versus ground truth.
    """
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(3):
            if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
                TN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
                FP2 += 1
            elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
                FN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
                h_pred = y_pred[i][j] * 255
                h_true = y_true[i][j] * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0

                if not rects:
                    debug_dir = "./debug"
                    if os.path.exists(debug_dir) is False:
                        os.makedirs(debug_dir)

                    logging.warning("No contours found in predicted heatmap for batch %d, channel %d", i, j)
                    cv2.imwrite(f"{debug_dir}/debug_pred_{i}_{j}.png", h_pred)
                    cv2.imwrite(f"{debug_dir}/debug_true_{i}_{j}.png", h_true)
                    FP1 += 1
                    continue

                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)


class OutcomeMetricsCallback(Callback):
    def __init__(self, validation_data, tol=10, log_dir="logs/outcome_metrics"):
        super(OutcomeMetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.tol = tol
        self.logger = logging.getLogger(__name__)
        # Initialize TensorBoard writer
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Initialize counters
        total_TP = total_TN = total_FP1 = total_FP2 = total_FN = 0
        num_batches = 0

        # Iterate over validation dataset
        for batch_frames, batch_heatmaps in self.validation_data:
            # Predict heatmaps
            y_pred = self.model.predict(batch_frames, verbose=0)
            y_true = batch_heatmaps

            # Convert tensors to NumPy arrays and transpose to (batch, height, width, channels)
            y_pred = np.transpose(y_pred, [0, 2, 3, 1])  # (batch, 288, 512, 3)
            y_true = np.transpose(y_true, [0, 2, 3, 1])  # (batch, 288, 512, 3)

            # Compute outcome metrics
            TP, TN, FP1, FP2, FN = outcome(y_pred, y_true, self.tol)

            # Accumulate metrics
            total_TP += TP
            total_TN += TN
            total_FP1 += FP1
            total_FP2 += FP2
            total_FN += FN
            num_batches += 1

        # Compute averages or totals
        metrics = {
            "TP": total_TP,
            "TN": total_TN,
            "FP1": total_FP1,
            "FP2": total_FP2,
            "FN": total_FN,
        }

        # Log metrics to TensorBoard
        with self.writer.as_default():
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(f"outcome/{metric_name}", metric_value, step=epoch)
            self.writer.flush()

        # Log to console
        self.logger.info(
            "Epoch %d Outcome Metrics: TP=%d, TN=%d, FP1=%d, FP2=%d, FN=%d",
            epoch + 1,
            total_TP,
            total_TN,
            total_FP1,
            total_FP2,
            total_FN,
        )


def custom_loss(y_true, y_pred):
    """
    Custom loss function for TrackNet training.
    """
    loss = (-1) * (K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) +
                  K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
    return loss

###################################
# Model/dataset related functions #
###################################

def get_model(model_name, height, width):
#     """
#     Retrieve an instance of a TrackNet model based on the specified model name.
    from model.VballNetFastV1 import VballNetFastV1
    from model.VballNetV1 import VballNetV1


    if model_name == 'VballNetFastV1':
        return VballNetFastV1(height, width, in_dim=9, out_dim=3)

    return VballNetV1(height, width, in_dim=9, out_dim=3)


def save_info(info, video_path):
    success = False
    video_name = 'csv/' + os.path.split(video_path)[-1][:-4]
    csv_path = video_name+'_ball.csv'
    try:
        with open(csv_path, 'w') as file:
            file.write("Frame,Visibility,X,Y\n")
            for frame in info:
                data = "{},{},{:.3f},{:.3f}".format(info[frame]["Frame"], info[frame]["Visibility"],
                                            info[frame]["X"],info[frame]["Y"])
                file.write(data+'\n')
        success = True
        print("Save info successfully into", video_name+'_ball.csv')
    except:
        print("Save info failure ", csv_path)

    return success

def load_info(csv_path):
    with open(csv_path, 'r') as file:
        lines = file.readlines()
        n_frames = len(lines) - 1
        info = {
            idx:{
            'Frame': idx,
            'Visibility': 0,
            'x': -1,
            'y': -1
            } for idx in range(n_frames)
        }

        for line in lines[1:]:
            frame, Visibility, x, y = line.split(',')[0:4]
            frame = int(frame)

            if info.get(frame) is None:
                print("Frame {} not found in info, creating new entry.".format(frame))
                info[frame] = {
                    'Frame': frame,
                    'Visibility': 0,
                    'X': -1,
                    'Y': -1
                }


            info[frame]['Frame'] = frame
            info[frame]['Visibility'] = int(Visibility)
            info[frame]['X'] = float(x)
            info[frame]['Y'] = float(y)

    return info

def show_image(image, frame_no, x, y):
    h, w, _ = image.shape
    if x != -1 and y != -1:
        x_pos = int(x)
        y_pos = int(y)
        cv2.circle(image, (x_pos, y_pos), 5, (0, 0, 255), -1)
    text = "Frame: {}".format(frame_no)
    cv2.putText(image, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
    return image

def go2frame(cap, frame_no, info):
    x,y = -1, -1
    if frame_no in info:
        x, y = info[frame_no]['X'], info[frame_no]['Y']

    cap.set(1, frame_no)
    ret, image = cap.read()
    image = show_image(image, frame_no, x, y)
    return image
