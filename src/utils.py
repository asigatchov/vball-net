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
from constants import CUSTOMER_DATASET_ROOT, WIDTH, HEIGHT
from model.VballNetV1 import VballNetV1
import tensorflow as tf
from tensorflow.keras.callbacks import Callback



def create_heatmap(x, y, visibility, height=288, width=512, r=5, mag=1.0):
    """
    Creates a heatmap tensor with a circular region of specified magnitude using cv2.
    """
    logger = logging.getLogger(__name__)
    visibility = tf.cast(visibility, tf.int32)
    if visibility == 0 or x <= 0 or y <= 0:
        #logger.debug("Zero heatmap due to visibility=0 or invalid coordinates (x=%s, y=%s)", x, y)
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

