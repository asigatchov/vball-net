import tensorflow as tf
import os
import pandas as pd
import numpy as np
import logging
from constants import HEIGHT, WIDTH

# Configuration
DATASET_DIR = "./data/frames"
TFRECORD_DIR = "./data/tfrecords"
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH   # 512

def setup_logging():
    """
    Configure logging for the script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

def _bytes_feature(value):
    """
    Create a TFRecord feature for bytes data.
    """
    if isinstance(value, str):
        value = value.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """
    Create a TFRecord feature for int64 data.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(data_dir, mode, output_path, grayscale=False):
    """
    Convert PNG images and CSV annotations to TFRecord for a given mode (train/test).
    
    Args:
        data_dir (str): Base directory containing train/test folders.
        mode (str): 'train' or 'test'.
        output_path (str): Path to save the TFRecord file.
        grayscale (bool): If True, verify images as grayscale; else RGB.
    """
    logger = setup_logging()
    logger.info(f"Creating TFRecord for {mode} at {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with tf.io.TFRecordWriter(output_path) as writer:
        mode_dir = os.path.join(data_dir, mode)
        if not os.path.exists(mode_dir):
            logger.error(f"Directory {mode_dir} does not exist")
            raise FileNotFoundError(f"Directory {mode_dir} not found")

        video_dirs = [d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))]
        if not video_dirs:
            logger.warning(f"No video directories found in {mode_dir}")
            return

        for track_id in video_dirs:
            csv_path = os.path.join(mode_dir, f"{track_id}_ball.csv")
            if not os.path.exists(csv_path):
                logger.warning(f"CSV not found for track_id {track_id} at {csv_path}")
                continue

            # Read CSV with annotations
            try:
                df = pd.read_csv(
                    csv_path,
                    dtype={"Frame": np.int64, "X": np.int64, "Y": np.int64, "Visibility": np.int64}
                )
                df = df.fillna({"X": 0, "Y": 0, "Visibility": 0})
                logger.debug(f"Loaded CSV for track_id {track_id} with {len(df)} frames")
            except Exception as e:
                logger.error(f"Failed to read CSV {csv_path}: {e}")
                continue

            # Process each frame
            for idx in range(len(df)):
                frame_path = os.path.join(mode_dir, track_id, f"{idx}.png")
                if not os.path.exists(frame_path):
                    logger.warning(f"Frame not found: {frame_path}")
                    continue

                try:
                    # Read and verify image
                    image_string = tf.io.read_file(frame_path)
                    image = tf.image.decode_png(image_string, channels=1 if grayscale else 3)
                    if image.shape[0] != IMG_HEIGHT or image.shape[1] != IMG_WIDTH:
                        logger.warning(f"Frame {frame_path} has incorrect dimensions: {image.shape}")
                        continue
                    if grayscale and image.shape[2] != 1:
                        logger.warning(f"Frame {frame_path} is not grayscale")
                        continue
                    if not grayscale and image.shape[2] != 3:
                        logger.warning(f"Frame {frame_path} is not RGB")
                        continue

                    # Create TFRecord example
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'track_id': _bytes_feature(track_id),
                        'frame_idx': _int64_feature(idx),
                        'image': _bytes_feature(image_string.numpy()),
                        'x': _int64_feature(df.iloc[idx]["X"]),
                        'y': _int64_feature(df.iloc[idx]["Y"]),
                        'visibility': _int64_feature(df.iloc[idx]["Visibility"])
                    }))
                    writer.write(example.SerializeToString())
                    logger.debug(f"Wrote frame {idx} for track_id {track_id}")
                except Exception as e:
                    logger.error(f"Failed to process frame {frame_path}: {e}")
                    continue

    logger.info(f"TFRecord created at {output_path}")

def main():
    """
    Main function to create TFRecord files for train and test datasets.
    """
    logger = setup_logging()
    logger.info("Starting TFRecord creation")
    
    # Create TFRecords for RGB
    create_tfrecord(DATASET_DIR, "train", os.path.join(TFRECORD_DIR, "train.tfrecord"), grayscale=False)
    create_tfrecord(DATASET_DIR, "test", os.path.join(TFRECORD_DIR, "test.tfrecord"), grayscale=False)
    
    # Optionally create TFRecords for grayscale if needed
    # create_tfrecord(DATASET_DIR, "train", os.path.join(TFRECORD_DIR, "train_grayscale.tfrecord"), grayscale=True)
    # create_tfrecord(DATASET_DIR, "test", os.path.join(TFRECORD_DIR, "test_grayscale.tfrecord"), grayscale=True)
    
    logger.info("TFRecord creation completed")

if __name__ == "__main__":
    main()