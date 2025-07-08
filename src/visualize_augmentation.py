import tensorflow as tf
import numpy as np
import os
import argparse
import logging
from PIL import Image
import matplotlib.pyplot as plt
from constants import HEIGHT, WIDTH, SIGMA
from utils import create_heatmap
from train_v1 import load_image_frames, get_video_and_csv_pairs, load_data, reshape_tensors, augment_sequence

# Parameters
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH  # 512
DATASET_DIR = "data/frames"  # Base directory for frames
#SIGMA = 5  # Radius for circular heatmap
MAG = 1.0  # Magnitude for heatmap
RATIO = 1.0  # Scaling factor for coordinates
DEBUG_DIR = "debug"  # Directory for saving combined images

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

def overlay_heatmap(frame, heatmap, alpha=0.4):
    """
    Overlay heatmap on frame using a colormap and transparency.

    Args:
        frame: NumPy array of shape (height, width, 3) with RGB values in [0, 1]
        heatmap: NumPy array of shape (height, width, 1) with values in [0, 1]
        alpha: Transparency for heatmap overlay (0.0 to 1.0)

    Returns:
        NumPy array of shape (height, width, 3) with overlaid heatmap
    """
    logger = logging.getLogger(__name__)

    # Convert heatmap to colormap (using 'jet' for visualization)
    cmap = plt.get_cmap('jet')
    heatmap_normalized = heatmap  # Already in [0, 1]
    heatmap_colored = cmap(heatmap_normalized[:, :, 0])[:, :, :3]  # Get RGB from colormap

    # Blend frame and heatmap
    blended = (frame * (1 - alpha) + heatmap_colored * alpha).astype(np.float32)

    logger.debug("Overlayed heatmap on frame, resulting shape: %s", blended.shape)
    return blended

def combine_images(frame, heatmap, track_id, frame_indices, debug_dir):
    """
    Combine original frame and frame with overlaid heatmap into a single image.

    Args:
        frame: NumPy array of shape (height, width, 3) with RGB values in [0, 1]
        heatmap: NumPy array of shape (height, width, 1) with values in [0, 1]
        track_id: String identifier for the track
        frame_indices: List of frame indices
        debug_dir: Directory to save the combined image
    """
    logger = logging.getLogger(__name__)

    try:
        # Overlay heatmap on frame
        frame_with_heatmap = overlay_heatmap(frame, heatmap)

        # Convert to uint8 for saving
        frame = (frame * 255).astype(np.uint8)
        frame_with_heatmap = (frame_with_heatmap * 255).astype(np.uint8)

        # Create combined image (stack vertically)
        combined = np.zeros((IMG_HEIGHT * 2, IMG_WIDTH, 3), dtype=np.uint8)
        combined[:IMG_HEIGHT, :, :] = frame
        combined[IMG_HEIGHT:, :, :] = frame_with_heatmap

        # Create filename with track_id and frame indices
        frame_str = "_".join(map(str, frame_indices))
        output_path = os.path.join(debug_dir, f"combined_{track_id}_{frame_str}.png")

        # Save combined image
        combined_img = Image.fromarray(combined)
        combined_img.save(output_path, format="PNG")
        logger.info("Saved combined image to %s", output_path)

    except Exception as e:
        logger.error("Error combining images for track_id %s, frame_indices %s: %s",
                     track_id, frame_indices, str(e))
        raise

def main():
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize augmented frames with overlaid heatmaps.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=3,
        help="Number of frames in sequence."
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=10,
        help="Maximum number of pairs to process for visualization."
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    logger.info(
        "Starting visualization script with seq=%d, debug=%s, max_pairs=%d",
        args.seq,
        args.debug,
        args.max_pairs
    )

    # Validate seq
    if args.seq < 1:
        logger.error("Sequence length must be at least 1, got %d", args.seq)
        raise ValueError(f"Invalid sequence length: {args.seq}")

    # Create debug directory
    os.makedirs(DEBUG_DIR, exist_ok=True)
    logger.info("Created debug directory: %s", DEBUG_DIR)

    # Get training pairs
    train_pairs = get_video_and_csv_pairs("train", args.seq)
    logger.info("Number of training pairs: %d", len(train_pairs))

    if len(train_pairs) == 0:
        logger.error(
            "No training data found. Check DATASET_DIR/train and frame/CSV files."
        )
        raise ValueError(
            "No training data found. Check DATASET_DIR/train and frame/CSV files."
        )

    # Limit the number of pairs to process
    train_pairs = train_pairs[:min(args.max_pairs, len(train_pairs))]
    logger.info("Processing %d pairs for visualization", len(train_pairs))

    # Process each pair
    for track_id, csv_path, frame_indices in train_pairs:
        try:
            # Load data
            frames, heatmaps = load_data(track_id, csv_path, frame_indices, "train", args.seq)

            # Reshape tensors
            frames, heatmaps = reshape_tensors(frames, heatmaps, args.seq)

            # Apply augmentation
            aug_frames, aug_heatmaps = augment_sequence(frames, heatmaps, args.seq)

            # Extract first RGB frame and first heatmap
            aug_frames = tf.transpose(aug_frames, [1, 2, 0]).numpy()  # (288, 512, 9)
            aug_heatmaps = tf.transpose(aug_heatmaps, [1, 2, 0]).numpy()  # (288, 512, 3)
            frame_rgb = aug_frames[:, :, :3]  # First frame (RGB)
            heatmap_single = aug_heatmaps[:, :, 0:1]  # First heatmap

            # Combine and save
            combine_images(frame_rgb, heatmap_single, track_id, frame_indices, DEBUG_DIR)

        except Exception as e:
            logger.error("Failed to process track_id %s, frame_indices %s: %s",
                         track_id, frame_indices, str(e))
            continue

    logger.info("Visualization completed. Check %s for saved images.", DEBUG_DIR)

if __name__ == "__main__":
    main()