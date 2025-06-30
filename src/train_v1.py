from model.VballNetV1 import VballNetV1
from constants import HEIGHT, WIDTH
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import cv2
import os
import argparse
from datetime import datetime
import glob``

# Parameters
IMG_HEIGHT = HEIGHT  # 288
IMG_WIDTH = WIDTH   # 512
NUM_FRAMES = 3
BATCH_SIZE = 4  # Increased for better training efficiency
DATASET_DIR = '/home/gled/frames'  # Base directory for frames
SIGMA = 5  # Radius for circular heatmap
MAG = 1.0  # Magnitude for heatmap
RATIO = 1.0  # Scaling factor for coordinates
MODEL_DIR = '/home/projects/www/vb-soft/vball-net/models'  # Directory for model saving

def custom_loss(y_true, y_pred):
    """
    Custom loss function for TrackNet training.
    """
    loss = (-1) * (
        K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) +
        K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1))
    )
    return K.mean(loss)

def display_heatmap(heatmap):
    """
    Displays the heatmap using OpenCV with a color map.
    """
    heatmap_np = heatmap.numpy().squeeze()
    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imshow('Heatmap', heatmap_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_heatmap(x, y, visibility, height=288, width=512, r=5, mag=1.0):
    """
    Creates a heatmap tensor with a circular region of specified magnitude.
    """
    visibility = tf.cast(visibility, tf.int32)
    if visibility == 0 or x <= 0 or y <= 0:
        return tf.zeros((height, width, 1), dtype=tf.float32)

    x = tf.cast((x+1.0), tf.float32)
    y = tf.cast((y+1.0), tf.float32)

    if x >= width or y >= height:
        tf.print("Warning: Coordinates out of bounds after offset (x, y):", x, y)
        return tf.zeros((height, width, 1), dtype=tf.float32)

    x_grid, y_grid = tf.meshgrid(tf.range(width, dtype=tf.float32),
                                 tf.range(height, dtype=tf.float32))

    heatmap = (y_grid - y)**2 + (x_grid - x)**2
    heatmap = tf.where(heatmap <= r**2, mag, 0.0)
    heatmap = tf.expand_dims(heatmap, axis=-1)
    if heatmap.shape != (height, width, 1):
        return tf.zeros((height, width, 1), dtype=tf.float32)

    tf.debugging.assert_shapes([(heatmap, (height, width, 1))])
    return heatmap


def load_image_frames(track_id, frame_indices, mode, height=288, width=512):
    """
    Loads three preprocessed image frames from DATASET_DIR/mode/track_id/.
    """
    frames = []
    track_dir = os.path.join(DATASET_DIR, mode, track_id)
    for idx in frame_indices:
        frame_path = os.path.join(track_dir, f"{idx}.jpg")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        # Load image
        frame = cv2.imread(frame_path)
        if frame is None:
           # import pdb; pdb.set_trace()
            raise ValueError(f"Failed to load frame: {frame_path}")

        # Check image shape
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame {frame_path} has unexpected shape {frame.shape}, expected (H, W, 3)")

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to TensorFlow tensor
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)

        # Resize to target dimensions
        # try:
        #     frame = tf.image.resize(frame, [height, width], method='bilinear')
        # except Exception as e:
        #     raise ValueError(f"Error resizing frame {frame_path}: {e}")
        # Normalize to [0, 1]
        frame = frame / 255.0

        frames.append(frame)

    # Concatenate frames along the channel axis
    return tf.concat(frames, axis=2)
def get_video_and_csv_pairs(mode):
    """
    Returns a list of (track_id, csv_path, frame_indices) tuples for videos in DATASET_DIR/mode.
    """
    pairs = []
    mode_dir = os.path.join(DATASET_DIR, mode)
    if not os.path.exists(mode_dir):
        print(f"Warning: Directory {mode_dir} does not exist.")
        return pairs

    video_dirs = [d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))]
    #import pdb; pdb.set_trace()
    for track_id in video_dirs:
        csv_path = os.path.join(mode_dir, f"{track_id}_ball.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found for {track_id} at {csv_path}, skipping.")
            continue
        df = pd.read_csv(csv_path, dtype={'X': np.int64, 'Y': np.int64, 'Visibility': np.int64})
        df = df.fillna({'X': 0, 'Y': 0, 'Visibility': 0})
        num_frames = len(df)
        if num_frames < NUM_FRAMES:
            print(f"Warning: {csv_path} has {num_frames} frames, need at least {NUM_FRAMES}, skipping.")
            continue
        for t in range(2, num_frames - 3):
            frame_indices = [t-2, t-1, t]
            # Verify all frames exist
            track_dir = os.path.join(mode_dir, track_id)
            if all(os.path.exists(os.path.join(track_dir, f"{idx}.jpg")) for idx in frame_indices):
                pairs.append((track_id, csv_path, frame_indices))
            else:
                print(f"Warning: Missing frames for {track_id} at indices {frame_indices}, skipping.")
    return pairs

def load_data(track_id, csv_path, frame_indices, mode):
    """
    Loads frames and heatmaps for a single sequence.
    """
    if isinstance(track_id, tf.Tensor):
        track_id = track_id.numpy().decode('utf-8') if track_id.dtype == tf.string else str(track_id.numpy())
    if isinstance(csv_path, tf.Tensor):
        csv_path = csv_path.numpy().decode('utf-8') if csv_path.dtype == tf.string else str(csv_path.numpy())
    if isinstance(frame_indices, tf.Tensor):
        frame_indices = frame_indices.numpy().tolist()

    frames = load_image_frames(track_id, frame_indices, mode, IMG_HEIGHT, IMG_WIDTH)

    df = pd.read_csv(csv_path, dtype={'X': np.int64, 'Y': np.int64, 'Visibility': np.int64})
    df = df.fillna({'X': 0, 'Y': 0, 'Visibility': 0})
    heatmaps = []
    for idx in frame_indices:
        if idx >= len(df):
            raise IndexError(f"Frame index {idx} out of range for CSV {csv_path}")
        row = df.iloc[idx]
        x, y, visibility = row['X'], row['Y'], row['Visibility']
        if not isinstance(visibility, (int, np.int64)) or visibility not in [0, 1]:
            print(f"Warning: Invalid visibility {visibility} in {csv_path} at index {idx}, setting to 0")
            visibility = 0
        x = x / RATIO
        y = y / RATIO
        heatmap = create_heatmap(x, y, visibility, IMG_HEIGHT, IMG_WIDTH, SIGMA, MAG)
        heatmaps.append(heatmap)

    heatmaps = tf.concat(heatmaps, axis=2)
    frames.set_shape([IMG_HEIGHT, IMG_WIDTH, 9])
    heatmaps.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return frames, heatmaps

# Create train and test datasets
train_pairs = get_video_and_csv_pairs('train')
test_pairs = get_video_and_csv_pairs('test')

print(f"Number of training pairs: {len(train_pairs)}")
print(f"Number of test pairs: {len(test_pairs)}")
if len(train_pairs) == 0:
    raise ValueError("No training data found. Check DATASET_DIR/train and frame/CSV files.")
if len(test_pairs) == 0:
    print("Warning: No test data found. Check DATASET_DIR/test and frame/CSV files.")

train_dataset = tf.data.Dataset.from_tensor_slices((
    [p[0] for p in train_pairs],
    [p[1] for p in train_pairs],
    [p[2] for p in train_pairs]
)).map(
    lambda t, c, f: tf.py_function(
        func=lambda x, y, z: load_data(x, y, z, 'train'),
        inp=[t, c, f],
        Tout=[tf.float32, tf.float32]
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = tf.data.Dataset.from_tensor_slices((
    [p[0] for p in test_pairs],
    [p[1] for p in test_pairs],
    [p[2] for p in test_pairs]
)).map(
    lambda t, c, f: tf.py_function(
        func=lambda x, y, z: load_data(x, y, z, 'test'),
        inp=[t, c, f],
        Tout=[tf.float32, tf.float32]
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

def reshape_tensors(frames, heatmaps):
    """
    Reshape tensors to (channels, height, width) if needed by the model.
    """
    frames = tf.ensure_shape(frames, [IMG_HEIGHT, IMG_WIDTH, 9])
    heatmaps = tf.ensure_shape(heatmaps, [IMG_HEIGHT, IMG_WIDTH, 3])
    # Transpose only if model expects (channels, height, width)
    frames = tf.transpose(frames, [2, 0, 1])  # (9, 288, 512)
    heatmaps = tf.transpose(heatmaps, [2, 0, 1])  # (3, 288, 512)
    return frames, heatmaps

train_dataset = train_dataset.map(reshape_tensors, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(reshape_tensors, num_parallel_calls=tf.data.AUTOTUNE)

def augment_sequence(frames, heatmaps):
    """
    Apply data augmentation to frames and heatmaps.
    """
    frames = tf.transpose(frames, [1, 2, 0])  # (288, 512, 9)
    heatmaps = tf.transpose(heatmaps, [1, 2, 0])  # (288, 512, 3)
    combined = tf.concat([frames, heatmaps], axis=2)
    combined = tf.image.random_flip_left_right(combined)
    #combined = tf.image.random_flip_up_down(combined)
    combined = tf.image.random_brightness(combined, max_delta=0.1)
    frames = combined[:, :, :9]
    heatmaps = combined[:, :, 9:]
    frames = tf.transpose(frames, [2, 0, 1])
    heatmaps = tf.transpose(heatmaps, [2, 0, 1])
    return frames, heatmaps

#train_dataset = train_dataset.map(augment_sequence, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Debugging info
train_size = tf.data.experimental.cardinality(train_dataset).numpy()
test_size = tf.data.experimental.cardinality(test_dataset).numpy()
print(f"Number of training batches: {train_size}")
print(f"Number of test batches: {test_size}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train VballNetV1 model.')
parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint.')
args = parser.parse_args()

# Create timestamped directory for saving models
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_save_dir = os.path.join(MODEL_DIR, timestamp)
os.makedirs(model_save_dir, exist_ok=True)

# Initialize model
model = VballNetV1(input_height=IMG_HEIGHT, input_width=IMG_WIDTH)

# Load latest checkpoint if resuming
initial_epoch = 0
if args.resume:
    checkpoint_files = glob.glob(os.path.join(MODEL_DIR, '*', 'vballNetV1_*.keras'))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Resuming training from {latest_checkpoint}")
        model = tf.keras.models.load_model(latest_checkpoint, custom_objects={'custom_loss': custom_loss})
        epoch_str = latest_checkpoint.split('_')[-1].replace('.keras', '')
        initial_epoch = int(epoch_str) if epoch_str.isdigit() else 0
    else:
        print("No checkpoints found, starting training from scratch.")

# Compile model
model.compile(
    optimizer='adam',
    loss=custom_loss,
    metrics=['mae']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_dir, 'vballNetV1_{epoch:02d}.keras'),
        save_best_only=False,
        monitor='val_loss'
    ),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs', timestamp)),
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
]

# Print shapes for debugging
for frames, heatmaps in train_dataset.take(1):
    print("Frames shape:", frames.shape)  # Expected: (batch_size, 9, 288, 512)
    print("Heatmaps shape:", heatmaps.shape)  # Expected: (batch_size, 3, 288, 512)

# Train model
print("Starting training...")
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)