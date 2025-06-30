import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import argparse
import glob
import os
from collections import deque

from utils import create_heatmap


def display_heatmap_frames(csv_files, play=False):
    """
    Displays heatmaps, processed frames, unprocessed frames, and frame differences frame-by-frame for all CSV files using OpenCV.

    Args:
        csv_files (list): List of paths to CSV files containing Frame, Visibility, X, Y columns.
        play (bool): If True, automatically play frames with a delay; if False, wait for key press.
    """
    # Window name for display
    window_name = 'Heatmap and Frame'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for csv_path in csv_files:
        try:
            # Read CSV data from file
            df = pd.read_csv(csv_path)

            print(csv_path)
            print(df[df['X'] >= 511])  # Найти X > 512
            print(df[df['Y'] >= 287])  # Найти Y > 288

            # Verify required columns
            required_columns = ['Frame', 'Visibility', 'X', 'Y']
            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Skipping '{csv_path}' - missing required columns: {required_columns}")
                continue

            # Get total number of frames
            total_frames = len(df)
            print(f"Total frames in '{csv_path}': {total_frames}")

            # Get directory containing frames (same name as CSV without '_ball.csv')
            csv_dir = os.path.dirname(csv_path)
            csv_basename = os.path.basename(csv_path)
            frame_dir_name = csv_basename.replace('_ball.csv', '')
            frame_dir = os.path.join(csv_dir, frame_dir_name)

            # Get filename for display
            filename = os.path.basename(csv_path)

            # Initialize track points for history (max 10 positions)
            track_points = deque(maxlen=10)

            # Initialize previous frame for difference calculation
            prev_frame = None

            # Current frame index
            current_index = 0

            while current_index < len(df):
                row = df.iloc[current_index]
                frame_num = int(row['Frame'])
                visibility = int(row['Visibility'])
                x = float(row['X'])
                y = float(row['Y'])

                # Generate heatmap for the current frame
                heatmap = create_heatmap(x=x, y=y, visibility=visibility)

                # Convert TensorFlow tensor to NumPy array and remove channel dimension
                heatmap_np = heatmap.numpy().squeeze()

                # Normalize to [0, 255] for display
                heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_uint8 = heatmap_norm.astype(np.uint8)

                # Apply JET color map
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                # Add frame number, filename, and total frames text to heatmap
                cv2.putText(heatmap_color, f'File: {filename}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(heatmap_color, f'Frame: {frame_num}/{total_frames}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Load corresponding frame image (processed and unprocessed versions)
                frame_path = os.path.join(frame_dir, f"{frame_num}.png")
                if os.path.exists(frame_path):
                    frame_img = cv2.imread(frame_path)
                    unprocessed_img = cv2.imread(frame_path)  # Load unprocessed frame
                    if frame_img is None or unprocessed_img is None:
                        print(f"Warning: Could not load frame image '{frame_path}'")
                        frame_img = np.zeros((288, 512, 3), dtype=np.uint8)
                        unprocessed_img = np.zeros((288, 512, 3), dtype=np.uint8)
                    else:
                        # Resize both frames to 512x288
                        frame_img = cv2.resize(frame_img, (512, 288))
                        unprocessed_img = cv2.resize(unprocessed_img, (512, 288))
                else:
                    print(f"Warning: Frame image '{frame_path}' not found")
                    frame_img = np.zeros((288, 512, 3), dtype=np.uint8)
                    unprocessed_img = np.zeros((288, 512, 3), dtype=np.uint8)

                # Compute frame difference before drawing annotations
                if prev_frame is not None and frame_img.shape == prev_frame.shape:
                    diff_img = cv2.absdiff(frame_img, prev_frame)
                    # Convert to grayscale and enhance contrast for visibility
                    diff_img_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
                    diff_img_norm = cv2.normalize(diff_img_gray, None, 0, 255, cv2.NORM_MINMAX)
                    diff_img_color = cv2.cvtColor(diff_img_norm, cv2.COLOR_GRAY2BGR)
                    # Resize to 512x288
                    diff_img_color = cv2.resize(diff_img_color, (512, 288))
                else:
                    diff_img_color = np.zeros((288, 512, 3), dtype=np.uint8)
                prev_frame = frame_img.copy()  # Update previous frame before annotations

                # Update track points
                if visibility == 1 and x >= 0 and y >= 0:
                    x_int = int(x * 512 / frame_img.shape[1])
                    y_int = int(y * 288 / frame_img.shape[0])
                    track_points.append((x_int, y_int))
                else:
                    track_points.clear()  # Clear history if ball is not visible

                # Draw track history (blue lines and points) on processed frame
                for i in range(1, len(track_points)):
                    if track_points[i - 1] is not None and track_points[i] is not None:
                        cv2.line(frame_img, track_points[i - 1], track_points[i], (255, 0, 0), 2)
                        cv2.circle(frame_img, track_points[i - 1], 3, (255, 0, 0), -1)

                # Draw current point in red if visible
                if visibility == 1 and x >= 0 and y >= 0:
                    cv2.circle(frame_img, (x_int, y_int), 5, (0, 0, 255), -1)

                # Create top row: processed frame + unprocessed frame
                top_row = np.hstack((frame_img, unprocessed_img))

                # Create bottom row: heatmap + diff image
                bottom_row = np.hstack((heatmap_color, diff_img_color))

                # Combine top and bottom rows
                combined = np.vstack((top_row, bottom_row))

                # Display the combined image
                cv2.imshow(window_name, combined)

                # Check if last frame
                if current_index == len(df) - 1:
                    print("Reached the last frame of this file.")

                # Handle playback and navigation
                if play:
                    key = cv2.waitKey(33) & 0xFF  # ~30 fps
                else:
                    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):  # Next frame
                    current_index = min(current_index + 1, len(df) - 1)
                elif key == ord('p'):  # Previous frame
                    current_index = max(current_index - 1, 0)
                elif key == ord('>'):  # 10 frames forward
                    current_index = min(current_index + 10, len(df) - 1)
                elif key == ord('<'):  # 10 frames backward
                    current_index = max(current_index - 10, 0)
                elif key == 32:  # Space to toggle play/pause
                    play = not play

        except FileNotFoundError:
            print(f"Warning: CSV file '{csv_path}' not found")
        except Exception as e:
            print(f"Warning: Error processing '{csv_path}': {e}")

    # Clean up
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Display heatmaps and frames from all CSV files in a directory frame-by-frame.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path pattern to CSV files (e.g., /home/gled/frames/*_ball.csv)")
    parser.add_argument("--play", action="store_true", default=False,
                        help="If set, automatically play frames with a delay; otherwise, wait for key press.")
    args = parser.parse_args()

    # Find all CSV files matching the pattern
    csv_files = glob.glob(args.csv_path)
    if not csv_files:
        print(f"Error: No CSV files found matching pattern '{args.csv_path}'.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")
    display_heatmap_frames(csv_files, play=args.play)


if __name__ == "__main__":
    main()