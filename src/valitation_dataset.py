import pandas as pd
import numpy as np
import argparse
import cv2
import os
from scipy.optimize import curve_fit
from collections import deque

def quadratic(t, a, b, c):
    """Quadratic polynomial function: y = a*t^2 + b*t + c"""
    return a * t**2 + b * t + c

def cubic(t, a, b, c, d):
    """Cubic polynomial function: y = a*t^3 + b*t^2 + c*t + d"""
    return a * t**3 + b * t**2 + c * t + d

def fit_polynomial(t, x, y):
    """
    Fit polynomials to x and y coordinates over time (frame numbers).
    Uses cubic polynomial for segments with 4+ points, quadratic for 3 points.
    Returns parameters and residuals.
    """
    try:
        # Choose polynomial based on number of points
        if len(t) >= 4:
            poly_func = cubic
        else:  # len(t) == 3
            poly_func = quadratic

        # Fit polynomial to x coordinates
        popt_x, _ = curve_fit(poly_func, t, x)
        x_fit = poly_func(t, *popt_x)
        x_residuals = x - x_fit

        # Fit polynomial to y coordinates
        popt_y, _ = curve_fit(poly_func, t, y)
        y_fit = poly_func(t, *popt_y)
        y_residuals = y - y_fit

        return popt_x, popt_y, x_residuals, y_residuals
    except RuntimeError:
        print("Warning: Curve fitting failed for segment.")
        return None, None, None, None

def detect_anomalies(df, residual_threshold=10.0, jump_threshold=50.0, min_segment_length=3):
    """
    Analyze ball trajectory from DataFrame to detect false detections.

    Args:
        df (pd.DataFrame): DataFrame with Frame, Visibility, X, Y columns.
        residual_threshold (float): Threshold for residual to flag as anomaly (pixels).
        jump_threshold (float): Threshold for displacement between consecutive frames (pixels).
        min_segment_length (int): Minimum number of frames in a segment to fit a curve.

    Returns:
        list: List of anomaly descriptions with frame numbers.
        set: Set of frame numbers flagged as anomalies.
    """
    anomalies = []
    anomaly_frames = set()

    # Segment visible frames into continuous sequences
    visible_segments = []
    current_segment = []
    for _, row in df.iterrows():
        if row['Visibility'] == 1:
            current_segment.append(row)
        else:
            if current_segment:
                visible_segments.append(pd.DataFrame(current_segment))
                current_segment = []
    if current_segment:
        visible_segments.append(pd.DataFrame(current_segment))

    # Analyze each segment
    for segment in visible_segments:
        if len(segment) < min_segment_length:
            continue  # Skip segments too short for fitting

        t = segment['Frame'].values
        x = segment['X'].values
        y = segment['Y'].values

        # Fit polynomial to segment
        popt_x, popt_y, x_residuals, y_residuals = fit_polynomial(t, x, y)

        if x_residuals is not None and y_residuals is not None:
            # Detect large residuals
            for i, (frame, x_res, y_res) in enumerate(zip(segment['Frame'], x_residuals, y_residuals)):
                residual_magnitude = np.sqrt(x_res**2 + y_res**2)
                if residual_magnitude > residual_threshold:
                    anomalies.append(
                        f"Frame {int(frame)}: Large deviation from smooth trajectory "
                        f"(residual = {residual_magnitude:.2f} pixels)"
                    )
                    anomaly_frames.add(int(frame))

        # Check for large jumps in consecutive visible frames
        for i in range(1, len(segment)):
            frame_prev = segment['Frame'].iloc[i - 1]
            frame_curr = segment['Frame'].iloc[i]
            x_prev, y_prev = segment['X'].iloc[i - 1], segment['Y'].iloc[i - 1]
            x_curr, y_curr = segment['X'].iloc[i], segment['Y'].iloc[i]
            displacement = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
            if displacement > jump_threshold:
                anomalies.append(
                    f"Frame {int(frame_curr)}: Implausible jump from Frame {int(frame_prev)} "
                    f"(displacement = {displacement:.2f} pixels)"
                )
                anomaly_frames.add(int(frame_curr))

    # Check for isolated visible points (potential false positives)
    for i in range(1, len(df) - 1):
        if (df['Visibility'].iloc[i] == 1 and
            df['Visibility'].iloc[i - 1] == 0 and
            df['Visibility'].iloc[i + 1] == 0):
            frame = df['Frame'].iloc[i]
            anomalies.append(f"Frame {int(frame)}: Isolated visible point (potential false detection)")
            anomaly_frames.add(int(frame))

    return anomalies, anomaly_frames

def visualize_anomalies(csv_path, play=False, residual_threshold=10.0, jump_threshold=50.0, min_segment_length=3):
    """
    Visualize frames with ball trajectory and highlight false detections.

    Args:
        csv_path (str): Path to CSV file with Frame, Visibility, X, Y columns.
        play (bool): If True, automatically play frames; if False, wait for key press.
        residual_threshold (float): Threshold for residual to flag as anomaly (pixels).
        jump_threshold (float): Threshold for displacement between consecutive frames (pixels).
        min_segment_length (int): Minimum number of frames in a segment to fit a curve.
    """
    # Read CSV file
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)
    required_columns = ['Frame', 'Visibility', 'X', 'Y']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file '{csv_path}' missing required columns: {required_columns}")
        return

    # Detect anomalies
    anomalies, anomaly_frames = detect_anomalies(
        df, residual_threshold, jump_threshold, min_segment_length
    )

    # Print anomalies
    if not anomalies:
        print("No anomalies detected.")
    else:
        print(f"Found {len(anomalies)} anomalies:")
        for anomaly in anomalies:
            print(anomaly)

    # Get directory containing frames
    csv_dir = os.path.dirname(csv_path)
    csv_basename = os.path.basename(csv_path)
    frame_dir_name = csv_basename.replace('_ball.csv', '')
    frame_dir = os.path.join(csv_dir, frame_dir_name)

    # Initialize window
    window_name = 'Ball Trajectory with Anomalies'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Initialize track points for history (max 10 positions)
    track_points = deque(maxlen=10)

    # Iterate through frames
    for _, row in df.iterrows():
        frame_num = int(row['Frame'])
        visibility = int(row['Visibility'])
        x = float(row['X'])
        y = float(row['Y'])

        # Load frame image
        frame_path = os.path.join(frame_dir, f"{frame_num}.jpg")
        if os.path.exists(frame_path):
            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                print(f"Warning: Could not load frame image '{frame_path}'")
                frame_img = np.zeros((288, 512, 3), dtype=np.uint8)
            else:
                frame_img = cv2.resize(frame_img, (512, 288))
        else:
            print(f"Warning: Frame image '{frame_path}' not found")
            frame_img = np.zeros((288, 512, 3), dtype=np.uint8)

        # Update track points
        if visibility == 1 and x >= 0 and y >= 0:
            x_int = int(x * 512 / frame_img.shape[1])
            y_int = int(y * 288 / frame_img.shape[0])
            track_points.append((x_int, y_int))
        else:
            track_points.clear()  # Clear history if ball is not visible

        # Draw trajectory history (blue lines and points)
        for i in range(1, len(track_points)):
            if track_points[i - 1] is not None and track_points[i] is not None:
                cv2.line(frame_img, track_points[i - 1], track_points[i], (255, 0, 0), 2)
                cv2.circle(frame_img, track_points[i - 1], 3, (255, 0, 0), -1)

        # Draw current point (yellow for anomaly, red for normal)
        if visibility == 1 and x >= 0 and y >= 0:
            color = (0, 255, 255) if frame_num in anomaly_frames else (0, 0, 255)  # Yellow for anomaly, red for normal
            cv2.circle(frame_img, (x_int, y_int), 5, color, -1)

        # Add text annotations
        cv2.putText(frame_img, f'Frame: {frame_num}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if frame_num in anomaly_frames:
            cv2.putText(frame_img, 'Anomaly Detected', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display frame
        cv2.imshow(window_name, frame_img)

        # Handle playback
        if play:
            key = cv2.waitKey(33) & 0xFF  # ~30 fps
        else:
            key = cv2.waitKey(0) & 0xFF  # Wait for key press

        if key == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize volleyball ball trajectory for false detections.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file (e.g., /path/to/4m2w_transmash_20250504_0001_ball.csv)")
    parser.add_argument("--play", action="store_true", default=False,
                        help="If set, automatically play frames; otherwise, wait for key press.")
    parser.add_argument("--residual_threshold", type=float, default=30.0,
                        help="Threshold for residual to flag as anomaly (pixels).")
    parser.add_argument("--jump_threshold", type=float, default=50.0,
                        help="Threshold for displacement between consecutive frames (pixels).")
    parser.add_argument("--min_segment_length", type=int, default=3,
                        help="Minimum number of frames in a segment to fit a curve.")
    args = parser.parse_args()

    visualize_anomalies(
        csv_path=args.csv_path,
        play=args.play,
        residual_threshold=args.residual_threshold,
        jump_threshold=args.jump_threshold,
        min_segment_length=args.min_segment_length
    )

if __name__ == "__main__":
    main()