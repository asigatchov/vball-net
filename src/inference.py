import argparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import os
import time
from tqdm import tqdm
from model.VballNetFastV1 import VballNetFastV1
from model.VballNetV1 import VballNetV1
from model.TrackNetV4 import TrackNetV4
from utils import custom_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Volleyball ball detection and tracking")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--track_length", type=int, default=8, help="Length of the ball track")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output video and CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights file (e.g., models/20250620_2352/vballNetV1_02.keras)")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable visualization on display using cv2")
    parser.add_argument("--only_csv", action="store_true", default=False, help="Save only CSV, skip video output")
    return parser.parse_args()

def load_model(model_path, input_height=288, input_width=512):
    if not os.path.exists(model_path):
        raise ValueError(f"Model weights file not found: {model_path}")

    model = None
    if 'VballNetV1' in model_path:
        model = VballNetV1(input_height, input_width, in_dim=9, out_dim=3)  # Three frames: in_dim=9, out_dim=3
    elif 'VballNetFastV1' in model_path:
        model = VballNetFastV1(input_height, input_width, in_dim=9, out_dim=3)
    elif 'TrackNetV4' in model_path:
        model = TrackNetV4(input_height, input_width, 'TypeB')
 
    elif  'PlayerNetFastV1' in model_path:
        from model.PlayerNetFastV1 import PlayerNetFastV1
        #    Создаём только PlayerNetFastV1 с 3 выходными каналами
        model = PlayerNetFastV1(input_shape=(9, input_height, input_width), output_channels=3)
    else:
        raise ValueError("Model type not recognized in model_path")

    model.load_weights(model_path)
    return model

def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames

def setup_output_writer(video_basename, output_dir, frame_width, frame_height, fps, only_csv):
    if output_dir is None or only_csv:
        return None, None

    output_path = os.path.join(output_dir, f'{video_basename}_predict.mp4')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    return out_writer, output_path

def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    csv_path = os.path.join(output_dir, f'{video_basename}_predict_ball.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Initialize CSV with headers
    pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y']).to_csv(csv_path, index=False)
    return csv_path

def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    # Append single result to CSV
    pd.DataFrame([result]).to_csv(csv_path, mode='a', header=False, index=False)

def preprocess_frame(frame, input_height=288, input_width=512):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32) / 255.0
    return frame

def postprocess_output(output, threshold=0.5, input_height=288, input_width=512):
    results = []
    for frame_idx in range(3):
        heatmap = output[0, frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results

def visualize_heatmaps(output, frame_index, input_height=288, input_width=512):
    for frame_idx in range(3):
        heatmap = output[0, frame_idx, :, :]
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        cv2.imshow(f'Heatmap Frame {frame_idx}', heatmap_color)
    cv2.waitKey(1)

def draw_track(frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0)):
    for point in list(track_points)[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, history_color, -1)
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 5, current_color, -1)
    return frame

def main():
    args = parse_args()
    input_width, input_height = 512, 288

    model = load_model(args.model_path, input_height, input_width)
    cap, frame_width, frame_height, fps, total_frames = initialize_video(args.video_path)

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv)
    csv_path = setup_csv_file(video_basename, args.output_dir)

    frame_buffer = deque(maxlen=3)
    track_points = deque(maxlen=args.track_length)
    prediction_buffer = {}
    frame_index = 0

    # Initialize progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    while cap.isOpened():
        start_time = time.time()  # Start time for FPS calculation

        ret = None
        ret, frame = cap.read()
        # ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = preprocess_frame(frame_rgb, input_height, input_width)
        frame_buffer.append(processed_frame)

        if len(frame_buffer) < 3:
            for _ in range(2):
                frame_buffer.append(processed_frame)

        if len(frame_buffer) == 3:

            input_tensor = np.concatenate(frame_buffer, axis=2)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

            output = model.predict(input_tensor, verbose=0)

            predictions = postprocess_output(output, input_height=input_height, input_width=input_width)

            visibility, x, y = predictions[2]
            if visibility == 0:
                x_orig, y_orig = -1, -1
                if len(track_points) > 0:
                    track_points.popleft()
            else:
                x_orig = x * frame_width / input_width
                y_orig = y * frame_height / input_height
                track_points.append((int(x_orig), int(y_orig)))

            result = {
                'Frame': frame_index,
                'Visibility': visibility,
                'X': int(x_orig),
                'Y': int(y_orig)
            }
            append_to_csv(result, csv_path)

            if args.visualize or out_writer is not None:
                vis_frame = frame.copy()
                vis_frame = draw_track(vis_frame, track_points)
                if args.visualize:
                    visualize_heatmaps(output, frame_index, input_height, input_width)
                    cv2.namedWindow(
                        "Ball Tracking", cv2.WINDOW_NORMAL
                    )  # Create window with freedom of dimensions

                    cv2.imshow('Ball Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)

        # Calculate and print FPS
        end_time = time.time()
        batch_time = end_time - start_time
        batch_fps = 1 / batch_time if batch_time > 0 else 0

        # Update progress bar (increment by 3 since we process 3 frames at a time)
        pbar.update(1)
        frame_index += 1

    # Close progress bar
    pbar.close()

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
