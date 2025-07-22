import cv2
import pandas as pd
import os
import argparse
from collections import deque
import numpy as np
import glob


def get_distinct_colors(n):
    """Return up to n high-contrast colors."""
    # Predefined high-contrast colors (BGR format)
    color_pool = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203),  # Pink
        (0, 128, 128),  # Teal
    ]
    return color_pool[: min(n, len(color_pool))]


def main(video_path, csv_path):
    # Derive video name from video file
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Search for CSV files matching the pattern ./demo/*/video_name_predict_ball.csv

    # Generate distinct colors for each CSV file
    colors = get_distinct_colors(1)
    # Read CSV files and store data
    df = pd.read_csv(csv_path)
    episode = os.path.basename(os.path.dirname(csv_path))  # e.g., 'ep210'
    tracks = []
    tracks.append({"df": df, "episode": episode, "history": deque(maxlen=10)})

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video
    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (width, height),
    )

    frame_idx = 0
    paused = False
    while cap.isOpened():
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Process each track
        for track_idx, track in enumerate(tracks):
            color = colors[track_idx]
            episode = track["episode"]
            df = track["df"]
            history = track["history"]

            # Find the row for the current frame
            row = df[df["Frame"] == frame_idx]
            if not row.empty and row.iloc[0]["Visibility"] == 1:
                x, y = int(row.iloc[0]["X"]), int(row.iloc[0]["Y"])
                if x != -1 and y != -1:
                    history.append((x, y))
            else:
                # Shorten track if ball is not visible
                if history:
                    history.popleft()

            # Draw the track
            for i in range(1, len(history)):
                cv2.line(frame, history[i - 1], history[i], color, 2)

            # Draw the current position if visible
            if history:
                cv2.circle(frame, history[-1], 5, color, -1)

            # Draw episode label in the top-left corner
            label_y = 30 + track_idx * 30
            cv2.putText(
                frame, episode, (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        # Write frame to output video
        out.write(frame)
        cv2.imshow("Frame", frame)

        # Handle keypresses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            frame_idx = min(frame_idx + 1, frame_count - 1)  # Skip forward 10 frames
            paused = True
        elif key == ord("p"):
            frame_idx = max(frame_idx - 1, 0)  # Skip backward 10 frames
            paused = True
        elif key == ord(" "):  # Space to toggle pause
            paused = not paused

        if not paused:
            frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize ball detection tracks on a video."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Directory containing CSV files"
    )
    args = parser.parse_args()
    main(args.video_path, args.csv_path)
