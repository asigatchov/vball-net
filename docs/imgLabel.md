# Frame-by-Frame Video Annotation Script (imgLabel.py)

This Python script (`imgLabel.py`) is designed for annotating video frames to label the presence and position of an object (e.g., a ball) in each frame. It uses OpenCV to display video frames and allows users to mark object coordinates with mouse clicks, save annotations to a CSV file, and navigate through frames using keyboard controls. The script supports loading existing annotations and creating new ones.

## Features
- **Input**: Accepts a video file (`.mp4`) and an optional CSV file with prior annotations.
- **Annotation**:
  - Left-click to mark the object's center (sets `Visibility=1`, records `X` and `Y` coordinates).
  - Middle-click to indicate no object or unclear visibility (sets `Visibility=0`, `X=-1`, `Y=-1`).
- **Navigation**:
  - `n`: Next frame
  - `p`: Previous frame
  - `f`: Go to first frame
  - `l`: Go to last frame
  - `>`: Fast-forward 36 frames
  - `<`: Fast-backward 36 frames
- **Saving and Exiting**:
  - `s`: Save annotations to a CSV file.
  - `e`: Exit the program (prompts to save if unsaved changes exist).
- **CSV Support**: Loads existing annotations from a CSV file or creates a new dictionary for annotation.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- `argparse` (included in Python standard library)
- A utility module (`utils.py`) with functions: `save_info`, `load_info`, `go2frame`, and `show_image`.

## Usage
1. **Run the Script**:
   ```bash
   python imgLabel.py --video_path <path_to_video.mp4> [--csv_path <path_to_csv.csv>]
   ```
   - `--video_path`: Path to the input video file (required).
   - `--csv_path`: Path to an existing CSV file with annotations (optional).

2. **Annotation Process**:
   - The video opens in a window (`imgLabel`).
   - Left-click to mark the object's center.
   - Middle-click to indicate no object is present.
   - Use keyboard controls to navigate frames.
   - Press `s` to save annotations.
   - Press `e` to exit (with a prompt if unsaved changes exist).

3. **Output**:
   - Annotations are stored in a dictionary with frame indices and fields: `Frame`, `Visibility`, `X`, `Y`.
   - Saved to a CSV file in the `csv/` directory with the name derived from the video file (e.g., for `video.mp4`, the output is `csv/video.csv`) when `s` is pressed.

## Example
```bash
python imgLabel.py --video_path sample_video.mp4 --csv_path annotations.csv
```
- Loads `sample_video.mp4` and `annotations.csv` (if provided).
- Displays the first frame for annotation.
- User annotates frames, navigates, and saves results to `csv/sample_video.csv`.

## Notes
- Ensure the video file is a valid `.mp4`.
- If a CSV file is provided, it must match the video's frame count for successful loading.
- The script saves the output CSV file in the `csv/` directory, creating the directory if it does not exist.
- Unsaved changes prompt a confirmation before exiting.
- The script assumes the `utils.py` module is in the same directory with required helper functions.

## Example Annotation Workflow
1. Run: `python imgLabel.py --video_path ball_video.mp4`
2. Left-click to mark the ball's center in the current frame.
3. Press `n` to move to the next frame.
4. Middle-click if the ball is not visible.
5. Press `s` to save annotations to `csv/ball_video.csv`.
6. Press `e` to exit after saving.

This script is ideal for tasks requiring precise frame-by-frame object tracking and annotation in videos.
