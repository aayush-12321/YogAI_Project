import os
import csv
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# MediaPipe 0.10.x requires a downloaded .task model file.
# This is the official lightweight pose landmarker model from Google.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")


def _ensure_model(model_path: str) -> None:
    """Download the .task model file if it is not already present."""
    if not os.path.exists(model_path):
        print(f"Downloading pose landmarker model to: {model_path}")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded successfully.")


def _build_csv_headers() -> list[str]:
    """Return CSV column headers for 33 landmarks (x, y, z, visibility each)."""
    headers = ["source_id", "label", "frame_number"]
    for i in range(33):
        headers.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"])
    return headers


def _landmarks_to_row(source_id: str, label: str, frame_number: int, landmarks) -> list:
    """Flatten a single pose result into a CSV row."""
    row = [source_id, label, frame_number]
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return row


def _resolve_output_path(raw_pose_path: str, pose_name: str) -> str:
    """
    Derive the default output CSV path from the input directory structure.
    Assumes: <root>/data/raw/<pose_name>/
    Output:  <root>/data/annotations/<pose_name>/<pose_name>_raw.csv
    """
    abs_input_path = os.path.abspath(raw_pose_path)
    raw_dir_path = os.path.dirname(abs_input_path)
    data_dir_path = os.path.dirname(raw_dir_path)
    annotations_dir = os.path.join(data_dir_path, "annotations", pose_name)
    os.makedirs(annotations_dir, exist_ok=True)
    return os.path.join(annotations_dir, f"{pose_name}_raw.csv")


def _process_image(filepath: str, source_id: str, class_name: str, landmarker) -> list | None:
    """
    Run pose detection on a single static image.

    Returns a flattened landmark row, or None if no pose was detected.
    """
    image = cv2.imread(filepath)
    if image is None:
        print(f"  [Warning] Could not read image: {source_id}")
        return None

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print(f"  [Warning] No pose detected in image: {source_id}")
        return None

    # Frame number 0 marks static images.
    return _landmarks_to_row(source_id, class_name, 0, result.pose_landmarks[0])


def _process_video(filepath: str, source_id: str, class_name: str, model_path: str) -> list[list]:
    """
    Extract pose landmarks from every 8th frame of a video.

    VIDEO mode requires a fresh landmarker instance per file because it is
    stateful (tracks poses across frames using timestamps).

    Returns a list of flattened landmark rows.
    """
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
    )

    rows = []
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 8 != 0:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # VIDEO mode requires a monotonically increasing timestamp in milliseconds.
            timestamp_ms = int((frame_count / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                rows.append(
                    _landmarks_to_row(source_id, class_name, frame_count, result.pose_landmarks[0])
                )

    cap.release()
    return rows


def extract_pose_landmarks(raw_pose_path: str, output_csv_path: str | None = None) -> None:
    """
    Extract 33 MediaPipe pose landmarks from all images and videos in a pose directory.

    Directory structure expected under raw_pose_path:
        <pose_name>/
            <class_name>/          (e.g. 'correct', 'arms_not_dropped')
                image_or_video.*

    Args:
        raw_pose_path:   Path to the raw pose folder (e.g. 'data/raw/mountain_pose').
        output_csv_path: Optional explicit path for the output CSV file.
                         When omitted, the file is saved to
                         <data_root>/annotations/<pose_name>/<pose_name>_raw.csv
    """
    abs_input_path = os.path.abspath(raw_pose_path)
    pose_name = os.path.basename(abs_input_path)

    csv_filepath = output_csv_path or _resolve_output_path(abs_input_path, pose_name)

    # Ensure the output directory exists when a custom path is provided.
    os.makedirs(os.path.dirname(os.path.abspath(csv_filepath)), exist_ok=True)

    model_path = DEFAULT_MODEL_PATH
    _ensure_model(model_path)

    # IMAGE mode landmarker is stateless and can be reused across all images.
    image_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )

    print(f"Starting extraction for: {pose_name}")
    print(f"Output CSV: {csv_filepath}")

    with open(csv_filepath, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(_build_csv_headers())

        with mp_vision.PoseLandmarker.create_from_options(image_options) as image_landmarker:
            for class_name in sorted(os.listdir(abs_input_path)):
                class_dir = os.path.join(abs_input_path, class_name)
                if not os.path.isdir(class_dir):
                    continue

                print(f"\nProcessing class: {class_name}")

                for filename in sorted(os.listdir(class_dir)):
                    filepath = os.path.join(class_dir, filename)
                    ext = filename.lower().rsplit(".", 1)[-1]
                    source_id = filename

                    if ext in {"mp4", "mov", "avi"}:
                        rows = _process_video(filepath, source_id, class_name, model_path)
                        for row in rows:
                            writer.writerow(row)
                        print(f"  [Video] {filename} -> {len(rows)} frames extracted.")

                    elif ext in {"jpg", "jpeg", "png"}:
                        row = _process_image(filepath, source_id, class_name, image_landmarker)
                        if row:
                            writer.writerow(row)
                            print(f"  [Image] {filename} -> 1 frame extracted.")

    print(f"\nExtraction complete. Data saved to: {csv_filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MediaPipe pose landmarks to CSV.")
    parser.add_argument(
        "pose_folder",
        nargs="?",
        default="../../data/raw/warrior2_pose",
        help="Path to the raw pose folder (default: ../../data/raw/warrior2_pose)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Optional path for the output CSV file. "
            "When omitted, saves to <data_root>/annotations/<pose_name>/<pose_name>_raw.csv"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.pose_folder):
        print(f"Error: '{args.pose_folder}' does not exist. Check your path.")
    else:
        extract_pose_landmarks(args.pose_folder, output_csv_path=args.output)