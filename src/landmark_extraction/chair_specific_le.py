import os
import csv
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
FRAME_TO_SKIP = 5

# 
# MediaPipe landmark indices (for reference)
# 
# 0  = nose
# 11 = left_shoulder,  12 = right_shoulder
# 23 = left_hip,       24 = right_hip
# 

# Visibility threshold: a landmark must meet this confidence to be trusted
# for orientation detection. Lower than this and we fall back to default.
VISIBILITY_THRESHOLD = 0.5

# Default orientation for chair pose data:
#   Most recordings show the person's LEFT side facing the camera,
#   meaning the person is facing RIGHT (nose.x > hip_center.x).
#   is_left_facing = False  →  right-facing (default)
DEFAULT_IS_LEFT_FACING = False


def _ensure_model(model_path: str) -> None:
    """Download the .task model file if it is not already present."""
    if not os.path.exists(model_path):
        print(f"Downloading pose landmarker model to: {model_path}")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded successfully.")


# 
# Orientation detection
# 

def _detect_orientation(landmarks) -> bool | None:
    """
    Auto-detect whether the person is left-facing from pose landmarks.

    Strategy (nose + shoulder consensus vs hip center):
    
    In a side-view chair pose (Utkatasana), the nose and shoulder centre
    will both be clearly to one side of the hip centre:

        nose.x > hip_center.x   AND   shoulder_center.x > hip_center.x
            → person faces RIGHT  →  is_left_facing = False

        nose.x < hip_center.x   AND   shoulder_center.x < hip_center.x
            → person faces LEFT   →  is_left_facing = True

    If the two signals disagree (rare edge case — e.g. extreme forward lean
    that shifts one reference point) we return None so the caller can fall
    back to the dataset default.

    Visibility guard:
        All four landmarks (nose, both shoulders, both hips) must exceed
        VISIBILITY_THRESHOLD. If any is invisible/unreliable, return None.

    Args:
        landmarks: list of 33 NormalizedLandmark objects from MediaPipe.

    Returns:
        True  → left-facing
        False → right-facing
        None  → undetermined (caller should use dataset default)
    """
    nose       = landmarks[0]
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_hip      = landmarks[23]
    r_hip      = landmarks[24]

    # Visibility guard — all key landmarks must be reliably detected
    key_landmarks = [nose, l_shoulder, r_shoulder, l_hip, r_hip]
    if any(lm.visibility < VISIBILITY_THRESHOLD for lm in key_landmarks):
        return None

    hip_center_x      = (l_hip.x      + r_hip.x)      / 2.0
    shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2.0

    nose_ahead_of_hip      = nose.x           > hip_center_x
    shoulder_ahead_of_hip  = shoulder_center_x > hip_center_x

    if nose_ahead_of_hip and shoulder_ahead_of_hip:
        # Both signals agree: person faces RIGHT
        return False
    elif not nose_ahead_of_hip and not shoulder_ahead_of_hip:
        # Both signals agree: person faces LEFT
        return True
    else:
        # Signals disagree — undetermined
        return None


def _determine_orientation(landmarks, source_id: str) -> bool:
    """
    Wrapper that calls _detect_orientation and falls back to the dataset
    default when detection is inconclusive.

    Returns:
        bool — True if left-facing, False if right-facing.
    """
    result = _detect_orientation(landmarks)
    if result is None:
        print(
            f"  [Orientation] Could not confidently determine facing for: {source_id}. "
            f"Defaulting to {'left-facing' if DEFAULT_IS_LEFT_FACING else 'right-facing'}."
        )
        return DEFAULT_IS_LEFT_FACING
    return result


# 
# CSV helpers
# 

def _build_csv_headers() -> list[str]:
    """Return CSV column headers: metadata + orientation flag + 33 landmarks."""
    headers = ["source_id", "label", "frame_number", "is_left_facing"]
    for i in range(33):
        headers.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"])
    return headers


def _landmarks_to_row(
    source_id: str,
    label: str,
    frame_number: int,
    left_facing: bool,
    landmarks,
) -> list:
    """Flatten a single pose result into a CSV row."""
    row = [source_id, label, frame_number, int(left_facing)]
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
    raw_dir_path   = os.path.dirname(abs_input_path)
    data_dir_path  = os.path.dirname(raw_dir_path)
    annotations_dir = os.path.join(data_dir_path, "annotations", pose_name)
    os.makedirs(annotations_dir, exist_ok=True)
    return os.path.join(annotations_dir, f"{pose_name}_raw.csv")


# 
# Per-file processors
# 

def _process_image(
    filepath: str,
    source_id: str,
    class_name: str,
    landmarker,
) -> list | None:
    """
    Run pose detection on a single static image.

    Orientation is auto-detected from landmarks.
    Returns a flattened landmark row, or None if no pose was detected.
    """
    image = cv2.imread(filepath)
    if image is None:
        print(f"  [Warning] Could not read image: {source_id}")
        return None

    img_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print(f"  [Warning] No pose detected in image: {source_id}")
        return None

    landmarks   = result.pose_landmarks[0]
    left_facing = _determine_orientation(landmarks, source_id)

    return _landmarks_to_row(source_id, class_name, 0, left_facing, landmarks)


def _process_video(
    filepath: str,
    source_id: str,
    class_name: str,
    model_path: str,
) -> list[list]:
    """
    Extract pose landmarks from every FRAME_TO_SKIP+1th frame of a video.

    Orientation is auto-detected independently per frame from landmarks.
    If a frame's detection is inconclusive, the dataset default is used.

    VIDEO mode requires a fresh landmarker instance per file because it is
    stateful (tracks poses across frames using timestamps).

    Returns a list of flattened landmark rows.
    """
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
    )

    rows        = []
    cap         = cv2.VideoCapture(filepath)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % (FRAME_TO_SKIP+1) != 0:
                continue

            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # VIDEO mode requires a monotonically increasing timestamp in milliseconds.
            timestamp_ms = int((frame_count / fps) * 1000)
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks   = result.pose_landmarks[0]
                left_facing = _determine_orientation(landmarks, f"{source_id}@frame{frame_count}")
                rows.append(
                    _landmarks_to_row(source_id, class_name, frame_count, left_facing, landmarks)
                )

    cap.release()
    return rows


# 
# Main extraction entry point
# 

def extract_pose_landmarks(raw_pose_path: str, output_csv_path: str | None = None) -> None:
    """
    Extract 33 MediaPipe pose landmarks from all images and videos in a
    chair pose (Utkatasana) directory.

    Orientation detection
    -
    Unlike the Warrior 2 extractor, chair pose videos have no filename
    prefix convention. Orientation is auto-detected purely from landmarks:

        Detection logic (nose + shoulder consensus vs hip centre):
        - nose.x > hip_center.x  AND  shoulder_center.x > hip_center.x
              → right-facing  (is_left_facing = 0)  [dataset default]
        - nose.x < hip_center.x  AND  shoulder_center.x < hip_center.x
              → left-facing   (is_left_facing = 1)
        - Signals disagree or visibility too low
              → falls back to dataset default (right-facing)

    The flag is stored as `is_left_facing` (0 or 1) in the output CSV so
    that the downstream feature-engineering step can swap left/right
    landmark index pairs to normalise all samples to a consistent facing
    direction — no pixel flipping is performed here.

    Supported classes (sub-folders under raw_pose_path):
        correct | bent_forward | dropped_arms | shallow_stance

    Directory structure expected under raw_pose_path:
        <pose_name>/
            <class_name>/
                video_or_image.*

    Args:
        raw_pose_path:   Path to the raw pose folder (e.g. 'data/raw/chair_pose').
        output_csv_path: Optional explicit path for the output CSV file.
                         When omitted, saves to
                         <data_root>/annotations/<pose_name>/<pose_name>_raw.csv
    """
    abs_input_path = os.path.abspath(raw_pose_path)
    pose_name      = os.path.basename(abs_input_path)

    csv_filepath = output_csv_path or _resolve_output_path(abs_input_path, pose_name)
    os.makedirs(os.path.dirname(os.path.abspath(csv_filepath)), exist_ok=True)

    model_path = DEFAULT_MODEL_PATH
    _ensure_model(model_path)

    image_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )

    print(f"Starting extraction for: {pose_name}")
    print(f"Output CSV: {csv_filepath}")
    print(f"Default orientation: {'left-facing' if DEFAULT_IS_LEFT_FACING else 'right-facing'}")

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
                    ext      = filename.lower().rsplit(".", 1)[-1]
                    source_id = filename

                    if ext in {"mp4", "mov", "avi"}:
                        rows = _process_video(
                            filepath, source_id, class_name, model_path
                        )
                        for row in rows:
                            writer.writerow(row)

                        # Summarise orientations detected across all frames
                        if rows:
                            left_count  = sum(1 for r in rows if r[3] == 1)
                            right_count = len(rows) - left_count
                            print(
                                f"  [Video] {filename} -> {len(rows)} frames extracted "
                                f"(right-facing: {right_count}, left-facing: {left_count})"
                            )
                        else:
                            print(f"  [Video] {filename} -> 0 frames extracted (no pose detected)")

                    elif ext in {"jpg", "jpeg", "png"}:
                        row = _process_image(
                            filepath, source_id, class_name, image_landmarker
                        )
                        if row:
                            writer.writerow(row)
                            orientation_tag = "left-facing" if row[3] == 1 else "right-facing"
                            print(
                                f"  [Image/{orientation_tag}] {filename} -> 1 frame extracted."
                            )

    print(f"\nExtraction complete. Data saved to: {csv_filepath}")


# 
# CLI
# 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks to CSV for chair pose (Utkatasana)."
    )
    parser.add_argument(
        "pose_folder",
        nargs="?",
        default="../../data/raw/chair_pose",
        help="Path to the raw pose folder (default: ../../data/raw/chair_pose)",
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