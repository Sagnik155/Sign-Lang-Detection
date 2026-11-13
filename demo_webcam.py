import argparse
import collections
import json
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

import config
from preprocessing import preprocess_sequence

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_TASK_PATH = os.path.join(config.ARTIFACTS_DIR, "hand_landmarker.task")

# Standard 21-point MediaPipe hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky & palm
]


class MediaPipeHandExtractor:
    """
    Wrapper supporting both MediaPipe Tasks HandLandmarker (current API)
    and legacy mp.solutions.hands, returning (42, 3) landmark arrays.
    """

    def __init__(self):
        self.use_tasks_api = False
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not os.path.exists(HAND_LANDMARKER_TASK_PATH):
                print("Downloading MediaPipe hand_landmarker.task model...")
                urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_TASK_PATH)

            base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_TASK_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
            self.timestamp_ms = 0

    def extract(self, rgb_frame: np.ndarray, swap_handedness: bool = False) -> np.ndarray:
        """
        Extracts a (42, 3) array: rows 0..20 = Left hand, rows 21..41 = Right hand.
        Missing hands remain zeros[cite: 3].
        """
        frame_landmarks = np.zeros((42, 3), dtype=np.float32)

        if not self.use_tasks_api:
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    if swap_handedness:
                        label = "Right" if label == "Left" else "Left"

                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32
                    )
                    if label == "Left":
                        frame_landmarks[:21] = coords
                    else:
                        frame_landmarks[21:] = coords
        else:
            self.timestamp_ms += 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

            if results.hand_landmarks and results.handedness:
                for hand_lms, handedness_list in zip(
                    results.hand_landmarks, results.handedness
                ):
                    label = handedness_list[0].category_name  # "Left" or "Right"
                    if swap_handedness:
                        label = "Right" if label == "Left" else "Left"

                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32
                    )
                    if label == "Left":
                        frame_landmarks[:21] = coords
                    else:
                        frame_landmarks[21:] = coords

        return frame_landmarks

    def close(self) -> None:
        if not self.use_tasks_api:
            self.hands.close()
        else:
            self.landmarker.close()


def draw_landmarks_on_frame(bgr_frame: np.ndarray, landmarks_42: np.ndarray) -> None:
    """Draws extracted left (green) and right (cyan) hand skeletons onto the display frame."""
    h, w, _ = bgr_frame.shape
    hand_configs = [
        (landmarks_42[:21], (0, 255, 0)),    # Left hand: green
        (landmarks_42[21:], (255, 200, 0)),  # Right hand: cyan
    ]

    for hand_pts, color in hand_configs:
        if not np.any(hand_pts != 0.0):
            continue

        pixel_pts = []
        for x, y, _ in hand_pts:
            px, py = int(x * w), int(y * h)
            pixel_pts.append((px, py))
            cv2.circle(bgr_frame, (px, py), 4, color, -1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(bgr_frame, pixel_pts[start_idx], pixel_pts[end_idx], color, 2)


def run_webcam_demo(camera_index: int = 0, enable_tts: bool = False) -> None:
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_SAVE_PATH}. Run train.py first."
        )

    with open(config.CHOSEN_LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}

    tts_engine = None
    if enable_tts:
        try:
            import pyttsx3

            tts_engine = pyttsx3.init()
        except Exception as exc:
            print(f"Warning: Could not initialize pyttsx3 TTS ({exc}). Continuing without audio.")

    print(f"Loading trained LSTM model from {config.MODEL_SAVE_PATH}...")
    model = keras.models.load_model(config.MODEL_SAVE_PATH)
    extractor = MediaPipeHandExtractor()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}.")

    # Rolling 32-frame buffer and prediction smoothing queue[cite: 6, 7]
    frame_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
    recent_predictions = collections.deque(maxlen=config.SMOOTHING_WINDOW)

    mirror_view = True
    swap_hands = False
    frame_counter = 0

    raw_top_label = "..."
    raw_top_conf = 0.0
    smoothed_label = "Waiting for sign..."
    last_spoken_label = None

    print("\nWebcam Demo Controls:")
    print("  [q] Quit")
    print("  [m] Toggle horizontal webcam mirror")
    print("  [s] Toggle Left/Right hand label swap")
    print("  [c] Clear rolling frame buffer\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mirror_view:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks_42 = extractor.extract(rgb_frame, swap_handedness=swap_hands)
            draw_landmarks_on_frame(frame, landmarks_42)

            hand_detected = np.any(landmarks_42 != 0.0)
            if hand_detected:
                frame_buffer.append(landmarks_42)
            elif len(frame_buffer) > 0:
                # Append zero frame when hand leaves view briefly
                frame_buffer.append(landmarks_42)

            frame_counter += 1

            # Run inference once buffer is full, every N frames[cite: 6, 7]
            if (
                len(frame_buffer) == config.SEQUENCE_LENGTH
                and frame_counter % config.PREDICT_EVERY_N_FRAMES == 0
            ):
                seq_array = np.array(frame_buffer, dtype=np.float32)  # (32, 42, 3)
                processed_seq = preprocess_sequence(seq_array)        # (32, 126)
                input_batch = np.expand_dims(processed_seq, axis=0)   # (1, 32, 126)

                probs = model.predict(input_batch, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                raw_top_label = idx_to_label[best_idx]
                raw_top_conf = float(probs[best_idx])

                if raw_top_conf >= config.CONFIDENCE_THRESHOLD:
                    recent_predictions.append(raw_top_label)
                else:
                    recent_predictions.append(None)

                # Confirm prediction only when all SMOOTHING_WINDOW slots match[cite: 7]
                if (
                    len(recent_predictions) == config.SMOOTHING_WINDOW
                    and recent_predictions[0] is not None
                    and all(p == recent_predictions[0] for p in recent_predictions)
                ):
                    smoothed_label = recent_predictions[0]
                    if tts_engine is not None and smoothed_label != last_spoken_label:
                        tts_engine.say(smoothed_label)
                        tts_engine.runAndWait()
                        last_spoken_label = smoothed_label

            # Draw HUD overlay
            cv2.rectangle(frame, (10, 10), (620, 135), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"Confirmed Sign: {smoothed_label.upper()}",
                (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 120),
                2,
            )
            cv2.putText(
                frame,
                f"Raw Top-1: {raw_top_label} ({raw_top_conf*100:.1f}%) | Buffer: {len(frame_buffer)}/{config.SEQUENCE_LENGTH}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                frame,
                f"Mirror [m]: {mirror_view} | Swap Hands [s]: {swap_hands} | Quit [q]",
                (20, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 200, 255),
                1,
            )

            cv2.imshow("ASL Sign Recognition (Stacked LSTM)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("m"):
                mirror_view = not mirror_view
            elif key == ord("s"):
                swap_hands = not swap_hands
                frame_buffer.clear()
                recent_predictions.clear()
            elif key == ord("c"):
                frame_buffer.clear()
                recent_predictions.clear()
                smoothed_label = "Waiting for sign..."
                last_spoken_label = None

    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Webcam ASL Sign Recognition Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument(
        "--tts", action="store_true", help="Enable text-to-speech output via pyttsx3"
    )
    args = parser.parse_args()
    run_webcam_demo(camera_index=args.camera, enable_tts=args.tts)