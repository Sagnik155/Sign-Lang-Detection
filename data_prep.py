import json
import os
import numpy as np
import pandas as pd

import config
from preprocessing import preprocess_sequence


ROWS_PER_FRAME = 543  # Total MediaPipe Holistic landmarks per frame


def load_hand_landmarks_from_parquet(parquet_path: str) -> np.ndarray:
    """
    Reads a single Kaggle asl-signs parquet file and extracts only the
    21 left-hand and 21 right-hand landmarks (42 points x 3 coords).

    Returns:
        Array of shape (num_frames, 42, 3)
    """
    df = pd.read_parquet(parquet_path, columns=["x", "y", "z"])
    num_frames = len(df) // ROWS_PER_FRAME

    if num_frames == 0:
        return np.zeros(
            (0, config.NUM_HANDS * config.LANDMARKS_PER_HAND, config.COORDS_PER_LANDMARK),
            dtype=np.float32,
        )

    # Reshape flat table into (num_frames, 543, 3)
    landmarks = df.values.reshape(num_frames, ROWS_PER_FRAME, config.COORDS_PER_LANDMARK)

    left_hand = landmarks[:, config.LEFT_HAND_START : config.LEFT_HAND_END, :]
    right_hand = landmarks[:, config.RIGHT_HAND_START : config.RIGHT_HAND_END, :]

    # Concatenate along landmark axis -> (num_frames, 42, 3)
    hands = np.concatenate([left_hand, right_hand], axis=1).astype(np.float32)
    return hands


def build_dataset() -> None:
    if not os.path.exists(config.TRAIN_CSV_PATH):
        raise FileNotFoundError(
            f"Could not find train.csv at {config.TRAIN_CSV_PATH}. "
            "Verify your dataset path in config.py."
        )

    print(f"Reading metadata from {config.TRAIN_CSV_PATH}...")
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)

    # Filter strictly to the chosen subset of signs
    filtered_df = train_df[train_df["sign"].isin(config.CHOSEN_SIGNS)].copy()
    filtered_df.reset_index(drop=True, inplace=True)

    # Create deterministic class index mapping (0 .. NUM_CLASSES - 1)
    label_to_idx = {sign: idx for idx, sign in enumerate(config.CHOSEN_SIGNS)}
    idx_to_label = {str(idx): sign for idx, sign in enumerate(config.CHOSEN_SIGNS)}

    # Save label map alongside artifacts for training, evaluation, and demos
    with open(config.CHOSEN_LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chosen_signs": config.CHOSEN_SIGNS,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
            },
            f,
            indent=2,
        )
    print(f"Saved label map ({len(config.CHOSEN_SIGNS)} signs) to {config.CHOSEN_LABEL_MAP_PATH}")

    total_clips = len(filtered_df)
    print(f"Found {total_clips} sequences across {filtered_df['participant_id'].nunique()} participants.")

    X_list = []
    y_list = []
    participant_list = []
    path_list = []
    skipped = 0

    for i, row in filtered_df.iterrows():
        rel_path = row["path"]
        parquet_full_path = os.path.join(config.RAW_DATA_DIR, rel_path)

        if not os.path.exists(parquet_full_path):
            skipped += 1
            continue

        raw_hands = load_hand_landmarks_from_parquet(parquet_full_path)
        if raw_hands.shape[0] == 0:
            skipped += 1
            continue

        # Apply shared normalization + 32-frame resampling
        processed_seq = preprocess_sequence(raw_hands)  # (32, 126)

        X_list.append(processed_seq)
        y_list.append(label_to_idx[row["sign"]])
        participant_list.append(int(row["participant_id"]))
        path_list.append(rel_path)

        if (i + 1) % 500 == 0 or (i + 1) == total_clips:
            print(f"Processed {i + 1}/{total_clips} clips...")

    X = np.stack(X_list, axis=0).astype(np.float32)               # (N, 32, 126)
    y = np.array(y_list, dtype=np.int32)                          # (N,)
    participant_ids = np.array(participant_list, dtype=np.int32)  # (N,)
    paths = np.array(path_list, dtype=str)                        # (N,)

    np.savez_compressed(
        config.PROCESSED_DATA_PATH,
        X=X,
        y=y,
        participant_ids=participant_ids,
        paths=paths,
    )

    print("\nDataset preparation complete!")
    print(f"  Saved cache : {config.PROCESSED_DATA_PATH}")
    print(f"  X shape     : {X.shape}")
    print(f"  y shape     : {y.shape}")
    print(f"  Skipped     : {skipped}")


if __name__ == "__main__":
    build_dataset()