import argparse
import json
import os
import numpy as np
from tensorflow import keras

import config
from data_prep import load_hand_landmarks_from_parquet
from dataset import load_and_split_data
from preprocessing import preprocess_sequence


def run_offline_demo(num_samples: int = 10, seed: int = config.RANDOM_SEED) -> None:
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_SAVE_PATH}. Run train.py first."
        )

    with open(config.CHOSEN_LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}

    # Load held-out test split (unseen signers only)[cite: 3, 6]
    splits = load_and_split_data(augment_multiplier=0)
    _, y_test, p_test, paths_test = splits["test"]

    if len(y_test) == 0:
        raise RuntimeError("No held-out test samples found.")

    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, len(y_test))
    chosen_indices = rng.choice(len(y_test), size=sample_count, replace=False)

    print(f"Loading trained LSTM model from {config.MODEL_SAVE_PATH}...")
    model = keras.models.load_model(config.MODEL_SAVE_PATH)

    print("\n" + "=" * 82)
    print(f"OFFLINE DEMO: PREDICTING ON {sample_count} HELD-OUT TEST CLIPS (UNSEEN SIGNERS)")
    print("=" * 82)

    correct_top1 = 0
    correct_top3 = 0

    for idx_num, test_idx in enumerate(chosen_indices, start=1):
        rel_path = str(paths_test[test_idx])
        parquet_path = os.path.join(config.RAW_DATA_DIR, rel_path)
        true_label = idx_to_label[int(y_test[test_idx])]
        signer_id = int(p_test[test_idx])

        # Run full raw -> shared preprocessing -> inference pipeline[cite: 6]
        raw_hands = load_hand_landmarks_from_parquet(parquet_path)
        processed_seq = preprocess_sequence(raw_hands)  # (32, 126)
        input_batch = np.expand_dims(processed_seq, axis=0)  # (1, 32, 126)

        probs = model.predict(input_batch, verbose=0)[0]
        top3_indices = np.argsort(probs)[::-1][:3]

        pred_label = idx_to_label[int(top3_indices[0])]
        pred_conf = float(probs[top3_indices[0]])
        top3_str = ", ".join(
            [f"{idx_to_label[int(i)]} ({probs[i]*100:.1f}%)" for i in top3_indices]
        )

        is_top1 = pred_label == true_label
        is_top3 = true_label in [idx_to_label[int(i)] for i in top3_indices]
        correct_top1 += int(is_top1)
        correct_top3 += int(is_top3)

        status = "PASS" if is_top1 else ("TOP-3" if is_top3 else "MISS")
        print(f"[{idx_num:02d}] Signer: {signer_id} | File: {rel_path}")
        print(
            f"     True: {true_label:<10} | Predicted: {pred_label:<10} "
            f"({pred_conf*100:5.1f}%) | Status: {status}"
        )
        print(f"     Top-3: {top3_str}")
        print("-" * 82)

    print(
        f"Sample Summary -> Top-1: {correct_top1}/{sample_count} "
        f"({correct_top1/sample_count*100:.1f}%) | "
        f"Top-3: {correct_top3}/{sample_count} ({correct_top3/sample_count*100:.1f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline ASL Sign Recognition Demo")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random held-out test clips to evaluate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for sampling test clips",
    )
    args = parser.parse_args()
    run_offline_demo(num_samples=args.samples, seed=args.seed)