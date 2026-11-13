import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from dataset import make_tf_datasets
from model import build_lstm_model, build_dense_baseline


def set_reproducibility_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Fixes random seeds across Python, NumPy, and TensorFlow[cite: 5]."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_callbacks(save_path: str) -> list:
    """Creates EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks[cite: 4]."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_pipeline() -> None:
    set_reproducibility_seeds()

    print("Loading and splitting dataset by signer...")
    train_ds, val_ds, _, splits = make_tf_datasets(augment_multiplier=2)

    signers_info = splits["signers"]
    print(f"Train signers ({len(signers_info['train'])}): {signers_info['train']}")
    print(f"Val signers   ({len(signers_info['val'])}): {signers_info['val']}")
    print(f"Test signers  ({len(signers_info['test'])}): {signers_info['test']}")

    # Save exact signer split details for reporting and evaluation[cite: 3, 5]
    split_record_path = os.path.join(config.ARTIFACTS_DIR, "signer_splits.json")
    with open(split_record_path, "w", encoding="utf-8") as f:
        json.dump(signers_info, f, indent=2)

    # 1. Train Primary Stacked LSTM Model[cite: 4, 6]
    print("\n" + "=" * 60)
    print("Training Stacked LSTM Model...")
    print("=" * 60)
    lstm_model = build_lstm_model()
    lstm_model.summary()

    lstm_history = lstm_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(config.MODEL_SAVE_PATH),
        verbose=1,
    )

    # 2. Train Flattened Dense Baseline Model[cite: 4, 6]
    print("\n" + "=" * 60)
    print("Training Dense Baseline Model...")
    print("=" * 60)
    baseline_model = build_dense_baseline()
    baseline_model.summary()

    baseline_history = baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(config.BASELINE_MODEL_SAVE_PATH),
        verbose=1,
    )

    # Save training histories so evaluate.py can generate loss/accuracy curves[cite: 5, 6]
    history_save_path = os.path.join(config.ARTIFACTS_DIR, "training_history.json")
    history_payload = {
        "lstm": {k: [float(v) for v in vals] for k, vals in lstm_history.history.items()},
        "baseline": {k: [float(v) for v in vals] for k, vals in baseline_history.history.items()},
    }
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    print("\nTraining complete!")
    print(f"  Best LSTM saved to     : {config.MODEL_SAVE_PATH}")
    print(f"  Best Baseline saved to : {config.BASELINE_MODEL_SAVE_PATH}")
    print(f"  Training history saved : {history_save_path}")


if __name__ == "__main__":
    train_pipeline()