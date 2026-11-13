import numpy as np
import tensorflow as tf
import config


def split_by_signer(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    paths: np.ndarray,
):
    """
    Splits dataset strictly by participant_id (~70% train / 15% val / 15% test)[cite: 3].
    Ensures no signer appears in more than one split[cite: 3].
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    unique_signers = np.unique(participant_ids)
    shuffled_signers = rng.permutation(unique_signers)

    n_signers = len(shuffled_signers)
    n_train = int(round(n_signers * config.TRAIN_RATIO))
    n_val = int(round(n_signers * config.VAL_RATIO))

    train_signers = shuffled_signers[:n_train]
    val_signers = shuffled_signers[n_train : n_train + n_val]
    test_signers = shuffled_signers[n_train + n_val :]

    train_mask = np.isin(participant_ids, train_signers)
    val_mask = np.isin(participant_ids, val_signers)
    test_mask = np.isin(participant_ids, test_signers)

    splits = {
        "train": (X[train_mask], y[train_mask], participant_ids[train_mask], paths[train_mask]),
        "val": (X[val_mask], y[val_mask], participant_ids[val_mask], paths[val_mask]),
        "test": (X[test_mask], y[test_mask], participant_ids[test_mask], paths[test_mask]),
        "signers": {
            "train": train_signers.tolist(),
            "val": val_signers.tolist(),
            "test": test_signers.tolist(),
        },
    }
    return splits


def _augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Applies training-only augmentations to a single sequence of shape (32, 126)[cite: 3]:
      1. Random time-stretching (resampling along time axis)[cite: 3]
      2. Small random scaling[cite: 3]
      3. Small random 2D rotation in the x-y plane[cite: 3]
      4. Gaussian jitter on non-zero coordinates[cite: 3]
      5. Random frame dropout (zeroing out 1-3 frames)[cite: 3]
    Never mirrors horizontally (to preserve left/right hand semantics)[cite: 3].
    """
    T, F = seq.shape
    aug = seq.reshape(T, 42, 3).copy()

    # Track detected landmarks so zero-padded hands stay zero
    nonzero_mask = np.any(aug != 0.0, axis=-1, keepdims=True)  # (T, 42, 1)

    # 1. Random time-stretching (crop/stretch window and interpolate back to T)
    if rng.random() < 0.5:
        stretch_factor = rng.uniform(0.85, 1.15)
        new_len = max(8, int(round(T * stretch_factor)))
        old_idx = np.linspace(0.0, 1.0, T)
        stretched_idx = np.linspace(0.0, 1.0, new_len)

        flat_aug = aug.reshape(T, F)
        stretched = np.empty((new_len, F), dtype=np.float32)
        for f_idx in range(F):
            stretched[:, f_idx] = np.interp(stretched_idx, old_idx, flat_aug[:, f_idx])

        back = np.empty((T, F), dtype=np.float32)
        for f_idx in range(F):
            back[:, f_idx] = np.interp(old_idx, stretched_idx, stretched[:, f_idx])

        aug = back.reshape(T, 42, 3)
        nonzero_mask = np.any(aug != 0.0, axis=-1, keepdims=True)

    # 2. Small random spatial scaling
    if rng.random() < 0.7:
        scale = rng.uniform(0.90, 1.10)
        aug = aug * scale

    # 3. Small random rotation around z-axis (-15 to +15 degrees)
    if rng.random() < 0.7:
        angle = np.deg2rad(rng.uniform(-15.0, 15.0))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = aug[:, :, 0].copy()
        y = aug[:, :, 1].copy()
        aug[:, :, 0] = cos_a * x - sin_a * y
        aug[:, :, 1] = sin_a * x + cos_a * y

    # 4. Gaussian jitter on coordinates
    if rng.random() < 0.7:
        noise = rng.normal(loc=0.0, scale=0.015, size=aug.shape).astype(np.float32)
        aug = aug + noise

    # Re-apply mask so missing hands remain exact zeros
    aug = np.where(nonzero_mask, aug, 0.0)

    # 5. Random frame dropout (zero out up to 3 random frames)
    if rng.random() < 0.5:
        num_drop = int(rng.integers(1, 4))
        drop_indices = rng.choice(T, size=num_drop, replace=False)
        aug[drop_indices] = 0.0

    return aug.reshape(T, F).astype(np.float32)


def load_and_split_data(augment_multiplier: int = 2):
    """
    Loads cached processed_data.npz, splits by participant_id, and optionally
    augments the training split.

    Args:
        augment_multiplier: Number of augmented copies per training sample[cite: 3].

    Returns:
        Dictionary containing train/val/test arrays and signer lists[cite: 3].
    """
    data = np.load(config.PROCESSED_DATA_PATH, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    participant_ids = data["participant_ids"]
    paths = data["paths"]

    splits = split_by_signer(X, y, participant_ids, paths)
    X_train, y_train, p_train, path_train = splits["train"]

    if augment_multiplier > 0:
        rng = np.random.default_rng(config.RANDOM_SEED)
        aug_X_list = [X_train]
        aug_y_list = [y_train]

        for _ in range(augment_multiplier):
            aug_batch = np.stack([_augment_sequence(seq, rng) for seq in X_train], axis=0)
            aug_X_list.append(aug_batch)
            aug_y_list.append(y_train.copy())

        X_train_final = np.concatenate(aug_X_list, axis=0)
        y_train_final = np.concatenate(aug_y_list, axis=0)
        splits["train"] = (X_train_final, y_train_final, p_train, path_train)

    return splits


def make_tf_datasets(augment_multiplier: int = 2):
    """
    Builds shuffled training tf.data.Dataset and unshuffled val/test datasets[cite: 6].
    """
    splits = load_and_split_data(augment_multiplier=augment_multiplier)
    X_train, y_train, _, _ = splits["train"]
    X_val, y_val, _, _ = splits["val"]
    X_test, y_test, _, _ = splits["test"]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, splits