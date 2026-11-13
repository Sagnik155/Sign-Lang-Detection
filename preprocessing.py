import numpy as np
import config


def clean_landmarks(sequence: np.ndarray) -> np.ndarray:
    """
    Replaces NaN or infinite values (when a hand is not detected in a frame)
    with zeros[cite: 2, 3].

    Args:
        sequence: Array of shape (num_frames, 42, 3)

    Returns:
        Cleaned float32 array of shape (num_frames, 42, 3) with no NaNs[cite: 2].
    """
    return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _normalize_single_hand(hand_seq: np.ndarray) -> np.ndarray:
    """
    Normalizes a single hand sequence of shape (num_frames, 21, 3):
      1. Shifts all 21 landmarks relative to the wrist (landmark 0)[cite: 3].
      2. Scales coordinates by the wrist-to-middle-finger-MCP distance
         (landmark 0 to landmark 9) so coordinates are position- and
         size-independent[cite: 2, 3].
      3. Preserves all-zero frames (undetected hands) as zeros[cite: 3].
    """
    num_frames = hand_seq.shape[0]
    normalized = np.zeros_like(hand_seq, dtype=np.float32)

    # Identify frames where the hand was actually detected (non-zero landmarks)
    hand_present = np.any(hand_seq != 0.0, axis=(1, 2))
    if not np.any(hand_present):
        return normalized

    present_frames = hand_seq[hand_present]  # (N_present, 21, 3)

    # 1. Translate so wrist (index 0) is at the origin (0, 0, 0)[cite: 3]
    wrist = present_frames[:, config.WRIST_IDX : config.WRIST_IDX + 1, :]  # (N_present, 1, 3)
    centered = present_frames - wrist

    # 2. Compute hand scale: Euclidean distance from wrist (0) to middle finger base (9)[cite: 3]
    middle_mcp = centered[:, config.MIDDLE_FINGER_MCP_IDX, :]  # (N_present, 3)
    scale = np.linalg.norm(middle_mcp, axis=-1, keepdims=True)  # (N_present, 1)

    # Prevent division by zero if landmarks collapse on a noisy frame
    scale = np.where(scale < 1e-6, 1.0, scale)[:, :, np.newaxis]  # (N_present, 1, 1)

    normalized[hand_present] = centered / scale
    return normalized


def normalize_hands(sequence: np.ndarray) -> np.ndarray:
    """
    Normalizes both left and right hands independently and flattens each frame
    to 126 features[cite: 3].

    Args:
        sequence: Array of shape (num_frames, 42, 3), where indices 0..20 are
                  the left hand and indices 21..41 are the right hand[cite: 2].

    Returns:
        Normalized float32 array of shape (num_frames, 126)[cite: 3].
    """
    seq = clean_landmarks(sequence)
    num_frames = seq.shape[0]

    left_hand = seq[:, : config.LANDMARKS_PER_HAND, :]   # (T, 21, 3)
    right_hand = seq[:, config.LANDMARKS_PER_HAND :, :]  # (T, 21, 3)

    left_norm = _normalize_single_hand(left_hand)
    right_norm = _normalize_single_hand(right_hand)

    combined = np.concatenate([left_norm, right_norm], axis=1)  # (T, 42, 3)
    return combined.reshape(num_frames, config.FEATURES_PER_FRAME)  # (T, 126)


def resample_frames(
    sequence: np.ndarray, target_len: int = config.SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Resamples a variable-length sequence to a fixed number of frames (default 32)
    using linear interpolation along the time axis[cite: 2, 3].

    Args:
        sequence: Array of shape (num_frames, 126)
        target_len: Target sequence length (32)[cite: 2, 3]

    Returns:
        Resampled float32 array of shape (target_len, 126)[cite: 2].
    """
    num_frames, num_features = sequence.shape

    if num_frames == 0:
        return np.zeros((target_len, num_features), dtype=np.float32)

    if num_frames == target_len:
        return sequence.astype(np.float32)

    if num_frames == 1:
        return np.repeat(sequence, target_len, axis=0).astype(np.float32)

    # Interpolate each feature channel across normalized time [0, 1][cite: 3]
    old_indices = np.linspace(0.0, 1.0, num_frames)
    new_indices = np.linspace(0.0, 1.0, target_len)

    resampled = np.empty((target_len, num_features), dtype=np.float32)
    for feat_idx in range(num_features):
        resampled[:, feat_idx] = np.interp(
            new_indices, old_indices, sequence[:, feat_idx]
        )

    return resampled


def preprocess_sequence(raw_hands: np.ndarray) -> np.ndarray:
    """
    End-to-end preprocessing entry point used identically by data_prep.py,
    demo_offline.py, and demo_webcam.py[cite: 3, 6].

    Args:
        raw_hands: Array of shape (num_frames, 42, 3) containing raw left + right
                   hand landmarks[cite: 2].

    Returns:
        Preprocessed float32 array of shape (32, 126)[cite: 2].
    """
    normalized = normalize_hands(raw_hands)
    fixed_length = resample_frames(normalized, target_len=config.SEQUENCE_LENGTH)
    return fixed_length