import os

# ==============================================================================
# 1. DIRECTORY & FILE PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kaggle dataset directory (update if your raw data sits outside Sign-lang/data)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "asl-signs")
TRAIN_CSV_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
LABEL_MAP_JSON_PATH = os.path.join(RAW_DATA_DIR, "sign_to_prediction_index_map.json")

# Generated artifacts directory
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PROCESSED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "processed_data.npz")
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "best_lstm_model.keras")
BASELINE_MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "best_dense_baseline.keras")
CHOSEN_LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, "label_map.json")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================================================================
# 2. CHOSEN SIGNS SUBSET (20 Hand-Dominant ASL Signs)
# ==============================================================================
# Selected because they are common in the Kaggle asl-signs dataset and rely
# primarily on handshape/hand movement rather than facial expressions[cite: 2].
CHOSEN_SIGNS = [
    # 11 proven high-F1 signs kept from your previous run
    "water",
    "icecream",
    "shhh",
    "hello",
    "listen",
    "look",
    "food",
    "pizza",
    "yes",
    "book",
    "no",
    # 9 high-frequency, visually distinct handshape/motion signs from your top 50
    "drink",       # C-hand tilting upward (400 clips)
    "bye",         # Open-palm wave (402 clips)
    "bird",        # Index-thumb beak pinching motion (404 clips)
    "cow",         # Y-handshape twist (404 clips)
    "cat",         # F-handshape whisker pull (400 clips)
    "frog",        # V-handshape flicking outward (396 clips)
    "owl",         # Two-handed O-shapes rotating (396 clips)
    "balloon",     # Two-handed expanding sphere (393 clips)
    "up",          # Single index finger pointing/moving upward (394 clips)
]
NUM_CLASSES = len(CHOSEN_SIGNS)

# ==============================================================================
# 3. SEQUENCE & LANDMARK DIMENSIONS
# ==============================================================================
SEQUENCE_LENGTH = 32          # Fixed frames per clip via resampling[cite: 2, 3]
NUM_HANDS = 2                 # Left and right hand[cite: 2, 3]
LANDMARKS_PER_HAND = 21       # MediaPipe hand landmark count[cite: 1, 3]
COORDS_PER_LANDMARK = 3       # x, y, z[cite: 1, 3]
FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 126[cite: 3]

# MediaPipe Holistic Landmark Row Indices in Kaggle Parquet Files
# Total 543 landmarks per frame: Face (0-467), Left Hand (468-488),
# Pose (489-521), Right Hand (522-542)[cite: 2].
LEFT_HAND_START = 468
LEFT_HAND_END = 489           # Exclusive (21 landmarks)
RIGHT_HAND_START = 522
RIGHT_HAND_END = 543          # Exclusive (21 landmarks)

# Key landmark indices within a single 21-point hand
WRIST_IDX = 0
MIDDLE_FINGER_MCP_IDX = 9     # Base of middle finger for scale normalization[cite: 3]

# ==============================================================================
# 4. SPLITTING & REPRODUCIBILITY
# ==============================================================================
RANDOM_SEED = 42              # Fixed seed for NumPy and TensorFlow[cite: 5]
TRAIN_RATIO = 0.70            # ~70% of signers[cite: 3]
VAL_RATIO = 0.15              # ~15% of signers[cite: 3]
TEST_RATIO = 0.15             # ~15% of signers[cite: 3]

# ==============================================================================
# 5. TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 64               # Starting batch size[cite: 4]
EPOCHS = 50                   # Up to 50 epochs[cite: 4]
LEARNING_RATE = 1e-3          # Adam optimizer starting learning rate[cite: 4]
DROPOUT_RATE = 0.3            # Dropout rate between LSTM layers[cite: 4]
EARLY_STOPPING_PATIENCE = 8   # Patience for EarlyStopping callback[cite: 4]
REDUCE_LR_PATIENCE = 4        # Patience for ReduceLROnPlateau callback[cite: 4]
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# ==============================================================================
# 6. DEMO SETTINGS
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.65   # Minimum softmax probability to display prediction[cite: 7]
SMOOTHING_WINDOW = 5          # Consecutive identical frames required in webcam demo[cite: 7]
PREDICT_EVERY_N_FRAMES = 3    # Run inference every N frames for smooth FPS