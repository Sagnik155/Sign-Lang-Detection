import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def build_lstm_model(
    seq_length: int = config.SEQUENCE_LENGTH,
    num_features: int = config.FEATURES_PER_FRAME,
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """
    Builds the primary stacked LSTM classifier for isolated ASL sign sequences[cite: 4].
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_length, num_features), name="landmark_sequence"),
            layers.Masking(mask_value=0.0, name="mask_missing_frames"),
            layers.LSTM(128, return_sequences=True, name="lstm_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            layers.LSTM(64, return_sequences=False, name="lstm_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),
            layers.Dense(64, activation="relu", name="dense_hidden"),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="asl_stacked_lstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dense_baseline(
    seq_length: int = config.SEQUENCE_LENGTH,
    num_features: int = config.FEATURES_PER_FRAME,
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """
    Builds a plain Dense network on the flattened sequence as a baseline
    to quantify the value added by recurrent temporal modeling[cite: 4].
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_length, num_features), name="landmark_sequence"),
            layers.Flatten(name="flatten_sequence"),
            layers.Dense(256, activation="relu", name="dense_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            layers.Dense(128, activation="relu", name="dense_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="asl_dense_baseline",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model