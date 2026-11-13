import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix,
)

import config
from dataset import load_and_split_data


def plot_training_curves(history_path: str, save_dir: str) -> None:
    """Plots training & validation loss and accuracy curves for LSTM and Baseline[cite: 5]."""
    if not os.path.exists(history_path):
        print(f"Warning: {history_path} not found. Skipping training curves.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        histories = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row_idx, (model_key, title_prefix) in enumerate(
        [("lstm", "Stacked LSTM"), ("baseline", "Dense Baseline")]
    ):
        hist = histories.get(model_key, {})
        if not hist:
            continue

        epochs = range(1, len(hist["loss"]) + 1)

        # Loss subplot
        axes[row_idx, 0].plot(epochs, hist["loss"], label="Train Loss", marker="o")
        axes[row_idx, 0].plot(epochs, hist["val_loss"], label="Val Loss", marker="s")
        axes[row_idx, 0].set_title(f"{title_prefix} - Loss")
        axes[row_idx, 0].set_xlabel("Epoch")
        axes[row_idx, 0].set_ylabel("Sparse Categorical Cross-Entropy")
        axes[row_idx, 0].legend()
        axes[row_idx, 0].grid(True, alpha=0.3)

        # Accuracy subplot
        axes[row_idx, 1].plot(epochs, hist["accuracy"], label="Train Accuracy", marker="o")
        axes[row_idx, 1].plot(epochs, hist["val_accuracy"], label="Val Accuracy", marker="s")
        axes[row_idx, 1].set_title(f"{title_prefix} - Accuracy")
        axes[row_idx, 1].set_xlabel("Epoch")
        axes[row_idx, 1].set_ylabel("Accuracy")
        axes[row_idx, 1].legend()
        axes[row_idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {out_path}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_dir: str
) -> None:
    """Generates and saves the test set confusion matrix heatmap[cite: 5]."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Test Set Confusion Matrix (Held-Out Signers)")
    plt.xlabel("Predicted Sign")
    plt.ylabel("True Sign")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to {out_path}")


def print_top_confused_pairs(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list, top_n: int = 5
) -> None:
    """Prints the most frequently misclassified (True -> Predicted) sign pairs[cite: 5]."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.fill_diagonal(cm, 0)

    pairs = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            count = cm[true_idx, pred_idx]
            if count > 0:
                pairs.append((count, class_names[true_idx], class_names[pred_idx]))

    pairs.sort(reverse=True, key=lambda x: x[0])
    print("\nMost Confused Sign Pairs (True -> Predicted):")
    for count, true_sign, pred_sign in pairs[:top_n]:
        print(f"  {true_sign:12s} -> {pred_sign:12s} : {count} errors")


def evaluate_pipeline() -> None:
    # Load label map
    with open(config.CHOSEN_LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    class_names = label_data["chosen_signs"]

    # Load untouched held-out test signers[cite: 3, 5]
    splits = load_and_split_data(augment_multiplier=0)
    X_test, y_test, p_test, _ = splits["test"]
    print(f"Evaluating on {len(X_test)} held-out test clips across {len(np.unique(p_test))} unseen signers...")

    # Load trained models
    lstm_model = keras.models.load_model(config.MODEL_SAVE_PATH)
    baseline_model = keras.models.load_model(config.BASELINE_MODEL_SAVE_PATH)

    # Predict probabilities
    lstm_probs = lstm_model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
    lstm_preds = np.argmax(lstm_probs, axis=1)

    base_probs = baseline_model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=0)
    base_preds = np.argmax(base_probs, axis=1)

    # Compute core metrics[cite: 5]
    all_labels = list(range(len(class_names)))
    lstm_top1 = accuracy_score(y_test, lstm_preds)
    lstm_top3 = top_k_accuracy_score(y_test, lstm_probs, k=3, labels=all_labels)

    base_top1 = accuracy_score(y_test, base_preds)
    base_top3 = top_k_accuracy_score(y_test, base_probs, k=3, labels=all_labels)

    lstm_report_dict = classification_report(
        y_test, lstm_preds, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0
    )
    base_report_dict = classification_report(
        y_test, base_preds, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0
    )

    # 1. Model Comparison Table[cite: 5]
    print("\n" + "=" * 72)
    print("MODEL COMPARISON TABLE (HELD-OUT TEST SIGNERS)")
    print("=" * 72)
    print(
        f"{'Model':<22} | {'Top-1 Acc':>10} | {'Top-3 Acc':>10} | {'Macro F1':>10} | {'Weighted F1':>11}"
    )
    print("-" * 72)
    print(
        f"{'Dense Baseline':<22} | {base_top1*100:>9.2f}% | {base_top3*100:>9.2f}% | "
        f"{base_report_dict['macro avg']['f1-score']*100:>9.2f}% | "
        f"{base_report_dict['weighted avg']['f1-score']*100:>10.2f}%"
    )
    print(
        f"{'Stacked LSTM (Ours)':<22} | {lstm_top1*100:>9.2f}% | {lstm_top3*100:>9.2f}% | "
        f"{lstm_report_dict['macro avg']['f1-score']*100:>9.2f}% | "
        f"{lstm_report_dict['weighted avg']['f1-score']*100:>10.2f}%"
    )
    print("=" * 72)

    # 2. Detailed Per-Sign Classification Report for Stacked LSTM[cite: 5]
    print("\nSTACKED LSTM CLASSIFICATION REPORT:")
    print(
        classification_report(
            y_test, lstm_preds, labels=all_labels, target_names=class_names, zero_division=0
        )
    )

    # 3. Per-Signer Accuracy on Unseen Test Participants[cite: 5]
    print("PER-SIGNER TEST ACCURACY (Stacked LSTM):")
    per_signer_results = {}
    for signer_id in np.unique(p_test):
        mask = p_test == signer_id
        s_acc = accuracy_score(y_test[mask], lstm_preds[mask])
        per_signer_results[int(signer_id)] = {
            "num_clips": int(np.sum(mask)),
            "accuracy": float(s_acc),
        }
        print(f"  Participant {signer_id:<8} | Clips: {np.sum(mask):<4} | Accuracy: {s_acc*100:.2f}%")

    # 4. Most Confused Sign Pairs[cite: 5]
    print_top_confused_pairs(y_test, lstm_preds, class_names, top_n=5)

    # 5. Save Figures & JSON Summary for Report[cite: 5]
    history_path = os.path.join(config.ARTIFACTS_DIR, "training_history.json")
    plot_training_curves(history_path, config.PLOTS_DIR)
    plot_confusion_matrix(y_test, lstm_preds, class_names, config.PLOTS_DIR)

    metrics_summary_path = os.path.join(config.ARTIFACTS_DIR, "test_metrics_summary.json")
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "lstm": {
                    "top1_accuracy": float(lstm_top1),
                    "top3_accuracy": float(lstm_top3),
                    "classification_report": lstm_report_dict,
                    "per_signer_accuracy": per_signer_results,
                },
                "baseline": {
                    "top1_accuracy": float(base_top1),
                    "top3_accuracy": float(base_top3),
                    "classification_report": base_report_dict,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved full evaluation summary to {metrics_summary_path}")


if __name__ == "__main__":
    evaluate_pipeline()