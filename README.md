# Isolated ASL Recognition with Stacked LSTM Networks

A lightweight, real-time, signer-independent system that recognizes 20 isolated American Sign Language (ASL) signs from hand-landmark sequences and translates them into English words (with optional speech output).

**Course:** PCCCS582 — Neural Network & Deep Learning
**UN SDG Alignment:** SDG 10 — Reduced Inequalities

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
![Test Accuracy](https://img.shields.io/badge/Top--1%20Acc-71.19%25-brightgreen)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Tech Stack & Dataset](#2-tech-stack--dataset)
3. [Vocabulary & Preprocessing](#3-vocabulary--preprocessing)
4. [Model Architecture](#4-model-architecture)
5. [Results](#5-results)
6. [Repository Structure](#6-repository-structure)
7. [Getting Started](#7-getting-started)
8. [Limitations & Future Work](#8-limitations--future-work)

---

## 1. Overview

Communication barriers between Deaf/Hard-of-Hearing communities and hearing individuals remain a major obstacle in classrooms, healthcare, public services, and workplaces. Human interpreters are often unavailable on demand, while raw-video deep learning models need heavy GPU compute and break down under changing backgrounds, lighting, or camera distance.

This project separates spatial perception from temporal modeling instead of feeding raw RGB frames into a 3D-CNN or Vision Transformer:

1. **Spatial feature extraction (pretrained):** MediaPipe Hands extracts 21 three-dimensional keypoints `(x, y, z)` per hand per frame. This removes background clutter, skin-tone bias, and lighting sensitivity, and reduces each frame to a compact **126-element** vector.
2. **Temporal sequence modeling (Stacked LSTM):** A two-layer LSTM learns how hand shapes and trajectories evolve across a fixed **32-frame** window to classify **20 everyday ASL signs**.

### SDG 10 — Reduced Inequalities

The pipeline runs smoothly on a standard CPU (e.g., a laptop webcam, no dedicated GPU), showing how accessible, low-latency assistive AI can promote equal access to education, digital interfaces, and public services.

---

## 2. Tech Stack & Dataset

| Category | Technology | Role |
| :--- | :--- | :--- |
| Language | Python 3.10+ | Core development |
| Deep Learning | TensorFlow / Keras | Model construction, masking, training callbacks, inference |
| Landmark Extraction | MediaPipe Hands (`mediapipe`) | 21-point 3D hand skeleton for live inference |
| Computer Vision | OpenCV (`opencv-python`) | Webcam capture, mirroring, HUD, skeleton drawing |
| Data Processing | NumPy, Pandas, PyArrow | Reading `.parquet` landmarks, normalization, resampling |
| Evaluation & Plots | Scikit-learn, Matplotlib, Seaborn | Metrics, confusion matrices, training curves |
| Text-to-Speech (optional) | `pyttsx3` | Offline speech for confirmed live predictions |
| Dataset | Kaggle GISLR (`asl-signs`) | Pre-extracted MediaPipe landmark sequences from 21 participants |

---

## 3. Vocabulary & Preprocessing

### 3.1 Curated 20-Sign Vocabulary

A balanced subset (~393–415 clips per class) of high-frequency, hand-centric signs that are distinct in finger articulation and motion, without relying heavily on facial expression or torso contact:

`water`, `icecream`, `shhh`, `hello`, `listen`, `look`, `food`, `pizza`, `yes`, `book`, `no`, `drink`, `bye`, `bird`, `cow`, `cat`, `frog`, `owl`, `balloon`, `up`

### 3.2 Shared Preprocessing (`preprocessing.py`)

Training and live webcam inference use the **same module**, which prevents training-serving skew.

1. **Landmark extraction & NaN cleaning** — Left-hand rows (`0..20`) and right-hand rows (`21..41`) give `42 × 3 = 126` features per frame. Missing hands are zero-filled.
2. **Wrist-centered translation invariance** — The wrist (index 0) is subtracted from all 21 landmarks of each detected hand, so coordinates don't depend on where the signer stands.
3. **Hand-scale normalization** — Coordinates are divided by the distance between the wrist (index 0) and middle-finger MCP (index 9), so near and far hands share the same scale.
4. **Fixed-length temporal resampling** — Variable-length clips are linearly interpolated to **32 frames**, giving a `(32, 126)` tensor per clip.

### 3.3 Signer-Grouped Splitting & Augmentation (`dataset.py`)

- **Strict split by `participant_id`** (~70% train / 15% val / 15% test) to prevent leakage, where the model memorizes a person's body proportions instead of the sign. The three test participants (`4718`, `28656`, `36257`) are never seen during training.
- **Training-only augmentation:** random time-stretching, spatial scaling (±10%), Z-axis rotation (±15°), Gaussian coordinate jitter, and random frame dropout.
- **Horizontal mirroring is intentionally omitted** to preserve left/right hand semantics.

---

## 4. Model Architecture

### 4.1 Stacked LSTM Classifier

```text
Input Sequence: (32, 126)
       │
       ▼
Masking(mask_value=0.0)           ── Ignores zero-padded frames where no hands appear
       │
       ▼
LSTM(128, return_sequences=True)  ── Captures fine-grained frame-to-frame finger transitions
       │
       ▼
Dropout(0.3)                      ── Regularizes against signer overfitting
       │
       ▼
LSTM(64, return_sequences=False)  ── Compresses the 32-frame trajectory into a context vector
       │
       ▼
Dropout(0.3)                      ── Prevents co-adaptation before classification
       │
       ▼
Dense(64, activation='relu')      ── Non-linear feature projection
       │
       ▼
Dense(20, activation='softmax')   ── Probability distribution over the 20 signs
```

### 4.2 Design Rationale

- **Landmarks + LSTM instead of 3D-CNNs.** Raw-video models need millions of parameters and high-end GPUs, and overfit to backgrounds and lighting. Normalized 3D landmarks cut input dimensionality by more than 99% and enable real-time CPU inference.
- **Stacked LSTM instead of a flat Dense network.** A sign is defined by temporal order. Flattening `(32, 126)` into 4,032 values destroys that order. Stacking creates a temporal hierarchy: the first layer (128 units) models short sub-movements, and the second (64 units) aggregates them into a full gesture representation.
- **Unidirectional instead of Bidirectional.** In ablations, the Bidirectional variant had more capacity to overfit to the 15 training signers, while the leaner unidirectional model generalized better to unseen signers. It also maps naturally to a causal, left-to-right streaming buffer in live inference.
- **Explicit `Masking(0.0)`.** One-handed signs leave the inactive hand's 63 features at `0.0`, and dropped detections yield all-zero frames. Masking keeps these timesteps from polluting hidden-state updates.

---

## 5. Results

All models were evaluated on **1,097 held-out test clips** from 3 completely unseen signers.

### 5.1 Architecture & Feature Ablations

| Model / Experiment | Input Shape | Top-1 Acc | Top-3 Acc | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dense Baseline (flattened MLP, initial 20 signs) | (32, 126) | 46.84% | 71.17% | 44.72% | 45.69% |
| Stacked LSTM + Motion Deltas (initial 20 signs) | (32, 252) | 50.48% | 76.92% | 48.10% | 48.87% |
| Bidirectional Stacked LSTM (initial 20 signs) | (32, 126) | 53.74% | 78.54% | 52.38% | 53.18% |
| Unidirectional Stacked LSTM (initial 20 signs) | (32, 126) | 57.28% | 79.69% | 55.59% | 56.50% |
| Dense Baseline (final curated 20 signs) | (32, 126) | 53.69% | 77.58% | 53.71% | 53.91% |
| **Stacked LSTM — Final Model (curated 20 signs)** | **(32, 126)** | **71.19%** | **87.33%** | **71.06%** | **71.18%** |

**Key findings**

- **Temporal modeling gain:** On the final vocabulary, the Stacked LSTM beats the Dense baseline by **+17.50%** Top-1 (71.19% vs. 53.69%) and **+17.35%** Macro F1, showing that recurrent modeling is essential for dynamic gestures.
- **Vocabulary curation:** Removing face-anchored signs (whose meaning depends on facial contact points that hand-only landmarks can't capture) while keeping normalized `(32, 126)` spatial features raised Top-1 accuracy from 57.28% to 71.19%.

### 5.2 Per-Signer Generalization

| Unseen Participant | Test Clips | Top-1 Accuracy |
| :--- | :---: | :---: |
| 4718 | 303 | 63.37% |
| 28656 | 389 | 77.38% |
| 36257 | 405 | 71.11% |
| **Overall held-out test set** | **1,097** | **71.19%** |

### 5.3 Per-Class F1 (Stacked LSTM)

- **Strongest signs:** cow (0.85), shhh (0.84), frog (0.84), icecream (0.78), no (0.77), bye (0.77), food (0.76), pizza (0.76), listen (0.75), hello (0.74), up (0.71)
- **Main confusions:** `cat → water` (16 errors) and `balloon → water` (15 errors), largely due to shared open/pinched finger configurations as the hand enters the frame.

---

## 6. Repository Structure

```text
Sign-lang/
├── config.py              # Paths, 20 chosen signs, sequence dimensions (32x126), seeds, hyperparameters
├── preprocessing.py       # Shared NaN handling, wrist/scale normalization, 32-frame resampling
├── data_prep.py           # Filters train.csv, extracts .parquet landmarks, caches processed_data.npz
├── dataset.py             # Signer-grouped train/val/test split (~70/15/15) and training-only augmentation
├── model.py               # Stacked LSTM architecture and flattened Dense baseline
├── train.py               # Training loop with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
├── evaluate.py            # Held-out evaluation, per-signer metrics, confusion matrix, learning curves
├── demo_offline.py        # Replay and inference on unseen test .parquet clips
├── demo_webcam.py         # Real-time webcam demo (OpenCV + MediaPipe + 32-frame buffer + smoothing)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/
│   └── asl-signs/         # Raw dataset (train.csv and train_landmark_files/)
└── artifacts/
    ├── best_lstm_model.keras        # Best Stacked LSTM (71.19% Top-1)
    ├── best_dense_baseline.keras    # Best Dense baseline (53.69% Top-1)
    ├── processed_data.npz           # Cached preprocessed sequences
    ├── label_map.json               # Class index <-> sign gloss mapping
    ├── signer_splits.json           # Participant IDs per split
    ├── training_history.json        # Epoch-by-epoch loss and accuracy
    ├── test_metrics_summary.json    # Full evaluation metrics export
    └── plots/
        ├── confusion_matrix.png     # 20x20 test-set confusion matrix
        └── training_curves.png      # Train vs. validation loss and accuracy
```

---

## 7. Getting Started

### 7.1 Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the Kaggle GISLR dataset (`asl-signs`) and place it in `data/asl-signs/`.

### 7.2 Run the Pipeline

```bash
# 1. Prepare and cache the dataset
python data_prep.py

# 2. Train the Stacked LSTM and Dense baseline
python train.py

# 3. Evaluate on unseen test signers and generate plots
python evaluate.py

# 4. Offline demo on held-out test clips
python demo_offline.py --samples 10

# 5. Real-time webcam demo
python demo_webcam.py

# Optional: spoken output for confirmed predictions
python demo_webcam.py --tts
```

### 7.3 Webcam Controls (`demo_webcam.py`)

| Key | Action |
| :---: | :--- |
| `q` | Quit the webcam window |
| `m` | Toggle horizontal mirror view |
| `s` | Swap Left/Right hand slots (useful if mirroring flips MediaPipe handedness labels) |
| `c` | Clear the rolling 32-frame buffer and reset the confirmed sign display |

---

## 8. Limitations & Future Work

- **Hand-only representation.** Only the 42 hand landmarks are used (no face mesh or upper-body pose), so signs that depend on exact facial or chest contact points remain ambiguous. Adding facial anchor points (chin, mouth, forehead, ear) should help separate similar handshapes.
- **Isolated vs. continuous signing.** The 32-frame sliding buffer targets single-word signs. Continuous ASL sentences would require temporal segmentation (e.g., CTC loss) and grammar translation.
- **Webcam vs. smartphone domain shift.** Training clips were captured mostly on front-facing phones, while live inference runs on laptop webcams with different focal lengths and frame rates. Fine-tuning on a small calibration set recorded on the target webcam should improve live responsiveness.
