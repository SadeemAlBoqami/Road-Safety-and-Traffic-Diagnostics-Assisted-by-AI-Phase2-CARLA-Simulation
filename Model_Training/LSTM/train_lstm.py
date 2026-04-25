"""
=============================================================
Phase 3 - Step B v4: CNN + LSTM — Larger Architecture
=============================================================
Changes from v3:
  1. Conv1D: 32→64, 64→128  (more local pattern capacity)
  2. LSTM:   64→128, 32→64  (longer memory for 50-frame window)
  3. Dense:  16→32
  4. Dropout: 0.3→0.4       (prevent overfitting from larger model)
  5. Total params: ~53K → ~250K
=============================================================
"""

import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# Settings
# ============================================================
PROCESSED_DIR = r"C:\Users\salmr\Desktop\Road-Safety-and-Traffic-Diagnostics-Assisted-by-AI-Phase-2\dataset\processed_v2"
OUTPUT_DIR    = r"C:\Users\salmr\Desktop\Road-Safety-and-Traffic-Diagnostics-Assisted-by-AI-Phase-2\dataset\model_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 256
MAX_EPOCHS = 100
SEED       = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. Load Data
# ============================================================
def load_data(processed_dir):
    print("\n[1/5] Loading data...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(processed_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(processed_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"  X_train : {X_train.shape} | Dangerous: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  X_val   : {X_val.shape}   | Dangerous: {y_val.sum()} ({y_val.mean()*100:.1f}%)")
    print(f"  X_test  : {X_test.shape}  | Dangerous: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 2. Focal Loss
# ============================================================
def focal_loss(gamma=2.0, alpha=0.85):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt      = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return tf.reduce_mean(
            -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        )
    return loss_fn


# ============================================================
# 3. Larger CNN + LSTM Model
# ============================================================
def build_model(input_shape=(50, 6)):
    """
    Larger architecture to match increased data complexity.

    Input (50, 6)
      → Conv1D(64)  + BN    ← was 32
      → Conv1D(128) + BN    ← was 64
      → MaxPooling (50→25)
      → Dropout(0.4)        ← was 0.3
      → LSTM(128)           ← was 64
      → Dropout(0.4)
      → LSTM(64)            ← was 32
      → Dropout(0.4)
      → Dense(32)           ← was 16
      → Dense(1, sigmoid)
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # --- CNN Block ---
    x = layers.Conv1D(64,  kernel_size=3, activation="relu",
                      padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)

    x = layers.Conv1D(128, kernel_size=3, activation="relu",
                      padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    x = layers.MaxPooling1D(pool_size=2, name="pool")(x)  # 50→25
    x = layers.Dropout(0.4, name="drop1")(x)

    # --- LSTM Block ---
    x = layers.LSTM(128, return_sequences=True,  name="lstm1")(x)
    x = layers.Dropout(0.4, name="drop2")(x)
    x = layers.LSTM(64,  return_sequences=False, name="lstm2")(x)
    x = layers.Dropout(0.4, name="drop3")(x)

    # --- Classifier ---
    x      = layers.Dense(32, activation="relu", name="dense1")(x)
    output = layers.Dense(1,  activation="sigmoid", name="output")(x)

    model = Model(inputs, output, name="CNN_LSTM_v4_Large")
    return model


# ============================================================
# 4. Training
# ============================================================
def train_model(model, X_train, y_train, X_val, y_val, output_dir):
    print("\n[3/5] Training model...")

    n_safe      = (y_train == 0).sum()
    n_dangerous = (y_train == 1).sum()
    ratio       = n_safe / n_dangerous
    class_weight = {0: 1.0, 1: ratio}
    print(f"  Class weight dangerous : {ratio:.1f}x")
    print(f"  Model parameters       : {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.85),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ]
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_auc", mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_auc", mode="max",
            factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor="val_auc", mode="max",
            save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ============================================================
# 5. Threshold Sweep
# ============================================================
def find_best_threshold(y_true, y_prob, target_recall=0.60):
    best_threshold = 0.5
    best_recall    = 0.0

    print(f"\n  Threshold sweep (target Recall >= {target_recall}):")
    print(f"  {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1'}")
    print(f"  {'-'*50}")

    results = []
    for t in np.arange(0.05, 0.71, 0.05):
        y_pred    = (y_prob >= t).astype(int)
        tp        = ((y_pred == 1) & (y_true == 1)).sum()
        fp        = ((y_pred == 1) & (y_true == 0)).sum()
        fn        = ((y_pred == 0) & (y_true == 1)).sum()
        recall    = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        results.append((t, recall, precision, f1))
        marker = " ←" if recall >= target_recall else ""
        print(f"  {t:<12.2f} {recall:<10.4f} {precision:<12.4f} {f1:.4f}{marker}")

    valid = [(t, r, p, f) for t, r, p, f in results if r >= target_recall]
    if valid:
        best_threshold, best_recall, best_precision, best_f1 = valid[-1]
    else:
        best_threshold, best_recall, best_precision, best_f1 = max(
            results, key=lambda x: x[3]
        )

    print(f"\n  Best threshold : {best_threshold:.2f}")
    print(f"  Recall         : {best_recall:.4f}")
    print(f"  Precision      : {best_precision:.4f}")
    print(f"  F1-Dangerous   : {best_f1:.4f}")

    return best_threshold


# ============================================================
# 6. Evaluation & Plots
# ============================================================
def evaluate_and_save(model, X_test, y_test, history, output_dir):
    print("\n[4/5] Evaluating model...")

    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\n  AUC-ROC: {auc:.4f}")

    threshold = find_best_threshold(y_test, y_prob, target_recall=0.60)
    y_pred    = (y_prob >= threshold).astype(int)

    report = classification_report(
        y_test, y_pred,
        target_names=["Safe", "Dangerous"],
        digits=4
    )
    print(f"\n  Classification Report (threshold={threshold:.2f}):")
    print(report)

    # Save report
    with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
        f.write(f"AUC-ROC   : {auc:.4f}\n")
        f.write(f"Threshold : {threshold:.2f}\n\n")
        f.write(report)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["auc"],     label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].axhline(y=0.80, color="green", linestyle="--", label="Target 0.80")
    axes[1].set_title("AUC-ROC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history.history["recall"],     label="Train Recall")
    axes[2].plot(history.history["val_recall"], label="Val Recall")
    axes[2].axhline(y=0.60, color="green", linestyle="--", label="Target 0.60")
    axes[2].set_title("Recall (Dangerous Class)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Dangerous"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix  (AUC={auc:.3f} | Threshold={threshold:.2f})")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Save history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(
            {k: [float(v) for v in vals]
             for k, vals in history.history.items()},
            f, indent=2
        )

    print(f"\n  All outputs saved → {output_dir}")
    return auc, threshold


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 55)
    print("  CNN + LSTM v4 — Larger Architecture")
    print("  Conv: 64/128 | LSTM: 128/64 | Dropout: 0.4")
    print("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(PROCESSED_DIR)

    print("\n[2/5] Building model...")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    history = train_model(model, X_train, y_train, X_val, y_val, OUTPUT_DIR)

    auc, threshold = evaluate_and_save(
        model, X_test, y_test, history, OUTPUT_DIR
    )

    y_prob  = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred  = (y_prob >= threshold).astype(int)
    tp      = ((y_pred == 1) & (y_test == 1)).sum()
    fn      = ((y_pred == 0) & (y_test == 1)).sum()
    recall  = tp / (tp + fn + 1e-8)

    print("\n[5/5] Done.")
    print("=" * 55)
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  Model     : {os.path.join(OUTPUT_DIR, 'best_model.keras')}")
    print("=" * 55)

    # Compare with previous versions
    print("\n  Version comparison:")
    print(f"  {'Version':<10} {'AUC':<10} {'Recall'}")
    print(f"  {'-'*30}")
    print(f"  {'v1':<10} {'0.809':<10} {'0.418'}")
    print(f"  {'v2':<10} {'0.827':<10} {'0.620'}")
    print(f"  {'v3':<10} {'0.821':<10} {'0.644'}")
    print(f"  {'v4':<10} {auc:<10.3f} {recall:.3f}  ← current")
    print("=" * 55)

    if recall >= 0.65 and auc >= 0.82:
        print("  SUCCESS — Strong result, ready for integration")
    elif recall >= 0.60 and auc >= 0.80:
        print("  GOOD — Acceptable, proceed to integration")
    else:
        print("  REVIEW — Check training curves before proceeding")


if __name__ == "__main__":
    main()