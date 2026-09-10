"""
EfficientNetB0 Fine-Tuning Pipeline
Unfreezes top convolutional blocks (Block 6 & 7) while preserving BatchNormalization running statistics.
Trains with a reduced learning rate (1e-5) and saves best weights to models/efficientnet_finetuned.keras.
"""
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    MODELS_DIR,
    METRICS_DIR,
    TRAINING_CURVES_DIR,
    CLASSES,
    CONFIG,
)
from src.data.dataset import get_train_val_test_datasets

def plot_curves(history_df, exp_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    epochs = range(1, len(history_df) + 1)

    ax1.plot(epochs, history_df["accuracy"], "b-o", markersize=4, label="Train Accuracy")
    ax1.plot(epochs, history_df["val_accuracy"], "g--s", markersize=4, label="Val Accuracy")
    ax1.set_title(f"{exp_name} — Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim(0.7, 1.02)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, history_df["loss"], "r-o", markersize=4, label="Train Loss")
    ax2.plot(epochs, history_df["val_loss"], "m--s", markersize=4, label="Val Loss")
    ax2.set_title(f"{exp_name} — Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Categorical Crossentropy", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = TRAINING_CURVES_DIR / f"{exp_name.lower().replace(' ', '_')}_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curves plot to {plot_path}")

def fine_tune_efficientnet():
    print("\n=======================================================")
    print("Starting Fine-Tuning: EfficientNetB0 (Upper Blocks Unfrozen)")
    print("Strategy: Unfreeze upper Conv blocks, keep BatchNorm frozen")
    print("=======================================================")

    frozen_model_path = MODELS_DIR / "efficientnet_frozen.keras"
    if not frozen_model_path.exists():
        raise FileNotFoundError(f"Frozen checkpoint not found at {frozen_model_path}. Train transfer model first.")

    model = keras.models.load_model(str(frozen_model_path))

    # Model has 242 flat layers
    # Unfreeze the top 30 layers (excluding BatchNormalization to protect running stats)
    num_unfrozen = 30
    for layer in model.layers[:-num_unfrozen]:
        layer.trainable = False

    for layer in model.layers[-num_unfrozen:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_count = model.count_params()
    print(f"Total parameters: {total_count:,} | Trainable parameters: {trainable_count:,}")

    # Low learning rate for fine-tuning to prevent catastrophic forgetting
    fine_tune_lr = CONFIG["training"]["efficientnet_finetuned"]["initial_lr"]
    epochs = CONFIG["training"]["efficientnet_finetuned"]["epochs"]
    optimizer = keras.optimizers.Adam(learning_rate=fine_tune_lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        model_type="efficientnet",
        augment_training=True,
    )

    csv_log_path = TRAINING_CURVES_DIR / "efficientnet_finetuned_history.csv"
    checkpoint_path = MODELS_DIR / "efficientnet_finetuned.keras"

    callbacks = [
        keras.callbacks.CSVLogger(str(csv_log_path)),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["training"]["efficientnet_finetuned"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    start_train = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    train_duration = time.time() - start_train
    print(f"Fine-tuning completed in {train_duration:.2f} seconds.")

    # Save curves
    history_df = pd.read_csv(csv_log_path)
    plot_curves(history_df, "EfficientNetB0 Fine-Tuned")

    # Evaluate on untouched Test set
    print("Evaluating EfficientNetB0 Fine-Tuned on untouched Test partition...")
    all_preds = []
    all_targets = []
    start_infer = time.time()

    for batch_imgs, batch_lbls in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        all_preds.append(preds)
        all_targets.append(batch_lbls.numpy())

    infer_duration = time.time() - start_infer
    y_pred_probs = np.vstack(all_preds)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(np.vstack(all_targets), axis=1)

    test_samples = len(y_true_classes)
    latency_ms = (infer_duration / test_samples) * 1000
    fps = test_samples / infer_duration

    acc = accuracy_score(y_true_classes, y_pred_classes)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average="weighted")

    print(f"\nEfficientNetB0 Fine-Tuned Test Set Performance:")
    print(f"  - Test Accuracy:   {acc*100:.2f}%")
    print(f"  - Macro Precision: {macro_p*100:.2f}%")
    print(f"  - Macro Recall:    {macro_r*100:.2f}%")
    print(f"  - Macro F1-Score:  {macro_f1*100:.2f}%")
    print(f"  - Weighted F1:     {weighted_f1*100:.2f}%")
    print(f"  - Latency:         {latency_ms:.2f} ms/image ({fps:.1f} FPS)")
    print(f"  - Total Params:    {total_count:,}")
    print(f"  - Trainable Params:{trainable_count:,}")

    metrics = {
        "experiment_id": "E5",
        "model": "EfficientNetB0 (Fine-Tuned)",
        "backbone": "ImageNet (Top 30 layers unfrozen)",
        "epochs_trained": len(history_df),
        "train_duration_sec": round(train_duration, 2),
        "test_accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "inference_latency_ms": round(float(latency_ms), 2),
        "throughput_fps": round(float(fps), 1),
        "total_parameters": int(total_count),
        "trainable_parameters": int(trainable_count),
    }

    metrics_path = METRICS_DIR / "efficientnet_finetuned_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")
    print("--- EfficientNet Fine-Tuning Complete ---\n")
    return metrics

if __name__ == "__main__":
    fine_tune_efficientnet()
