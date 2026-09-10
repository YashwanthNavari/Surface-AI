"""
Custom CNN Training and Ablation Pipeline
Trains and compares the Custom CNN configurations:
- Experiment CNN-A: No Augmentation, No Regularization
- Experiment CNN-B: With Augmentation
- Experiment CNN-C: Full Regularized (Augmentation + BatchNorm + Dropout) [Production Candidate]
Saves best model weights to models/custom_cnn.keras and records learning curves.
"""
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    MODELS_DIR,
    METRICS_DIR,
    TRAINING_CURVES_DIR,
    CLASSES,
    CONFIG,
)
from src.data.dataset import get_train_val_test_datasets
from src.models.custom_cnn import build_custom_cnn

def plot_and_save_curves(history_df, exp_name):
    """
    Plots high-resolution training and validation loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    epochs = range(1, len(history_df) + 1)

    # Accuracy Plot
    ax1.plot(epochs, history_df["accuracy"], "b-o", markersize=4, label="Training Accuracy")
    ax1.plot(epochs, history_df["val_accuracy"], "g--s", markersize=4, label="Validation Accuracy")
    ax1.set_title(f"{exp_name} — Accuracy Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim(0.4, 1.02)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    ax1.set_axisbelow(True)
    ax2.plot(epochs, history_df["loss"], "r-o", markersize=4, label="Training Loss")
    ax2.plot(epochs, history_df["val_loss"], "m--s", markersize=4, label="Validation Loss")
    ax2.set_title(f"{exp_name} — Loss Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Categorical Crossentropy", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = TRAINING_CURVES_DIR / f"{exp_name.lower()}_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curves plot to {plot_path}")

def run_experiment(
    exp_id,
    exp_name,
    augment_training,
    use_batch_norm,
    dropout_rate,
    epochs=35,
    save_checkpoint=False,
):
    print(f"\n=======================================================")
    print(f"Starting {exp_id}: {exp_name}")
    print(f"Config: Augmentation={augment_training}, BatchNorm={use_batch_norm}, Dropout={dropout_rate}")
    print(f"=======================================================")

    # Prepare datasets
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        model_type="custom",
        augment_training=augment_training,
    )

    # Build model
    model = build_custom_cnn(
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        name=exp_id.lower(),
    )

    initial_lr = CONFIG["training"]["custom_cnn"]["initial_lr"]
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    csv_log_path = TRAINING_CURVES_DIR / f"{exp_id.lower()}_history.csv"
    callbacks = [
        keras.callbacks.CSVLogger(str(csv_log_path)),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["training"]["custom_cnn"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=CONFIG["training"]["custom_cnn"]["reduce_lr_factor"],
            patience=CONFIG["training"]["custom_cnn"]["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    checkpoint_path = MODELS_DIR / "custom_cnn.keras" if save_checkpoint else None
    if checkpoint_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

    start_train = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    train_duration = time.time() - start_train
    print(f"Training completed in {train_duration:.2f} seconds.")

    # Save and plot history
    history_df = pd.read_csv(csv_log_path)
    plot_and_save_curves(history_df, exp_name)

    # Benchmark on Untouched Test Set
    print(f"Evaluating {exp_id} on untouched Test partition...")
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

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

    print(f"\n{exp_id} Test Set Performance:")
    print(f"  - Test Accuracy:   {acc*100:.2f}%")
    print(f"  - Macro Precision: {macro_p*100:.2f}%")
    print(f"  - Macro Recall:    {macro_r*100:.2f}%")
    print(f"  - Macro F1-Score:  {macro_f1*100:.2f}%")
    print(f"  - Weighted F1:     {weighted_f1*100:.2f}%")
    print(f"  - Latency:         {latency_ms:.2f} ms/image ({fps:.1f} FPS)")
    print(f"  - Total Params:    {total_params:,}")

    metrics = {
        "experiment_id": exp_id,
        "name": exp_name,
        "augmentation": augment_training,
        "batch_norm": use_batch_norm,
        "dropout": dropout_rate,
        "epochs_trained": len(history_df),
        "train_duration_sec": round(train_duration, 2),
        "test_accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "inference_latency_ms": round(float(latency_ms), 2),
        "throughput_fps": round(float(fps), 1),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
    }

    metrics_path = METRICS_DIR / f"{exp_id.lower()}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def run_all_custom_cnn_experiments():
    print("\n--- Phase 8 & 9: Custom CNN Training and Ablations ---")
    results = []

    # 1. Experiment CNN-A: No Augmentation, No BatchNorm, Dropout=0.0
    res_a = run_experiment(
        exp_id="CNN-A",
        exp_name="Custom CNN (No Aug, Baseline)",
        augment_training=False,
        use_batch_norm=False,
        dropout_rate=0.0,
        epochs=25,
        save_checkpoint=False,
    )
    results.append(res_a)

    # 2. Experiment CNN-B: With Augmentation, No BatchNorm, Dropout=0.0
    res_b = run_experiment(
        exp_id="CNN-B",
        exp_name="Custom CNN (+ Augmentation)",
        augment_training=True,
        use_batch_norm=False,
        dropout_rate=0.0,
        epochs=30,
        save_checkpoint=False,
    )
    results.append(res_b)

    # 3. Experiment CNN-C: With Augmentation + BatchNorm + Dropout=0.5 (Production Checkpoint)
    res_c = run_experiment(
        exp_id="CNN-C",
        exp_name="Custom CNN (+ Aug + BatchNorm + Dropout)",
        augment_training=True,
        use_batch_norm=True,
        dropout_rate=0.5,
        epochs=35,
        save_checkpoint=True,
    )
    results.append(res_c)

    # Compile ablation comparison table
    ablation_df = pd.DataFrame(results)
    ablation_csv = METRICS_DIR / "custom_cnn_ablation_summary.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    print(f"\nAblation study summary written to {ablation_csv}")
    print("--- Custom CNN Phase Complete ---\n")

if __name__ == "__main__":
    run_all_custom_cnn_experiments()
