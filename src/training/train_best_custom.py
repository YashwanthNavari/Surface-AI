"""
Champion Custom CNN Trainer
Trains the top-performing Custom CNN configuration (Experiment CNN-B: 4-stage ConvNet + Data Augmentation)
and saves the checkpoint directly to models/custom_cnn.keras.
"""
import json
import time
import numpy as np
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

def train_and_save_champion_custom_cnn():
    print("\n=======================================================")
    print("Training Champion Custom CNN (CNN-B architecture)")
    print("Saving checkpoint -> models/custom_cnn.keras")
    print("=======================================================")

    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        model_type="custom",
        augment_training=True,
    )

    model = build_custom_cnn(
        use_batch_norm=False,
        dropout_rate=0.0,
        name="custom_cnn_champion",
    )

    initial_lr = CONFIG["training"]["custom_cnn"]["initial_lr"]
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = MODELS_DIR / "custom_cnn.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    start_train = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=24,
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.time() - start_train
    print(f"Champion Custom CNN trained in {duration:.2f}s. Checkpoint saved to {checkpoint_path}")

    # Evaluate on untouched test set
    all_preds = []
    all_targets = []
    for imgs, lbls in test_ds:
        all_preds.append(model.predict(imgs, verbose=0))
        all_targets.append(lbls.numpy())

    y_pred = np.argmax(np.vstack(all_preds), axis=1)
    y_true = np.argmax(np.vstack(all_targets), axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print(f"Custom CNN Champion Test Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%")

if __name__ == "__main__":
    train_and_save_champion_custom_cnn()
