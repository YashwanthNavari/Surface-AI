"""
Confusion Matrix and Per-Class Performance Generator
Produces publication-grade 6x6 confusion matrices with raw counts and normalized percentages,
alongside per-class Precision, Recall, and F1-Score tables.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.config import (
    CLASSES,
    CONFUSION_MATRICES_DIR,
    METRICS_DIR,
)
from src.data.dataset import get_train_val_test_datasets

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def generate_confusion_matrix_and_metrics(
    model,
    model_name,
    model_type="custom",
    save_prefix="model",
):
    print(f"\nGenerating Confusion Matrix & Per-Class Metrics for {model_name}...")
    _, _, test_ds = get_train_val_test_datasets(
        model_type=model_type,
        augment_training=False,
    )

    all_preds = []
    all_targets = []

    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        all_preds.append(preds)
        all_targets.append(lbls.numpy())

    y_pred = np.argmax(np.vstack(all_preds), axis=1)
    y_true = np.argmax(np.vstack(all_targets), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create annotated labels (Count + %)
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_norm[i, j] * 100
            if i == j:
                annot[i, j] = f"{c}\n({p:.1f}%)"
            elif c == 0:
                annot[i, j] = "0"
            else:
                annot[i, j] = f"{c}\n({p:.1f}%)"

    # Plot figure
    plt.figure(figsize=(9, 7.5), dpi=300)
    display_classes = [c.replace("_", " ").title() for c in CLASSES]
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=display_classes,
        yticklabels=display_classes,
        cbar=True,
        linewidths=1.2,
        linecolor="#dddddd",
    )

    plt.title(f"{model_name} — Confusion Matrix on Untouched Test Set", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Defect Category", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Ground Truth Defect Category", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    out_path = CONFUSION_MATRICES_DIR / f"{save_prefix}_cm.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {out_path}")

    # Compute Per-Class Metrics Table
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASSES)))
    per_class_df = pd.DataFrame({
        "Class": display_classes,
        "Precision": np.round(precision * 100, 2),
        "Recall": np.round(recall * 100, 2),
        "F1_Score": np.round(f1 * 100, 2),
        "Support": support,
    })

    return per_class_df, cm
