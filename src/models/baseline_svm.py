"""
Classical Computer Vision Baseline: HOG + GLCM + Support Vector Machine (SVM)
Extracts handcrafted gradient and texture descriptors and classifies with an RBF SVM.
Establishes the empirical non-deep-learning benchmark for the study.
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.config import (
    SPLITS_DIR,
    METRICS_DIR,
    PROJECT_ROOT,
    CLASSES,
    CONFIG,
)

def extract_features(filepath):
    """
    Extracts concatenated HOG (gradient) and GLCM (texture) features from a surface image.
    """
    full_path = PROJECT_ROOT / filepath
    img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {full_path}")

    # Resize to standard 224x224
    img_resized = cv2.resize(img, (224, 224))

    # 1. HOG features
    hog_feats = hog(
        img_resized,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )

    # 2. GLCM features (texture statistics across angles 0, 45, 90, 135 deg)
    glcm = graycomatrix(
        img_resized,
        distances=[1, 3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, "contrast").ravel()
    dissimilarity = graycoprops(glcm, "dissimilarity").ravel()
    homogeneity = graycoprops(glcm, "homogeneity").ravel()
    energy = graycoprops(glcm, "energy").ravel()
    correlation = graycoprops(glcm, "correlation").ravel()

    glcm_feats = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

    return np.hstack([hog_feats, glcm_feats])

def load_partition(manifest_name):
    csv_path = SPLITS_DIR / manifest_name
    df = pd.read_csv(csv_path)
    X = []
    y = df["label"].values

    for fp in df["filepath"]:
        X.append(extract_features(fp))

    return np.array(X, dtype=np.float32), y

def train_and_evaluate_svm():
    print("\n--- Training Classical Baseline: HOG + GLCM + SVM ---")
    start_feat_time = time.time()
    print("Extracting features from Train partition...")
    X_train, y_train = load_partition("train.csv")
    print("Extracting features from Validation partition...")
    X_val, y_val = load_partition("val.csv")
    print("Extracting features from Test partition...")
    X_test, y_test = load_partition("test.csv")
    feat_time = time.time() - start_feat_time
    print(f"Feature extraction completed in {feat_time:.2f}s. Feature dimension: {X_train.shape[1]}")

    # Build Pipeline: StandardScaler + RBF SVC
    clf = make_pipeline(
        StandardScaler(),
        SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
    )

    start_train_time = time.time()
    print("Fitting RBF Support Vector Classifier...")
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train_time
    print(f"SVM training completed in {train_time:.2f}s")

    # Evaluate on untouched Test set
    start_infer = time.time()
    y_pred = clf.predict(X_test)
    total_infer_time = time.time() - start_infer
    latency_ms = (total_infer_time / len(X_test)) * 1000
    fps = len(X_test) / total_infer_time

    accuracy = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print("\nClassical Baseline (HOG+SVM) Test Results:")
    print(f"  - Test Accuracy:   {accuracy*100:.2f}%")
    print(f"  - Macro Precision: {macro_p*100:.2f}%")
    print(f"  - Macro Recall:    {macro_r*100:.2f}%")
    print(f"  - Macro F1-Score:  {macro_f1*100:.2f}%")
    print(f"  - Weighted F1:     {weighted_f1*100:.2f}%")
    print(f"  - Inference Latency: {latency_ms:.2f} ms/image ({fps:.1f} FPS)")

    metrics = {
        "model": "HOG + GLCM + SVM",
        "category": "Classical Baseline",
        "feature_dim": int(X_train.shape[1]),
        "train_time_sec": round(train_time, 2),
        "test_accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "inference_latency_ms": round(float(latency_ms), 2),
        "throughput_fps": round(float(fps), 1),
    }

    metrics_file = METRICS_DIR / "baseline_svm_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved baseline metrics to {metrics_file}")
    print("--- Baseline SVM Execution Complete ---\n")
    return metrics

if __name__ == "__main__":
    train_and_evaluate_svm()
