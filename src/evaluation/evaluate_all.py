"""
Master Evaluation and Benchmark Suite
Executes comprehensive evaluation across all trained models:
1. Classical Baseline (HOG + GLCM + SVM)
2. Custom CNN (From Scratch)
3. EfficientNetB0 (Frozen Backbone)
4. EfficientNetB0 (Fine-Tuned)
Generates:
- 6x6 Confusion Matrices (PNG)
- Per-Class Precision, Recall, F1 Analysis (CSV)
- Computational Efficiency Profiling (Parameters, Disk Size, Latency, FPS)
- Multi-Model Master Comparison Table (results/model_comparison.csv)
- Grad-CAM Interpretability Gallery across all 6 defect classes
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    MODELS_DIR,
    METRICS_DIR,
    CONFUSION_MATRICES_DIR,
    GRADCAM_DIR,
    RESULTS_DIR,
    CLASSES,
    CONFIG,
    PROJECT_ROOT,
)
from src.data.dataset import get_train_val_test_datasets
from src.evaluation.confusion_matrix import generate_confusion_matrix_and_metrics
from src.evaluation.efficiency import profile_keras_model
from src.explainability.gradcam import generate_and_save_gradcam_trio

def evaluate_model_on_test(model, model_type="custom", num_runs=5):
    _, _, test_ds = get_train_val_test_datasets(
        model_type=model_type,
        augment_training=False,
    )

    all_preds = []
    all_targets = []

    # Measure inference latency over multiple complete passes
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        preds_run = []
        for imgs, lbls in test_ds:
            preds_run.append(model.predict(imgs, verbose=0))
        latencies.append(time.perf_counter() - start)

    for imgs, lbls in test_ds:
        all_preds.append(model.predict(imgs, verbose=0))
        all_targets.append(lbls.numpy())

    y_pred_probs = np.vstack(all_preds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(np.vstack(all_targets), axis=1)

    n_samples = len(y_true)
    avg_total_time = np.mean(latencies)
    latency_ms = (avg_total_time / n_samples) * 1000
    fps = n_samples / avg_total_time

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "latency_ms": float(latency_ms),
        "fps": float(fps),
    }

def run_master_evaluation():
    print("\n=======================================================")
    print("Executing Master Benchmark & Multi-Model Evaluation")
    print("=======================================================")

    results = []

    # 1. Classical Baseline
    svm_metrics_path = METRICS_DIR / "baseline_svm_metrics.json"
    if svm_metrics_path.exists():
        with open(svm_metrics_path, "r", encoding="utf-8") as f:
            svm_m = json.load(f)
        results.append({
            "Model": "HOG + GLCM + SVM",
            "Type": "Classical Machine Learning",
            "Accuracy (%)": round(svm_m["test_accuracy"] * 100, 2),
            "Macro Precision (%)": round(svm_m["macro_precision"] * 100, 2),
            "Macro Recall (%)": round(svm_m["macro_recall"] * 100, 2),
            "Macro F1 (%)": round(svm_m["macro_f1"] * 100, 2),
            "Weighted F1 (%)": round(svm_m["weighted_f1"] * 100, 2),
            "Latency (ms)": svm_m["inference_latency_ms"],
            "Throughput (FPS)": svm_m["throughput_fps"],
            "Total Parameters": "N/A",
            "Trainable Parameters": "N/A",
            "Disk Size (MB)": "N/A",
        })

    # Deep Learning Models to evaluate
    dl_models_config = [
        {
            "name": "Custom CNN (From Scratch)",
            "save_prefix": "custom_cnn",
            "filename": "custom_cnn.keras",
            "type": "custom",
            "category": "Deep Learning (Custom)",
        },
        {
            "name": "EfficientNetB0 (Frozen Backbone)",
            "save_prefix": "efficientnet_frozen",
            "filename": "efficientnet_frozen.keras",
            "type": "efficientnet",
            "category": "Transfer Learning",
        },
        {
            "name": "EfficientNetB0 (Fine-Tuned)",
            "save_prefix": "efficientnet_finetuned",
            "filename": "efficientnet_finetuned.keras",
            "type": "efficientnet",
            "category": "Transfer Learning (Fine-Tuned)",
        },
    ]

    loaded_models = {}
    per_class_summaries = {}

    for cfg in dl_models_config:
        model_path = MODELS_DIR / cfg["filename"]
        if not model_path.exists():
            print(f"Skipping {cfg['name']} (model file {model_path} not found yet).")
            continue

        print(f"\nEvaluating: {cfg['name']}...")
        model = keras.models.load_model(str(model_path))
        loaded_models[cfg["save_prefix"]] = (model, cfg["type"])

        # Test evaluation
        eval_metrics = evaluate_model_on_test(model, model_type=cfg["type"])
        prof = profile_keras_model(model_path)

        results.append({
            "Model": cfg["name"],
            "Type": cfg["category"],
            "Accuracy (%)": round(eval_metrics["accuracy"] * 100, 2),
            "Macro Precision (%)": round(eval_metrics["macro_precision"] * 100, 2),
            "Macro Recall (%)": round(eval_metrics["macro_recall"] * 100, 2),
            "Macro F1 (%)": round(eval_metrics["macro_f1"] * 100, 2),
            "Weighted F1 (%)": round(eval_metrics["weighted_f1"] * 100, 2),
            "Latency (ms)": eval_metrics["latency_ms"],
            "Throughput (FPS)": eval_metrics["fps"],
            "Total Parameters": f"{prof['total_params']:,}",
            "Trainable Parameters": f"{prof['trainable_params']:,}",
            "Disk Size (MB)": prof["disk_size_mb"],
        })

        # Confusion matrix & Per-class
        per_class_df, cm = generate_confusion_matrix_and_metrics(
            model=model,
            model_name=cfg["name"],
            model_type=cfg["type"],
            save_prefix=cfg["save_prefix"],
        )
        per_class_df.to_csv(METRICS_DIR / f"{cfg['save_prefix']}_per_class.csv", index=False)
        per_class_summaries[cfg["name"]] = per_class_df

    # 3. Save Master Model Comparison CSV
    comparison_df = pd.DataFrame(results)
    master_csv_path = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(master_csv_path, index=False)
    print(f"\n=======================================================")
    print(f"Master Comparison Table saved to {master_csv_path}")
    print(comparison_df.to_string(index=False))
    print("=======================================================\n")

    # 4. Generate Grad-CAM Gallery for Representative Defect Samples
    print("Generating Grad-CAM Explainability Gallery for all 6 defect classes...")
    test_manifest = pd.read_csv(PROJECT_ROOT / "data/splits/test.csv")

    for cls_name in CLASSES:
        cls_samples = test_manifest[test_manifest["class"] == cls_name]
        if cls_samples.empty:
            continue
        sample_path = PROJECT_ROOT / cls_samples.iloc[0]["filepath"]

        for prefix, (model, m_type) in loaded_models.items():
            out_filename = f"gradcam_{cls_name}_{prefix}.png"
            generate_and_save_gradcam_trio(
                model=model,
                image_input=sample_path,
                output_filename=out_filename,
                model_type=m_type,
                title_prefix=f"{cls_name.replace('_', ' ').title()} — {prefix.replace('_', ' ').title()}",
            )

    print("--- Master Evaluation & Explainability Suite Complete ---\n")

if __name__ == "__main__":
    run_master_evaluation()
