"""
Stratified Dataset Splitter
Splits the 1,800 NEU-CLS images into 70% Train, 15% Validation, and 15% Test partitions.
Ensures zero data leakage and generates reproducible CSV manifests.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SPLITS_DIR,
    CLASSES,
    CLASS_TO_IDX,
    CONFIG,
)

def create_stratified_splits():
    print("\n--- Generating Stratified 70 / 15 / 15 Dataset Splits ---")
    random_seed = CONFIG["dataset"]["random_seed"]
    ratios = CONFIG["dataset"]["split_ratios"]
    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]

    # Collect all image records
    records = []
    for cls_name in CLASSES:
        cls_dir = RAW_DATA_DIR / cls_name
        for img_file in sorted(cls_dir.glob("*.jpg")):
            rel_path = img_file.relative_to(PROJECT_ROOT)
            records.append({
                "filepath": str(rel_path).replace("\\", "/"),
                "filename": img_file.name,
                "class": cls_name,
                "label": CLASS_TO_IDX[cls_name],
            })

    df = pd.DataFrame(records)
    print(f"Total dataset entries collected: {len(df)}")

    # First split: train vs temp (val + test)
    # temp_ratio = val_ratio + test_ratio = 0.30
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=random_seed,
        shuffle=True,
    )

    # Second split: split temp into val and test (each 50% of temp = 15% of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=random_seed,
        shuffle=True,
    )

    # Integrity verification: check disjointness (zero data leakage)
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])

    assert len(train_paths.intersection(val_paths)) == 0, "Leakage detected between Train and Val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Leakage detected between Train and Test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Leakage detected between Val and Test!"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Sample count mismatch!"

    # Save CSV manifests
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save Summary JSON
    summary = {
        "random_seed": random_seed,
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "ratios": {
            "train": round(len(train_df) / len(df), 4),
            "val": round(len(val_df) / len(df), 4),
            "test": round(len(test_df) / len(df), 4),
        },
        "per_class_counts": {
            "train": train_df["class"].value_counts().to_dict(),
            "val": val_df["class"].value_counts().to_dict(),
            "test": test_df["class"].value_counts().to_dict(),
        },
    }

    summary_json = SPLITS_DIR / "split_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Splits successfully written to {SPLITS_DIR}:")
    print(f"  - Train: {len(train_df)} samples ({len(train_df)//len(CLASSES)} per class)")
    print(f"  - Val:   {len(val_df)} samples ({len(val_df)//len(CLASSES)} per class)")
    print(f"  - Test:  {len(test_df)} samples ({len(test_df)//len(CLASSES)} per class)")
    print(f"  - Leakage check: PASSED (0 overlapping samples across all splits)")
    print("--- Splitting Complete ---\n")

if __name__ == "__main__":
    create_stratified_splits()
