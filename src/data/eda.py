"""
Exploratory Data Analysis (EDA) Module
Generates class distribution plots, sample image grids, pixel intensity statistics,
and dataset summary documentation for the 6-class NEU database.
"""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.config import (
    RAW_DATA_DIR,
    CLASSES,
    RESULTS_DIR,
    METRICS_DIR,
)

# Set high visual styling for academic figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"

def run_eda():
    print("\n--- Running Exploratory Data Analysis (EDA) ---")
    data_records = []
    pixel_stats = []

    for cls_name in CLASSES:
        cls_dir = RAW_DATA_DIR / cls_name
        files = sorted(list(cls_dir.glob("*.jpg")))
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            mean_val = float(np.mean(img))
            std_val = float(np.std(img))
            min_val = int(np.min(img))
            max_val = int(np.max(img))

            data_records.append({
                "filepath": str(f.relative_to(RAW_DATA_DIR.parent.parent)),
                "filename": f.name,
                "class": cls_name,
                "height": h,
                "width": w,
                "mean_intensity": mean_val,
                "std_intensity": std_val,
                "min_intensity": min_val,
                "max_intensity": max_val,
            })
            pixel_stats.append({
                "class": cls_name,
                "mean": mean_val,
                "std": std_val,
            })

    df = pd.DataFrame(data_records)

    # 1. Save Dataset Statistics CSV
    stats_csv_path = METRICS_DIR / "dataset_statistics.csv"
    summary_stats = df.groupby("class").agg({
        "filename": "count",
        "height": "first",
        "width": "first",
        "mean_intensity": ["mean", "std"],
        "std_intensity": "mean",
    }).round(2)
    summary_stats.columns = ["Image_Count", "Height", "Width", "Mean_Pixel_Val", "Pixel_Val_Std", "Avg_Texture_Contrast"]
    summary_stats.to_csv(stats_csv_path)
    print(f"Saved dataset statistics to {stats_csv_path}")

    # 2. Generate Class Distribution Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    palette = sns.color_palette("mako", n_colors=len(CLASSES))
    counts = df["class"].value_counts()[CLASSES]
    bars = plt.bar(CLASSES, counts, color=palette, edgecolor="#333333", linewidth=1.2, width=0.55)

    plt.title("NEU Surface Defect Database — Class Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Defect Category", fontsize=12, labelpad=10)
    plt.ylabel("Number of Samples", fontsize=12, labelpad=10)
    plt.ylim(0, 360)
    plt.xticks(rotation=15, ha="right", fontsize=11)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 8, f"{int(yval)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    dist_plot_path = RESULTS_DIR / "class_distribution.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to {dist_plot_path}")

    # 3. Generate High-Resolution Sample Grid (2x3 classes, 3 samples each)
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(9, 16), dpi=300)
    for i, cls_name in enumerate(CLASSES):
        cls_dir = RAW_DATA_DIR / cls_name
        sample_files = sorted(list(cls_dir.glob("*.jpg")))[:3]
        for j, img_file in enumerate(sample_files):
            ax = axes[i, j]
            img = cv2.imread(str(img_file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(cls_name.replace("_", " ").title(), fontsize=12, fontweight="bold", labelpad=10)
            if i == 0:
                ax.set_title(f"Sample #{j+1}", fontsize=11, pad=8)

    plt.suptitle("NEU Surface Defect Database — Representative Samples", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    sample_grid_path = RESULTS_DIR / "sample_grid.png"
    plt.savefig(sample_grid_path, dpi=300)
    plt.close()
    print(f"Saved sample grid image to {sample_grid_path}")

    # 4. Generate Pixel Intensity Distribution Plot
    plt.figure(figsize=(10, 5), dpi=300)
    for cls_name in CLASSES:
        cls_df = df[df["class"] == cls_name]
        sns.kdeplot(cls_df["mean_intensity"], label=cls_name.replace("_", " ").title(), linewidth=2)
    plt.title("Pixel Mean Intensity Distribution across Defect Classes", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Pixel Intensity (0-255)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    intensity_plot_path = RESULTS_DIR / "intensity_distribution.png"
    plt.savefig(intensity_plot_path, dpi=300)
    plt.close()
    print(f"Saved intensity distribution plot to {intensity_plot_path}")

    # 5. Write Dataset Summary Markdown
    summary_md_path = RESULTS_DIR / "dataset_summary.md"
    summary_content = f"""# NEU Surface Defect Database: Exploratory Data Analysis Summary

## 1. Overview
- **Database Name:** Northeastern University (NEU) Surface Defect Database (NEU-CLS)
- **Problem Formulation:** Multi-class manufacturing surface defect classification (6 distinct defect classes on hot-rolled steel strip).
- **Total Images:** {len(df)} images
- **Number of Classes:** {len(CLASSES)}
- **Class Balance:** Exactly 300 images per class (perfectly balanced 16.67% per category).
- **Original Resolution:** 200 × 200 pixels
- **Target Preprocessing Resolution:** 224 × 224 × 3 (standardized for Custom CNN & EfficientNetB0).

## 2. Quantitative Class Summary Table

| Defect Class | Sample Count | Dimensions | Mean Pixel Intensity | Pixel Std Dev | Avg Contrast |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for cls_name in CLASSES:
        cls_sub = df[df["class"] == cls_name]
        summary_content += f"| {cls_name.replace('_', ' ').title()} | {len(cls_sub)} | 200 × 200 | {cls_sub['mean_intensity'].mean():.2f} | {cls_sub['mean_intensity'].std():.2f} | {cls_sub['std_intensity'].mean():.2f} |\n"

    summary_content += """
## 3. Academic & Domain Observations
1. **Perfect Class Symmetry:** Every class consists of exactly 300 images, eliminating sample imbalance biases during gradient updates.
2. **Texture vs. Edge Disparities:**
   - *Crazing* features high-frequency web-like microcracks with low contrast against the steel background.
   - *Inclusion* and *Patches* present distinct localized dark or light spatial regions with high gradient magnitudes.
   - *Scratches* present strong directional linear features, which will benefit significantly from rotational and affine augmentations.
3. **Artifacts Generated:**
   - Class Distribution Chart: `results/class_distribution.png`
   - Representative Sample Grid: `results/sample_grid.png`
   - Pixel Intensity Density: `results/intensity_distribution.png`
   - CSV Metrics: `results/metrics/dataset_statistics.csv`
"""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"Saved dataset summary markdown to {summary_md_path}")
    print("--- EDA Complete ---\n")

if __name__ == "__main__":
    run_eda()
