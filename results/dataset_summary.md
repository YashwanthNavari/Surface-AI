# NEU Surface Defect Database: Exploratory Data Analysis Summary

## 1. Overview
- **Database Name:** Northeastern University (NEU) Surface Defect Database (NEU-CLS)
- **Problem Formulation:** Multi-class manufacturing surface defect classification (6 distinct defect classes on hot-rolled steel strip).
- **Total Images:** 1800 images
- **Number of Classes:** 6
- **Class Balance:** Exactly 300 images per class (perfectly balanced 16.67% per category).
- **Original Resolution:** 200 × 200 pixels
- **Target Preprocessing Resolution:** 224 × 224 × 3 (standardized for Custom CNN & EfficientNetB0).

## 2. Quantitative Class Summary Table

| Defect Class | Sample Count | Dimensions | Mean Pixel Intensity | Pixel Std Dev | Avg Contrast |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Crazing | 300 | 200 × 200 | 141.11 | 28.67 | 29.05 |
| Inclusion | 300 | 200 × 200 | 106.37 | 33.87 | 13.02 |
| Patches | 300 | 200 × 200 | 132.51 | 43.06 | 54.22 |
| Pitted Surface | 300 | 200 × 200 | 176.67 | 49.35 | 26.68 |
| Rolled-In Scale | 300 | 200 × 200 | 117.91 | 24.62 | 16.09 |
| Scratches | 300 | 200 × 200 | 95.15 | 29.09 | 21.70 |

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
