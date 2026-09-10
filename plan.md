# Final Implementation Plan

## AI-Based Manufacturing Surface Defect Detection and Classification Using Custom CNN and Transfer Learning

**Project Level:** 30-Mark Deep Learning Project-Based Learning (PBL)  
**Task Domain:** Computer Vision / Industrial Quality Control / Surface Defect Classification  
**Target Architecture:** Custom CNN (Scratch) vs. EfficientNetB0 (Pretrained Transfer Learning & Fine-Tuning)  
**Serving & Interface:** FastAPI Backend + React (Vite) Industrial Vision Inspection Console  

---

# 0. Final Project Definition

### Title
> **AI-Based Manufacturing Surface Defect Detection and Classification Using Custom CNN and Transfer Learning**

### Primary Objective
Build an end-to-end, production-grade AI visual inspection system that analyzes manufacturing surface images and classifies defects using:
1. **Custom CNN trained from scratch**
2. **EfficientNetB0 using transfer learning (feature extraction with frozen backbone)**
3. **Fine-tuned EfficientNetB0 (unfreezing upper convolutional blocks)**

Then perform a rigorous academic comparison evaluating:
* Classification performance (Accuracy, Macro/Weighted Precision, Recall, F1-Score)
* Training dynamics & convergence stability
* Generalization across stratified splits
* Computational efficiency (FLOPs/parameter counts, model size, inference latency, throughput)
* Error patterns and failure modes
* Explainability via Grad-CAM (visual validation of defect localization)

Finally, serve the finalized models via a **FastAPI backend** and display them in a modern, responsive **React + Vite Industrial AI Vision Inspection Console**.

---

# 1. System Pipeline Architecture

```text
                         ┌─────────────────────┐
                         │ Manufacturing Image │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Preprocessing       │
                         │ Resize / Normalize  │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
          ┌──────────────────┐              ┌──────────────────┐
          │   Custom CNN     │              │  EfficientNetB0  │
          │  From Scratch    │              │ ImageNet Weights │
          └────────┬─────────┘              └────────┬─────────┘
                   │                                 │
                   │                                 ▼
                   │                         ┌────────────────┐
                   │                         │ Fine-Tuning    │
                   │                         └───────┬────────┘
                   │                                 │
                   └────────────────┬────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Evaluation Engine   │
                         ├─────────────────────┤
                         │ Accuracy            │
                         │ Precision           │
                         │ Recall              │
                         │ F1                  │
                         │ Confusion Matrix    │
                         │ Efficiency          │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
           ┌─────────────────┐             ┌─────────────────┐
           │ Error Analysis  │             │    Grad-CAM     │
           └─────────────────┘             └─────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                           ┌─────────────────┐
                           │ FastAPI Backend │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ React + Vite UI │
                           └─────────────────┘
```

---

# 2. Dataset Decision

## Primary Dataset: NEU Surface Defect Database (Northeastern University)
Use the standard 6-class NEU benchmark dataset:

```text
1. Crazing (Cr)
2. Inclusion (In)
3. Patches (Pa)
4. Pitted Surface (PS)
5. Rolled-in Scale (RS)
6. Scratches (Sc)
```

- **Volume:** 300 images per class (1,800 images total)
- **Original Format:** Grayscale BMP/JPEG images, 200 × 200 pixels
- **Academic Definition:** Multi-class manufacturing surface-defect classification (not normal-vs-anomaly detection).

---

# 3. Optional External Validation

**Secondary Benchmark:** KolektorSDD / KolektorSDD2
- Cross-dataset evaluation to measure performance degradation under domain shift (illumination, surface texture, electrical commutators vs. hot-rolled steel strip).
- Discussed as an academic research component demonstrating out-of-distribution robustness.

---

# 4. Phase 1 — Project Setup & Environment

### Tech Stack
* **Language & Core:** Python 3.10+, TensorFlow / Keras, OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
* **Backend:** FastAPI, Uvicorn, Pydantic, Python-Multipart
* **Frontend:** React 18/19, Vite, Lucide Icons, Vanilla Modern CSS / Tailwind (if preferred)
* **Storage & Config:** SQLite (inspection history), PyYAML (`config.yaml`)

---

# 5. Phase 2 — Repository Architecture

```text
manufacturing-defect-detection/
│
├── README.md
├── plan.md
├── requirements.txt
├── .gitignore
├── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_custom_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   ├── 06_fine_tuning.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_gradcam.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   └── split.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_cnn.py
│   │   ├── efficientnet.py
│   │   └── model_loader.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_custom.py
│   │   ├── train_transfer.py
│   │   └── fine_tune.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   ├── efficiency.py
│   │   └── comparison.py
│   │
│   └── explainability/
│       ├── __init__.py
│       └── gradcam.py
│
├── models/
│   ├── custom_cnn.keras
│   ├── efficientnet_frozen.keras
│   └── efficientnet_finetuned.keras
│
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── training_curves/
│   ├── gradcam/
│   └── model_comparison.csv
│
├── experiments/
│
├── backend/
│   ├── main.py
│   ├── inference.py
│   ├── schemas.py
│   └── gradcam_service.py
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
```

---

# 6. Phase 3 — Dataset Exploration (EDA)

Notebook: `notebooks/01_dataset_analysis.ipynb`

### Quality Checks & Metrics
* Integrity verification: corrupt files, truncated streams, dimensional uniformity.
* Global pixel statistics: channel means, standard deviations, intensity histograms.
* Class balance verification (300 images/class).

### Deliverables
* `results/metrics/dataset_statistics.csv`
* `results/dataset_summary.md`
* `results/class_distribution.png`
* `results/sample_grid.png`

---

# 7. Phase 4 — Dataset Splitting

* **Stratification:** Splitting done strictly before data augmentation.
* **Ratios:** 70% Training / 15% Validation / 15% Test
  * Training: ~1,260 images (210 per class)
  * Validation: ~270 images (45 per class)
  * Test: ~270 images (45 per class)
* **Integrity Guarantee:** Test set is strictly isolated until final inference benchmarks.

---

# 8. Phase 5 — Preprocessing Pipeline

1. **Load:** Read image via OpenCV / Keras preprocessing.
2. **Resize:** Target resolution $224 \times 224$ (bilinear/bicubic interpolation).
3. **Channel Formatting:** 3-channel RGB conversion (accommodating grayscale input for ImageNet compatibility).
4. **Rescaling / Normalization:**
   * Custom CNN: Pixel normalization to $[0, 1]$ or $[-1, 1]$.
   * EfficientNet: `tf.keras.applications.efficientnet.preprocess_input`.

---

# 9. Phase 6 — Data Augmentation

Applied strictly to training batches on the fly:
* Random horizontal & vertical flips
* Random rotation ($\pm 15^\circ$)
* Random zoom ($\pm 10\%$)
* Random translation ($\pm 10\%$)
* Random contrast adjustment

---

# 10. Phase 7 — Classical Baseline

Implement standard classical computer vision benchmark:
* **Feature Extraction:** Histogram of Oriented Gradients (HOG) + Gray-Level Co-occurrence Matrix (GLCM).
* **Classifier:** Linear / RBF Support Vector Machine (SVM) or Random Forest.
* Purpose: Establish empirical justification for Deep Learning architectures over handcrafted feature engineering.

---

# 11. Phase 8 — Custom CNN Architecture

```text
Input Layer (224 × 224 × 3)
         ↓
Conv2D (32 filters, 3×3, padding='same') -> BatchNorm -> ReLU -> MaxPool2D(2×2)
         ↓
Conv2D (64 filters, 3×3, padding='same') -> BatchNorm -> ReLU -> MaxPool2D(2×2)
         ↓
Conv2D (128 filters, 3×3, padding='same') -> BatchNorm -> ReLU -> MaxPool2D(2×2)
         ↓
Conv2D (256 filters, 3×3, padding='same') -> BatchNorm -> ReLU -> MaxPool2D(2×2)
         ↓
GlobalAveragePooling2D
         ↓
Dense (128 units, ReLU) -> Dropout (0.5)
         ↓
Dense (6 units, Softmax)
```

* **Loss:** Categorical Cross-Entropy
* **Optimizer:** Adam ($\alpha = 10^{-3}$)
* **Callbacks:** `EarlyStopping(patience=10)`, `ReduceLROnPlateau(factor=0.2, patience=4)`, `ModelCheckpoint`, `CSVLogger`

---

# 12. Phase 9 — Custom CNN Ablation Experiments

1. **CNN-A:** Base Custom CNN without augmentation
2. **CNN-B:** Base Custom CNN + online data augmentation
3. **CNN-C:** Full Custom CNN + data augmentation + Batch Normalization + Dropout (0.5)

---

# 13. Phase 10 & 11 — Transfer Learning (EfficientNetB0)

* **Backbone:** EfficientNetB0 pretrained on ImageNet ($224 \times 224 \times 3$).
* **Strategy:** Frozen backbone weights (`trainable = False`).
* **Classification Head:**
  ```text
  GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.4) -> Dense(6, Softmax)
  ```
* **Optimizer:** Adam ($\alpha = 3 \times 10^{-4}$)
* **Checkpoint:** `models/efficientnet_frozen.keras`

---

# 14. Phase 12 — Fine-Tuning EfficientNetB0

* **Strategy:** Unfreeze the top 20–30 layers (e.g., Block 6 & Block 7).
* **Learning Rate:** Reduced learning rate ($\alpha = 1 \times 10^{-5}$) with cosine decay or early stopping.
* **Checkpoint:** `models/efficientnet_finetuned.keras`

---

# 15. Phase 13 — Hyperparameter Experiments

Systematic grid exploration:
* Learning Rates: $[10^{-3}, 3 \times 10^{-4}, 10^{-5}]$
* Batch Sizes: $[16, 32]$
* Dropout Rates: $[0.3, 0.5]$

---

# 16. Phase 14 — Final Quantitative Evaluation

Evaluated strictly on the untouched test partition ($N = 270$):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Report Macro Precision, Macro Recall, Macro $F_1$, and Weighted $F_1$.

---

# 17. Phase 15 & 16 — Confusion Matrix & Per-Class Analysis

* Full $6 \times 6$ confusion matrices for each architecture.
* Per-class Breakdown Table:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Crazing | — | — | — | 45 |
| Inclusion | — | — | — | 45 |
| Patches | — | — | — | 45 |
| Pitted Surface | — | — | — | 45 |
| Rolled-in Scale | — | — | — | 45 |
| Scratches | — | — | — | 45 |

---

# 18. Phase 17 & 18 — Training Dynamics & Statistical Reliability

* Training vs. Validation Loss and Accuracy convergence curves.
* Multi-Seed Variance Testing (Seeds: 42, 101, 2024, 7, 99):
  $$\mu \pm \sigma \quad (\text{e.g., } 98.4\% \pm 0.3\%)$$

---

# 19. Phase 19 — Computational Efficiency Benchmarks

| Model | Test Acc (%) | Macro F1 | Total Params | Trainable Params | Size (MB) | Latency (ms/img) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HOG + SVM | — | — | N/A | N/A | — | — | — |
| Custom CNN | — | — | ~350K | ~350K | ~1.5 MB | — | — |
| EfficientNetB0 Frozen | — | — | ~4.2M | ~165K | ~17 MB | — | — |
| EfficientNetB0 Fine-Tuned | — | — | ~4.2M | ~1.2M | ~17 MB | — | — |

---

# 20. Phase 20 — Error Analysis

* Failure gallery of misclassified and low-confidence test samples.
* Systematic qualitative breakdown across defect size, contrast variation, and texture ambiguity.

---

# 21. Phase 21 & 22 — Explainability via Grad-CAM

* Target the final convolutional feature maps:
  * Custom CNN: final Conv2D layer (`conv2d_3`).
  * EfficientNetB0: top activation layer (`top_conv` / `block7a_project_conv`).
* Visualize: Original $\rightarrow$ Normalized Heatmap $\rightarrow$ Jet Colormap Overlay.
* Analyze alignment between Grad-CAM activation peaks and ground-truth defect regions.

---

# 22. Phase 23 — External Validation (KolektorSDD)

* Test top-performing fine-tuned model on external industrial dataset without retraining to evaluate domain shift and transferability.

---

# 23. Phase 24 — Model Freezing & Packaging

Export and freeze production weights:
* `models/custom_cnn.keras`
* `models/efficientnet_frozen.keras`
* `models/efficientnet_finetuned.keras`
* Labels metadata: `models/class_labels.json`

---

# 24. Phase 25 & 26 — FastAPI Backend Service

* `POST /api/v1/predict`: Single/batch inference returning multi-model classification, probabilities, and consensus agreement.
* `POST /api/v1/explain`: Returns base64 encoded Grad-CAM heatmap and overlay images with class activations.
* `GET /api/v1/history`: Returns inspection logs stored in SQLite.
* `GET /api/v1/metrics`: Serves compiled model evaluation benchmarks for the frontend research tab.

---

# 25. Phase 27 — React + Vite Industrial Inspection Console

* **Aesthetic:** Dark-mode industrial AI inspection terminal (high-contrast status badges, HUD telemetry cards, crisp SVG graphs).
* **Views:**
  1. **Live Inspection Console:** Drag-and-drop image upload, real-time multi-model inference, consensus status, confidence gauge.
  2. **Explainability Studio:** Interactive Grad-CAM model comparison slider (Custom vs. Pretrained vs. Fine-tuned).
  3. **Research & Benchmarks:** Interactive charts for training curves, confusion matrices, latency vs. accuracy Pareto frontier.
  4. **Audit History Log:** Filterable inspection records with export capability.

---

# 26. Phase 28 — Comprehensive Testing

* Unit tests for image preprocessing and input tensor validation (`tests/test_preprocessing.py`).
* Model inference tests (`tests/test_models.py`).
* API endpoint tests (`tests/test_api.py`).

---

# 27. Phase 29 & 30 — Reproducibility & Tracking

* Unified configuration via `config.yaml`.
* Experiment catalog in `results/model_comparison.csv`.

---

# 28. Master Experiment Matrix

| ID | Experiment Name | Model Architecture | Training Strategy | Key Metric Focus |
| :---: | :--- | :--- | :--- | :--- |
| **E0** | Classical Baseline | HOG + GLCM + SVM | RBF Kernel | Baseline benchmark |
| **E1** | Custom CNN (No Aug) | 4-Stage ConvNet | Scratch, no augmentation | Overfitting diagnosis |
| **E2** | Custom CNN (+ Aug) | 4-Stage ConvNet | Scratch, random flip/zoom/rot | Generalization gain |
| **E3** | Custom CNN (Regularized) | 4-Stage ConvNet | Aug + BatchNorm + Dropout (0.5) | Peak custom performance |
| **E4** | EfficientNetB0 (Frozen) | Pretrained ImageNet | Frozen feature extractor | Feature reuse efficacy |
| **E5** | EfficientNetB0 (Fine-Tuned) | Pretrained ImageNet | Unfreeze upper blocks | Peak accuracy & F1 |
| **E6** | Hyperparameter Search | Best Models | LR & batch size sweeps | Sensitivity analysis |
| **E7** | Multi-Seed Reliability | E3, E4, E5 | 5 Random Seeds | Statistical significance ($\mu \pm \sigma$) |
| **E8** | Explainability Study | All DL Models | Grad-CAM heatmap generation | Visual localization fidelity |
| **E9** | Out-of-Domain Validation | Fine-Tuned Model | Tested on KolektorSDD | Domain adaptation & robustness |

---

# 29. Academic Research Questions (Viva Defense)

* **RQ1:** Can a lightweight custom CNN trained from scratch achieve acceptable accuracy on surface defects?
* **RQ2:** How does transfer learning from generic natural images (ImageNet) compare against domain-specific custom training on industrial surfaces?
* **RQ3:** Does selective fine-tuning of deep convolutional blocks yield statistically significant gains over frozen feature extraction?
* **RQ4:** What is the quantifiable impact of data augmentation on convergence and generalization?
* **RQ5:** Which defect classes exhibit morphological similarity causing confusion, and how does fine-tuning resolve them?
* **RQ6:** What is the computational Pareto frontier (parameter count & latency vs. macro F1)?
* **RQ7:** Do the model's high-activation heatmaps physically correlate with actual defective surface regions?
* **RQ8:** How severely does performance degrade under real-world domain shift (external datasets)?

---

# 30. Priority Matrix

* **Tier 1 (Core Non-Negotiables):** NEU dataset verification, stratified splitting, Custom CNN, EfficientNet frozen, Fine-tuning, quantitative evaluation, confusion matrices, comparison tables.
* **Tier 2 (Strong PBL Differentiators):** Ablation studies, Grad-CAM explainability, error analysis gallery, computational efficiency profiling.
* **Tier 3 (Viva Excellence & Production Demonstration):** Multi-seed variance tests, external validation, FastAPI service, React inspection console.
