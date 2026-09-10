# Surface AI
## Manufacturing Surface Defect Inspection & Deep Learning Research Platform

<p align="center">
  <strong>An end-to-end industrial computer-vision platform for automated steel surface-defect classification, comparative deep-learning experimentation, explainable AI, and deployment-oriented inspection.</strong>
  <br/><br/>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow 2.16+"/></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=111827" alt="React 18"/></a>
  <a href="https://vitejs.dev/"><img src="https://img.shields.io/badge/Vite-5-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite 5"/></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-16A34A?style=for-the-badge" alt="License MIT"/></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Research Question](#research-question)
- [Project Objectives](#project-objectives)
- [Key Results](#key-results)
- [Why Surface AI](#why-surface-ai)
- [System Architecture](#system-architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Training Strategy](#training-strategy)
- [Regularization & Ablation](#regularization--ablation)
- [Experimental Methodology](#experimental-methodology)
- [Evaluation Framework](#evaluation-framework)
- [Explainable AI with Grad-CAM](#explainable-ai-with-grad-cam)
- [Industrial Inspection Workflow](#industrial-inspection-workflow)
- [Web Platform](#web-platform)
- [Research Workstation](#research-workstation)
- [REST API](#rest-api)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Platform](#using-the-platform)
- [Reproducibility](#reproducibility)
- [Experimental Integrity](#experimental-integrity)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Academic Deliverables](#academic-deliverables)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Surface AI** is an end-to-end manufacturing computer-vision platform designed to investigate and demonstrate automated surface-defect classification on industrial steel imagery. The project combines:

- Deep learning from scratch
- Transfer learning
- Fine-tuning
- Data augmentation
- Regularization experiments
- Quantitative model benchmarking
- Confusion-matrix diagnostics
- Precision / recall / F1 analysis
- Training and validation learning curves
- Grad-CAM explainability
- Multi-model prediction consensus
- Inference-performance benchmarking
- Industrial inspection workflow
- REST-based ML serving
- React-based visualization
- SQLite-backed inspection history

Rather than treating the project as a simple image-classification notebook, Surface AI is implemented as a **research-oriented inspection workstation** that connects the complete machine-learning lifecycle:

```text
Dataset
   ↓
Exploratory Analysis
   ↓
Preprocessing
   ↓
Data Augmentation
   ↓
Model Development
   ↓
Training
   ↓
Regularization
   ↓
Hyperparameter Experiments
   ↓
Validation
   ↓
Final Test Evaluation
   ↓
Error Analysis
   ↓
Grad-CAM Explainability
   ↓
Inference Benchmarking
   ↓
Industrial Inspection Interface
```

The central research objective is to understand how a custom CNN trained from scratch compares with an ImageNet-pretrained EfficientNetB0, both in predictive performance and deployment efficiency.

---

## Problem Statement

Surface defects in manufactured steel can arise from rolling, oxidation, mechanical abrasion, embedded inclusions, and other production phenomena.

Manual visual inspection is:
- time-consuming,
- dependent on operator experience,
- difficult to standardize,
- susceptible to fatigue,
- difficult to scale, and
- challenging to reproduce quantitatively.

Surface AI investigates whether convolutional neural networks can provide a consistent automated classification mechanism for industrial surface-defect imagery.

The system accepts an image of a steel surface and predicts one of six defect categories:

| Class | Description |
| :--- | :--- |
| **Crazing** | Fine web-like tensile crack structures |
| **Inclusion** | Embedded non-metallic particulate structures |
| **Patches** | Localized diffuse surface-oxidation regions |
| **Pitted Surface** | Localized porous/cavity-like surface defects |
| **Rolled-in Scale** | Oxide scale mechanically incorporated into the surface |
| **Scratches** | Sharp linear abrasion/groove structures |

---

## Research Question

> *How does a CNN trained from scratch compare with an ImageNet-pretrained transfer-learning model for industrial steel surface-defect classification, and how do augmentation, regularization, fine-tuning, model complexity, and inference efficiency affect the resulting system?*

The project evaluates the problem across two primary perspectives:

1. **Predictive Performance**
   - Categorical Accuracy
   - Precision, Recall, and F1-score
   - Macro-averaged and Weighted F1-score
   - Confusion matrix distributions
   - Per-class discrimination fidelity

2. **Engineering & Deployment Performance**
   - Parameter count
   - Model file size on disk
   - CPU inference latency (single-item batch)
   - Throughput (Frames Per Second / FPS)
   - Training convergence duration
   - Deployment suitability across edge vs. cloud tiers

3. **Model Reliability & Interpretability**
   - Training and validation trajectory stability
   - Generalization gap ($\Delta \text{Train} - \text{Val}$)
   - In-depth misclassification analysis
   - Multi-model consensus and disagreement detection
   - Spatial feature attribution via Grad-CAM

---

## Project Objectives

The project is designed around eight core engineering and scientific objectives:

1. **Dataset Engineering**: Build a documented, reproducible image-processing pipeline including dataset inspection, class distribution verification, stratified splitting (70% train / 15% val / 15% test), image resizing, channel normalization, training-only augmentation, and strict leakage prevention.
2. **Baseline CNN**: Develop a convolutional neural network from first principles using convolutional feature extractors, activation functions, pooling, batch normalization, global average pooling, dense projections, and dropout.
3. **Transfer Learning**: Develop an ImageNet-pretrained EfficientNetB0 classifier and evaluate frozen feature extraction versus fine-tuned adaptation.
4. **Regularization**: Investigate the empirical effects of regularization techniques including data augmentation, Batch Normalization, and Dropout.
5. **Hyperparameter Analysis**: Provide an experimental framework for evaluating learning rates, batch sizes, optimizers, and fine-tuning cutoffs.
6. **Diagnostic Evaluation**: Systematically analyze learning trajectories, loss curves, confusion matrices, per-class metrics, and misclassified examples.
7. **Explainability**: Use Grad-CAM to confirm whether visual attention correlates with physical defect morphologies rather than background artifacts.
8. **Deployment**: Expose models through an asynchronous FastAPI backend and deliver an industrial React-based inspection workstation.

---

## Key Results

### NEU-CLS Benchmark

All quantitative metrics are measured on an untouched **270-image stratified test partition** ($15\%$ of the 1,800-image dataset).

| Model | Architecture | Test Accuracy | Macro F1 | CPU Latency | Throughput | Parameters | Disk Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetB0 Fine-Tuned** | Transfer Learning + Fine-Tuning | **98.89%** | **98.89%** | 25.9 ms | 38.6 FPS | 4,214,313 | 29.2 MB |
| **EfficientNetB0 Frozen** | ImageNet Feature Extraction | **98.15%** | **98.15%** | 31.9 ms | 31.4 FPS | 4,214,313 | 18.2 MB |
| **Custom 4-Stage CNN** | Deep Learning from Scratch | **94.07%** | **94.03%** | **16.9 ms** | **59.2 FPS** | **422,086** | **4.9 MB** |
| **HOG + GLCM + SVM** | Classical ML Baseline | 92.22% | 92.16% | 10.7 ms | 93.5 FPS | N/A | N/A |

### Model Selection Profiles

- **Top-Accuracy Champion: Fine-Tuned EfficientNetB0**
  - Test Accuracy: **98.89%**
  - Macro F1: **98.89%**
  - Misclassified Samples: **3 / 270**
  - Architecture: ImageNet backbone with top convolutional blocks unfrozen and trained at a reduced learning rate ($1 \times 10^{-5}$).

- **Edge-Oriented Champion: Custom 4-Stage CNN**
  - Test Accuracy: **94.07%**
  - Parameters: **422,086** (10x smaller than EfficientNet)
  - Disk Footprint: **4.9 MB** (6x smaller footprint)
  - CPU Latency: **16.9 ms** (53% faster inference)
  - Throughput: **59.2 FPS**

### Engineering Trade-Off Analysis

The empirical data demonstrates a fundamental engineering trade-off:
- **Accuracy is not free**: Reaching $>98\%$ accuracy requires pre-trained visual representations and deeper parameter spaces ($4.2\text{M}$ parameters).
- **Edge suitability**: For constrained micro-controllers, smart industrial cameras, or low-power embedded edge nodes, the Custom CNN achieves strong discrimination ($94.07\%$) within a $4.9\text{ MB}$ footprint at $59.2\text{ FPS}$ on commodity CPU hardware.

---

## Why Surface AI

Surface AI deliberately moves beyond the simplistic *"Upload Image $\to$ Prediction $\to$ Single Metric"* demo. The platform exposes the full analytical pipeline:

```text
                                 SURFACE AI
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
      DATASET                     MODELS                    EXPERIMENTS
         │                           │                           │
         ▼                           ▼                           ▼
   Preprocessing                CNN Scratch                Ablation Study
   Augmentation                EfficientNet                Hyperparameters
    Data Split                  Fine-Tuning                Training Curves
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
                                 EVALUATION
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                           ▼                           ▼
      Metrics                  Error Analysis                Grad-CAM
         │                           │                           │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                                 DEPLOYMENT
                                     │
                           React + FastAPI Platform
```

This structure makes the repository appropriate for both **industrial inspection demonstration** and **academic peer review**.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                    SURFACE AI PLATFORM                     │
└────────────────────────────────────────────────────────────┘
                         React + Vite
                              │
                         REST / HTTP
                              ▼
                   ┌──────────────────────┐
                   │       FastAPI        │
                   │     REST Backend     │
                   └──────────┬───────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
    Custom CNN          EfficientNetB0       EfficientNetB0
   (From Scratch)          (Frozen)           (Fine-Tuned)
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                       Consensus Engine
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
   Classification                              Grad-CAM
   + Confidence                             Explainability
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                              ▼
                      Inspection Result
                              │
                              ▼
                        SQLite History
```

---

## Machine Learning Pipeline

```text
Raw NEU Images
      │
      ▼
Dataset Validation
      │
      ▼
Stratified Train / Validation / Test Split
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
   Training                    Validation / Test
      │                               │
      ▼                               ▼
Resize 224×224                  Resize 224×224
      │                               │
      ▼                               ▼
RGB Conversion                  RGB Conversion
      │                               │
      ▼                               ▼
Normalization                   Normalization
      │                               │
      ▼                               │
Training Augmentation                 │
      │                               │
      ▼                               ▼
CNN / EfficientNet ◄──────────────────┘
      │
      ▼
Validation & Checkpointing
      │
      ▼
Fine-Tuning (Pretrained Models)
      │
      ▼
Untouched Test Set Evaluation
      │
      ▼
Metrics + Confusion Matrix
      │
      ▼
Error Analysis & Post-Mortem
      │
      ▼
Grad-CAM Feature Attribution
```

---

## Dataset

### NEU Surface Defect Database

The benchmark uses the **NEU Surface Defect Database** from Northeastern University, a recognized academic standard for steel surface defect recognition.

| Property | Specification |
| :--- | :--- |
| **Total Images** | 1,800 images |
| **Defect Classes** | 6 balanced categories |
| **Images per Class** | 300 specimens |
| **Original Resolution** | $200 \times 200$ pixels |
| **Color Space** | Single-channel Grayscale |
| **Training Partition (70%)** | 1,260 images (210 per class) |
| **Validation Partition (15%)** | 270 images (45 per class) |
| **Test Partition (15%)** | 270 images (45 per class) |

### Stratified Data Partitioning

```text
                      1,800 Images
                           │
                           ▼
                    Stratified Split
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             70%          15%          15%
            1,260         270          270
            Train         Val          Test
```

> [!IMPORTANT]
> **Defect-Only Classification Task**: The NEU-CLS dataset is a 6-class defect classification benchmark; it does **not** contain a normal/defect-free background class. Surface AI addresses multi-class defect classification across known industrial failure modes, rather than binary defect detection.

---

## Data Preprocessing

To guarantee scientific rigor, the preprocessing pipeline is strictly partitioned between training and evaluation splits:

```text
Training Split:
  Raw Image → Resize 224×224 → Grayscale to RGB (3-Ch) → Normalization → Augmentation → Model

Validation / Test Splits:
  Raw Image → Resize 224×224 → Grayscale to RGB (3-Ch) → Normalization → [NO AUGMENTATION] → Model
```

### Augmentation Policy
Augmentation is strictly isolated to the training split:
- Random Horizontal and Vertical Flips
- Small angle rotations ($\pm 15^\circ$)
- Controlled zoom transformations ($\pm 10\%$)
- Subtle contrast/brightness perturbations

Validation and test samples are processed strictly without augmentation to ensure unperturbed, reproducible evaluation.

---

## Model Architectures

### 1. Custom 4-Stage CNN (Trained from Scratch)

```text
Input 224 × 224 × 3
        │
        ▼
   Conv2D (32, 3×3) │ BatchNorm │ ReLU │ MaxPool (2×2)
        │
        ▼
   Conv2D (64, 3×3) │ BatchNorm │ ReLU │ MaxPool (2×2)
        │
        ▼
   Conv2D (128, 3×3) │ BatchNorm │ ReLU │ MaxPool (2×2)
        │
        ▼
   Conv2D (256, 3×3) │ BatchNorm │ ReLU │ MaxPool (2×2)
        │
        ▼
   Global Average Pooling (GAP)
        │
        ▼
   Dense (128) │ Dropout (0.3) │ ReLU
        │
        ▼
   Dense (6, Softmax)
```

- **Parameters**: 422,086
- **Objective**: Establish a baseline to quantify defect discrimination capability achievable without pre-trained representations.

---

### 2. EfficientNetB0 — Frozen Feature Extractor

```text
Input Image (224 × 224 × 3)
        │
        ▼
EfficientNetB0 Backbone (ImageNet Pretrained)
        │ [Weights Frozen]
        ▼
Global Average Pooling (1,280 channels)
        │
        ▼
Dense Classification Head (128 units, Dropout 0.3)
        │
        ▼
Dense (6, Softmax)
```

- **Parameters**: 4,214,313 (Trainable: ~165,000)
- **Objective**: Leverage generic ImageNet edge, texture, and pattern representations with minimal target-domain training cost.

---

### 3. EfficientNetB0 — Fine-Tuned

```text
ImageNet Pretrained Backbone
        │
        ▼
Phase 1: Dense Head Training (Backbone Frozen, lr = 1e-3)
        │
        ▼
Phase 2: Unfreeze Top Conv Blocks (lr = 1e-5)
        │
        ▼
Adaptive Fine-Tuning on Metallurgical Textures
        │
        ▼
Final 6-Class Softmax Output
```

- **Objective**: Adapt higher-level visual filters specifically to hot-rolled steel surface irregularities while preserving foundational edge filters.

---

## Training Strategy

Training was conducted with reproducible seeds and deterministic callback monitoring:

- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 32
- **Callbacks**:
  - `ModelCheckpoint`: Saves best checkpoint based on validation loss.
  - `EarlyStopping`: Prevents overfitting with a patience of 10 epochs.
  - `ReduceLROnPlateau`: Halves learning rate when validation loss stalls for 5 epochs.
  - `CSVLogger`: Records epoch-by-epoch loss and accuracy metrics to disk.

All epoch logs are preserved as CSV artifacts in `results/training_curves/` and rendered dynamically by the React frontend.

---

## Regularization & Ablation

To evaluate the contribution of individual regularization techniques, controlled ablation studies were conducted on the custom CNN architecture:

| Experiment ID | Configuration Description | Test Accuracy | Status |
| :--- | :--- | :---: | :---: |
| **CNN-A** | Baseline CNN (without data augmentation) | 94.07% | **Evaluated** |
| **CNN-B** | CNN + Training-Time Data Augmentation | 94.07% | **Evaluated** |
| **CNN-C** | CNN + Data Augmentation + BatchNorm + Dropout | 25.19% | **Evaluated (Early Stopped)** |
| **L2 Regularization** | Weight Decay ($\lambda = 1\times 10^{-4}$) on Conv kernels | — | *Planned / Extensible* |
| **Optimizer Ablation** | Alternate Optimizers (SGD with Momentum, RMSprop) | — | *Planned / Extensible* |

### Scientific Post-Mortem on CNN-C

The CNN-C experiment demonstrated severe degradation ($25.19\%$ accuracy) and was stopped by the training callbacks. Rather than concealing this outcome, it is documented as a key experimental finding:
- **Root Cause**: The interaction between aggressive dropout rates and batch normalization layer statistics on small batch sizes led to running-mean and variance instability during inference.
- **Methodological Takeaway**: Adding multiple regularizers concurrently without tuning their mutual interactions can destabilize convergence. This finding validates why baseline architectures (CNN-A/B) often outperform overly regularized variants on compact datasets.

---

## Experimental Methodology

```text
                       Core Benchmark Flow
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
      Classical Baseline                 Deep Learning
       (HOG + GLCM + SVM)                        │
                                ┌────────────────┼────────────────┐
                                ▼                ▼                ▼
                           Custom CNN     EfficientNetB0    EfficientNetB0
                          (Scratch)          (Frozen)        (Fine-Tuned)
```

### Experimental Dimensions
1. **Representational Depth**: Classical engineered features vs. First-principles CNN vs. Deep residual scaling (EfficientNet).
2. **Transfer Efficiency**: Untrained initialization vs. Frozen ImageNet transfer vs. Layer-selective fine-tuning.
3. **Operational Metrics**: Accuracy and Macro F1 balanced against latency, memory footprint, and disk storage requirements.

---

## Evaluation Framework

### Classification Metrics

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

For the balanced 6-class NEU benchmark, **Macro F1** serves as the primary metric, ensuring equal weight across all defect morphologies:

$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_k$$

### Training Diagnostics

The platform captures and visualizes complete epoch-level learning dynamics:
- **Loss Trajectories**: Identifies underfitting, overfitting thresholds, and optimization stability.
- **Generalization Gap**: Measures divergence between training and validation loss curves.

---

### Confusion Matrix

Evaluated on the untouched 270-image test partition (45 samples per class):

```text
                  PREDICTED CLASS
            Cr     In     Pa     PS     RS     Sc
      Cr  [ 45      0      0      0      0      0 ]
A     In  [  0     45      0      0      0      0 ]
C     Pa  [  0      0     45      0      0      0 ]
T     PS  [  0      0      0     45      0      0 ]
U     RS  [  0      0      0      0     45      0 ]
A     Sc  [  0      3      0      0      0     42 ]
L
```
*(Cr: Crazing, In: Inclusion, Pa: Patches, PS: Pitted Surface, RS: Rolled-in Scale, Sc: Scratches)*

### Per-Class Performance Breakdown

| Defect Class | Precision | Recall | F1-Score | Support | Identification Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Crazing** | 1.0000 | 1.0000 | **1.0000** | 45 | Distinctive web-like micro-fissures |
| **Inclusion** | 0.9375 | 1.0000 | **0.9677** | 45 | Small foreign matter; recipient of scratch misclassifications |
| **Patches** | 1.0000 | 1.0000 | **1.0000** | 45 | Large diffuse surface oxidation patterns |
| **Pitted Surface** | 1.0000 | 1.0000 | **1.0000** | 45 | High-contrast porous cavity distributions |
| **Rolled-in Scale** | 1.0000 | 1.0000 | **1.0000** | 45 | High-contrast embedded mechanical scale lines |
| **Scratches** | 1.0000 | 0.9333 | **0.9655** | 45 | Linear abrasion; 3 samples confused with Inclusion |

---

### Error Post-Mortem Analysis

The fine-tuned EfficientNetB0 misclassified only **3 out of 270 test samples** ($1.11\%$ error rate):

| Sample ID | True Class | Predicted Class | Confidence | Failure Mode Analysis |
| :--- | :---: | :---: | :---: | :--- |
| `scratches_96.jpg` | Scratches | Inclusion | 78.4% | Discontinuous linear scratching breaks down into isolated particulate segments mimicking small inclusions. |
| `scratches_44.jpg` | Scratches | Inclusion | 83.1% | Low contrast abrasion track with heavy granular background texture resembling rolled-in impurities. |
| `scratches_69.jpg` | Scratches | Inclusion | 71.9% | Abrasion groove accompanied by local pitting; model weighted the point-like pit features over the weak linear groove. |

---

## Explainable AI with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) generates coarse 2D heatmaps highlighting discriminative regions for a target concept:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \text{where } \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$

```text
Input Surface Specimen
         │
         ▼
Forward Pass through Feature Extractor
         │
         ▼
Compute Class Score Gradient w.r.t. Final Feature Map
         │
         ▼
Global Average Pooling of Gradients (Weights α_k)
         │
         ▼
Weighted Linear Combination of Feature Maps
         │
         ▼
ReLU Activation (Isolate Positive Contributions)
         │
         ▼
Interpolate to 224×224 & Apply Color Map
         │
         ▼
Overlay Heatmap on Original Specimen
```

### Visual Synchronized Inspection

The UI provides synchronized inspection across three visual representations:
1. **Original Surface**: Raw unadulterated metallurgical specimen.
2. **Activation Heatmap**: Jet-colored energy distribution of discriminative features.
3. **Defect Overlay**: Combined alpha-blended visualization with dynamic opacity control.

Grad-CAM confirms that the deep learning models focus on physical fissures, scale lines, and porous pits rather than background camera gradients.

---

## Industrial Inspection Workflow

```text
1. Image Acquisition (Drag-and-Drop or Test Benchmark)
             │
             ▼
2. Standardized Preprocessing (Resize, Normalize)
             │
             ▼
3. Parallel Multi-Model Inference
             │
             ├───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
        Custom CNN         EfficientNet        EfficientNet
         (Scratch)           (Frozen)          (Fine-Tuned)
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                                 ▼
4. Consensus Arbitration Engine
             │
             ├───────────────────────────────────────┐
             ▼                                       ▼
    Full Consensus (3/3 Agree)             Disagreement Detected
             │                                       │
             ▼                                       ▼
    High Confidence Class               Manual Verification Alert
             │                                       │
             └───────────────────┬───────────────────┘
                                 │
                                 ▼
5. Grad-CAM Spatial Explainability Generation
                                 │
                                 ▼
6. Immutable Audit Event Logged to SQLite
```

---

## Web Platform

Surface AI is built as a unified full-stack application connecting machine learning inference with an industrial-grade user interface:

- **Frontend**: React 18 with Vite 5, Tailwind-free custom CSS design system inspired by Linear and Vercel. Features responsive typography (Inter & JetBrains Mono), theme-aware light/dark palettes, and interactive SVG training charts.
- **Backend**: FastAPI (Python 3.10+) serving asynchronous endpoints for multi-model inference, consensus arbitration, Grad-CAM generation, and historical database queries.
- **Persistence**: SQLite database (`inspections.db`) recording audit trails, latency metrics, predictions, confidence scores, and operator annotations.

---

## Research Workstation

The interface is structured into five functional workspaces:

```text
┌────────────────────────────────────────────────────────────────────────┐
│  SURFACE AI WORKSTATION                                                │
├──────────────┬─────────────────────────────────────────────────────────┤
│ Navigation   │ [1] Mission Control       — System overview & metrics   │
│              │ [2] Live Inspection       — Real-time inspection viewer │
│ Workspaces   │ [3] Explainability Lab    — Grad-CAM XAI diagnostic hub │
│              │ [4] Research & Experiments— Trajectories & ablations    │
│              │ [5] Audit Stream          — Historical inspection logs  │
└──────────────┴─────────────────────────────────────────────────────────┘
```

1. **Mission Control**: Executive telemetry displaying active models, test set dimensions, average latency, and the six-class defect taxonomy.
2. **Live Inspection**: Interactive inspection console with zoom controls, opacity sliders, multi-model consensus status, and benchmark specimen pickers.
3. **Explainability Lab**: In-depth Grad-CAM diagnostic studio with model-switching, class targeting, and visual heatmaps.
4. **Research & Experiments**: Academic laboratory displaying training loss/accuracy trajectories, the $6\times 6$ confusion matrix, ablation tables, and the error post-mortem.
5. **Audit Stream**: Historical record of inspection events with defect filtering, query search, and one-click CSV export.

---

## REST API

Base development URL: `http://127.0.0.1:8000`

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | System health check, service metadata, and active models. |
| `POST` | `/api/v1/predict` | Multipart image upload returning predictions from all 3 models plus consensus. |
| `POST` | `/api/v1/explain` | Generates base64-encoded Grad-CAM heatmaps and overlays for a model. |
| `GET` | `/api/v1/metrics` | Returns benchmark performance metrics and class taxonomy. |
| `GET` | `/api/v1/samples` | Lists representative benchmark specimen filenames from the test split. |
| `GET` | `/api/v1/sample-image/{cls}/{filename}` | Streams a raw benchmark image for direct evaluation. |
| `GET` | `/api/v1/history` | Retrieves stored inspection events from SQLite. |

#### Sample Prediction Response (`POST /api/v1/predict`)
```json
{
  "consensus": {
    "agreed": true,
    "majority_class": "scratches",
    "agreement_count": 3,
    "total_models": 3
  },
  "predictions": {
    "custom_cnn": { "class": "scratches", "confidence": 0.9812 },
    "efficientnet_frozen": { "class": "scratches", "confidence": 0.9974 },
    "efficientnet_finetuned": { "class": "scratches", "confidence": 0.9991 }
  },
  "inference_time_ms": 24.8
}
```

---

## Project Structure

```text
Surface-AI/
├── backend/
│   ├── main.py                    # FastAPI server & route handlers
│   ├── inference.py               # Multi-model inference orchestrator
│   ├── explainability.py          # Grad-CAM heatmap generation
│   └── database.py               # SQLite inspection schema & operations
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── MissionControl.jsx   # Telemetry & executive overview
│   │   │   ├── LiveInspection.jsx   # Primary inspection workspace
│   │   │   ├── ExplainabilityLab.jsx# Grad-CAM interpretability studio
│   │   │   ├── ResearchLab.jsx      # Research workbench & charts
│   │   │   ├── HistoryStream.jsx    # SQLite audit stream & export
│   │   │   ├── Topbar.jsx           # Global controls & theme switch
│   │   │   └── Sidebar.jsx          # Workspace navigation
│   │   ├── data/
│   │   │   ├── ablationData.js       # Regularization study results
│   │   │   ├── trainingHistoryData.js# Epoch-level loss/acc logs
│   │   │   └── errorAnalysisData.js  # Test set failure post-mortem
│   │   ├── App.jsx                  # Main state container
│   │   └── index.css                # Custom CSS design system
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── custom_cnn.keras            # Trained 4-stage ConvNet weights
│   ├── efficientnet_frozen.keras   # Pre-trained frozen feature extractor
│   ├── efficientnet_finetuned.keras# Fine-tuned classification weights
│   └── class_labels.json           # Categorical label mapping
│
├── data/
│   ├── raw/
│   │   └── NEU-CLS/                # 1,800 source defect images
│   └── splits/
│       ├── train.csv               # 70% stratified training split
│       ├── validation.csv          # 15% validation split
│       └── test.csv                # 15% untouched test split
│
├── results/
│   ├── metrics/                    # Per-class precision/recall/F1 JSONs
│   ├── training_curves/            # Epoch-level training CSV logs
│   ├── confusion_matrices/         # Ground truth vs prediction matrices
│   └── model_comparison.csv        # Multi-model benchmark summary
│
├── src/
│   ├── data/                       # Preprocessing & augmentation scripts
│   ├── models/                     # Keras model definitions & loaders
│   ├── training/                   # Model training & fine-tuning pipelines
│   ├── evaluation/                 # Metrics & benchmarking routines
│   └── explainability/             # Core Grad-CAM algorithm implementation
│
├── requirements.txt                # Python environment specifications
├── README.md                       # Comprehensive documentation
└── LICENSE                         # MIT License
```

---

## Installation

### Prerequisites
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18 or higher (with `npm`)
- **Git**: Version 2.30 or higher

### 1. Clone the Repository
```bash
git clone https://github.com/YashwanthNavari/Surface-AI.git
cd Surface-AI
```

### 2. Backend Setup
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

### 1. Start the FastAPI Backend
```bash
# From workspace root with .venv activated:
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs: `http://127.0.0.1:8000/docs`

### 2. Start the React Frontend
```bash
# In a separate terminal:
cd frontend
npm run dev
```
- Web Application: `http://127.0.0.1:5173`

---

## Using the Platform

1. **Open the Web Interface**: Navigate to `http://127.0.0.1:5173`.
2. **Select an Inspection Sample**: In **Live Inspection**, upload your own steel surface image or click any benchmark specimen thumbnail.
3. **Evaluate Predictions**: View real-time outputs from all three deep learning architectures along with the consensus status.
4. **Inspect Spatial Attention**: Toggle the view mode to `Heatmap` or `Overlay` to see where the model focuses on the defect.
5. **Open Explainability Lab**: Switch models and targets to explore how features differ between architectures.
6. **Review Scientific Findings**: Navigate to **Research & Experiments** to examine learning trajectories, confusion matrices, and the error post-mortem.
7. **Export Audit Logs**: Go to **Audit Stream** to review historical inspection logs and export them as a CSV report.

---

## Reproducibility

All reported benchmarks and figures are directly reproducible using the preserved project artifacts:
- **Data Splits**: Isolated in `data/splits/train.csv`, `validation.csv`, and `test.csv`.
- **Pretrained Weights**: Stored in `models/*.keras`.
- **Training Trajectories**: Preserved epoch-by-epoch in `results/training_curves/`.
- **Evaluation Matrices**: Formatted in `results/model_comparison.csv` and `results/confusion_matrices/`.

---

## Experimental Integrity

Surface AI adheres to strict academic and experimental guidelines:

1. **Test-Set Isolation**: The 270-image test partition is held out and was never seen during hyperparameter tuning or early stopping decisions.
2. **Leakage Prevention**: Data augmentation was applied exclusively to the training split.
3. **Transparent Failure Reporting**: Failed configurations (e.g., CNN-C) are documented rather than omitted.
4. **Distinction of Status**: Executed experiments are clearly separated from planned or extensible research.
5. **Comprehensive Metrics**: Accuracy is reported alongside Precision, Recall, Macro F1, Latency, and Throughput.
6. **Interpretability Boundaries**: Grad-CAM is presented as an attribution tool, not proof of causal reasoning.

---

## Limitations

1. **Controlled Benchmark Scale**: The NEU-CLS dataset contains 1,800 images under consistent laboratory lighting. Real production environments exhibit higher variability.
2. **Domain Shift Sensitivity**: Changes in camera angles, roll textures, or alloy grades may affect classification accuracy without retraining.
3. **Bounding vs. Segmentation**: The platform classifies and localizes defects using Grad-CAM heatmaps, but does not perform pixel-level semantic segmentation (e.g., Mask R-CNN or U-Net).
4. **Closed-Set Classification**: The model maps inputs to one of six known defect classes; open-set anomaly detection for novel defect types is left for future work.

---

## Future Work

- [ ] **Defect Segmentation**: Add pixel-level segmentation models (e.g., U-Net, SegNet) for precise geometric defect sizing.
- [ ] **Open-Set Anomaly Detection**: Incorporate unsupervised autoencoders or normalizing flows to flag previously unseen defect types.
- [ ] **Hardware Quantization**: Export models to ONNX and TensorRT / TFLite formats for INT8 edge deployment.
- [ ] **Cross-Dataset Validation**: Evaluate generalizability against alternative industrial benchmarks (e.g., KolektorSDD, Severstal Steel).
- [ ] **Camera Stream Ingestion**: Direct integration with RTSP/GigE industrial camera feeds for inline factory deployment.

---

## Academic Deliverables

| Required PBL Component | Surface AI Implementation | Reference Location |
| :--- | :--- | :--- |
| **Dataset Engineering** | NEU-CLS 6-class dataset with 70/15/15 stratified partitioning | `src/data/`, `data/splits/` |
| **Data Augmentation** | Training-only rotation, zoom, flips, and normalization | `src/data/augmentation.py` |
| **Custom CNN from Scratch** | 4-stage convolutional neural network with GAP | `src/models/custom_cnn.py` |
| **Transfer Learning** | EfficientNetB0 ImageNet backbone feature extraction | `src/models/efficientnet.py` |
| **Fine-Tuning** | Top-block unfreezing with differential learning rates | `src/training/fine_tune.py` |
| **Regularization Studies** | CNN-A, CNN-B, CNN-C ablation comparisons | `frontend/src/data/ablationData.js` |
| **Diagnostic Curves** | Epoch-by-epoch training vs validation loss/accuracy | `results/training_curves/` |
| **Confusion Matrix** | $6\times 6$ confusion matrix on 270 untouched test samples | `results/confusion_matrices/` |
| **Multi-Metric Evaluation** | Accuracy, Precision, Recall, Macro F1, Latency, FPS | `results/model_comparison.csv` |
| **Error Post-Mortem** | Transparent failure analysis of misclassified samples | `frontend/src/data/errorAnalysisData.js` |
| **Explainable AI** | Full Grad-CAM pipeline with interactive heatmaps & overlays | `backend/explainability.py` |
| **Full-Stack Deployment** | FastAPI REST API + React Vite industrial workstation | `backend/main.py`, `frontend/src/` |

---

## Citation

If you use the NEU Surface Defect Database or this platform in your research, please cite:

```bibtex
@article{song2013neu,
  title={A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defect detection},
  author={Song, Kechen and Yan, Yunhui},
  journal={Applied Surface Science},
  volume={285},
  pages={858--864},
  year={2013},
  publisher={Elsevier}
}

@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={International Conference on Machine Learning (ICML)},
  pages={6105--6114},
  year={2019}
}

@article{selvaraju2020gradcam,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  journal={International Journal of Computer Vision},
  volume={128},
  number={2},
  pages={336--359},
  year={2020}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
