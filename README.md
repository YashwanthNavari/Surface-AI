# Surface AI — Manufacturing Surface Defect Inspection Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, commercial-grade industrial computer-vision platform and research workstation for automated hot-rolled strip steel surface defect classification, multi-model consensus arbitration, and gradient-weighted spatial explainability (Grad-CAM).

---

## Benchmark Highlights (NEU-CLS Benchmark)

| Model | Architecture Type | Test Accuracy | Macro F1 | CPU Latency | Throughput | Total Parameters | Disk Size |
|---|---|---|---|---|---|---|---|
| **EfficientNetB0 (Fine-Tuned)** | Transfer Learning (Top 30 conv layers) | **98.89%** | **98.89%** | 25.9 ms | 38.6 FPS | 4,214,313 | 29.2 MB |
| **EfficientNetB0 (Frozen)** | Feature Extraction (ImageNet backbone) | **98.15%** | **98.15%** | 31.9 ms | 31.4 FPS | 4,214,313 | 18.2 MB |
| **Custom 4-Stage CNN** | Deep Learning from Scratch | **94.07%** | **94.03%** | **16.9 ms** | **59.2 FPS** | **422,086** | **4.9 MB** |
| **HOG + GLCM + SVM** | Classical Computer Vision & ML Baseline | 92.22% | 92.16% | 10.7 ms | 93.5 FPS | N/A | N/A |

- **Top Authority Champion**: Fine-Tuned EfficientNetB0 reaches **98.89% test accuracy**, misclassifying only 3 out of 270 test samples.
- **Edge Deployment Champion**: Custom 4-Stage ConvNet achieves **94.07% accuracy** with only **422K parameters** and **16.9 ms latency** (4.9 MB footprint), ideal for embedded vision controllers.

---

## Core Capabilities

- **Dominant Live Inspection**: Fullscreen image specimen viewer with interactive zoom (+/- / 1x reset), view mode toggles (`Original` | `Heatmap` | `Overlay`), live opacity slider, and real test sample thumbnails.
- **Multi-Model Consensus Engine**: Real-time arbitration across all 3 deep learning architectures with automatic agreement verification (`3/3 Consensus Confirmed` or `Arbitration Required`).
- **Explainability Lab (Grad-CAM XAI)**: Visual-first 3-panel pipeline (`Input Surface Matrix` $\to$ `Activation Heatmap` $\to$ `Defect Localization Overlay`) with layer selection and metallurgical domain reasoning.
- **Research & Experiments Workstation**:
  - *Model Performance Matrix*: Precision, recall, and F1 across all 6 defect categories.
  - *Regularization & Ablations*: Verified CNN-A (94.07%), CNN-B (94.07%), CNN-C (25.19% early stopped), and Keras 3 small-batch BatchNorm dynamics.
  - *Interactive SVG Training Curves*: Real loss and accuracy trajectories parsed from historical training logs.
  - *Full Confusion Matrix*: $6 \times 6$ ground truth vs prediction matrix on 270 unseen test images.
  - *Error Post-Mortem*: Complete transparent analysis of the 3 misclassified test split samples (`scratches_96.jpg`, `scratches_44.jpg`, `scratches_69.jpg`).
- **Audit History Stream**: Immutable event timeline logging all inspections to SQLite (`inspections.db`) with defect filtering, live search, and one-click CSV export.
- **Operator vs. Research System Modes**:
  - *Operator Mode*: Distraction-free, streamlined workflow for factory floor inspection.
  - *Research Mode*: Full academic view with statistical tables, loss trajectories, and ablation comparisons.

---

## Defect Taxonomy (NEU Surface Defect Database)

1. **Crazing**: Fine web-like tensile thermal fatigue micro-cracks formed during hot strip roll passes (F1: 100.0%).
2. **Inclusion**: Non-metallic particulate slag or embedded foreign oxides (F1: 96.77%).
3. **Patches**: Localized surface oxidation regions with diffuse boundary profiles (F1: 100.0%).
4. **Pitted Surface**: Porous cavities caused by mechanical roll indentation or localized scale pitting (F1: 100.0%).
5. **Rolled-in Scale**: Heavy iron oxide scale mechanically compressed into steel strip during finish rolling (F1: 100.0%).
6. **Scratches**: Sharp linear abrasive grooves resulting from mechanical friction along roll tables (F1: 96.55%).

---

## Project Structure

```
Surface-AI/
├── backend/
│   ├── main.py                  # FastAPI REST API endpoints & CORS middleware
│   ├── inference.py             # Multi-model prediction pipeline & consensus engine
│   ├── explainability.py        # Grad-CAM tensor gradients & Jet colormap blending
│   └── database.py              # SQLite audit persistence & history logging
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AppShell.jsx             # Shell wrapper with sidebar & topbar
│   │   │   ├── Sidebar.jsx              # Collapsible Left Sidebar navigation
│   │   │   ├── Topbar.jsx               # Breadcrumbs, mode & theme controls
│   │   │   ├── ImageViewer.jsx          # Dominant specimen viewport (zoom/pan/overlay)
│   │   │   ├── MetricCard.jsx           # Reusable KPI cards
│   │   │   ├── ConfidenceBar.jsx        # Model confidence progress meters
│   │   │   ├── StatusBadge.jsx          # Consensus and disposition status pills
│   │   │   ├── MissionControl.jsx       # Executive overview & accuracy vs latency scatter
│   │   │   ├── LiveInspection.jsx       # Hero inspection experience with real thumbnails
│   │   │   ├── ExplainabilityLab.jsx    # 3-panel Grad-CAM studio
│   │   │   ├── ResearchLab.jsx          # 7-tab analytics workstation
│   │   │   └── HistoryStream.jsx        # Event timeline with CSV export
│   │   ├── data/
│   │   │   ├── ablationData.js          # Real ablation metrics & BatchNorm insight
│   │   │   ├── trainingHistoryData.js   # Real epoch-by-epoch loss & accuracy curves
│   │   │   └── errorAnalysisData.js     # Post-mortem on 3 test failure cases
│   │   ├── App.jsx                      # Root application controller
│   │   └── index.css                    # Linear/Vercel design system (Inter, 8px grid)
│   └── package.json
├── models/
│   ├── custom_cnn.keras                 # Trained 4-Stage ConvNet weights (4.9 MB)
│   ├── efficientnet_frozen.keras        # Trained feature extractor weights (18.2 MB)
│   ├── efficientnet_finetuned.keras     # Trained fine-tuned champion weights (29.2 MB)
│   └── class_labels.json                # Class index mappings
├── data/
│   ├── raw/NEU-CLS/                     # 1,800 NEU grayscale surface specimens
│   └── splits/                          # Stratified train (1,260), val (270), test (270) CSVs
├── results/
│   ├── metrics/                         # Per-class evaluation CSVs & JSON summaries
│   ├── training_curves/                 # Epoch-by-epoch training logs
│   ├── confusion_matrices/              # Confusion matrix evaluation outputs
│   └── model_comparison.csv             # Master comparison benchmark table
├── src/
│   ├── data/                            # Dataset ingestion, stratified splitting, augmentation
│   ├── models/                          # Custom CNN & EfficientNet model definitions
│   ├── training/                        # Two-stage training pipelines & callbacks
│   ├── evaluation/                      # Confusion matrix & benchmark evaluation
│   └── explainability/                  # Grad-CAM heatmap generation
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# Clone repository
git clone https://github.com/YashwanthNavari/Surface-AI.git
cd Surface-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate # Windows

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend

# Install Node modules
npm install

# Start Vite development server
npm run dev
```
Open `http://127.0.0.1:5173/` in your browser.

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check & system status |
| `POST` | `/api/v1/predict` | Upload image matrix; returns predictions for all 3 models + consensus |
| `POST` | `/api/v1/explain` | Generate Grad-CAM activation heatmap & overlay for specified model |
| `GET` | `/api/v1/metrics` | Model comparison table & per-class precision/recall/F1 metrics |
| `GET` | `/api/v1/samples` | Curated representative test split specimens for instant testing |
| `GET` | `/api/v1/sample-image/{cls}/{filename}` | Stream test specimen image |
| `GET` | `/api/v1/history` | Chronological audit records from SQLite database |

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
