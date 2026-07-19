# PatchFinders: Road Surface Anomaly Detection using DINOv2

This work done using a deep learning framework for detecting road surface anomalies such as *cracks* and *potholes* using a frozen *DINOv2 Vision Transformer* backbone with lightweight detection heads. The project focuses on improving robustness under varying camera perspectives and environmental conditions through perspective-aware data augmentation.

---

## Features

- DINOv2 ViT-Small backbone with frozen pretrained weights
- Lightweight patch-level detection heads
- Crack and pothole detection
- Perspective-aware data augmentation
- Custom dataset harmonization pipeline
- Grid-based localization framework
- AUROC-based evaluation metrics
- Modular training and inference pipeline

---

## Project Structure

## Project Structure

```text
PatchFinders/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── Saved model checkpoints
│
├── samples/
│   └── Data augmentation examples
│
├── src/
│   ├── augmentations.py
│   ├── dataset.py
│   ├── dino_detector.py
│   ├── harmonize_urbansurface.py
│   ├── predict.py
│   ├── predict_aggregated.py
│   ├── run_pipeline.py
│   ├── train_dino.py
│   └── metrics.py
│
├── experiments/
├── comparison_models/
├── main.py
├── main_notebook.ipynb
├── environment.yml
└── README.md
```

---

## Model Architecture

The framework consists of:

1. *Frozen DINOv2 ViT-Small backbone*
   - Extracts dense patch embeddings from road images.

2. *Patch Classification Head*
   - Predicts:
     - Crack
     - Pothole
     - Background

3. *Bounding Box Regression Head*
   - Predicts normalized bounding box coordinates for each detected anomaly.

The backbone remains frozen during training, allowing efficient learning while reducing computational cost.

---

## Data Pipeline

The training pipeline consists of:

1. Dataset harmonization
2. Image preprocessing
3. Perspective-aware augmentation
4. Dataset loading
5. Model training
6. Evaluation
7. Prediction

Supported annotation format:

- YOLO bounding boxes
- JSON manifest files

---

## Data Augmentation

The project includes several augmentation techniques designed for road imagery:

- Perspective Warp
- Target-aware cropping
- Motion blur
- Light flare simulation
- Night shift simulation

Example augmentation outputs are available inside the samples/ directory.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PatchFinders.git
cd PatchFinders
```

Create the conda environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate patch_finder
```
---

## Running the Project

un the complete workflow:

```bash
python main.py
```

Or execute the notebook:

```bash
jupyter notebook main_notebook.ipynb
```

---

## Training

Training is handled through:

bash
python src/train_dino.py


The training pipeline includes:

- Dataset loading
- Augmentation
- Forward pass
- Loss computation
- Validation
- AUROC evaluation

---

## Evaluation

The framework reports:

- Macro AUROC
- Foreground AUROC
- Per-class AUROC

Metrics are computed using *TorchMetrics*.

---

## Datasets

The framework is designed for road damage datasets containing:

- Road images
- YOLO annotations
- Crack labels
- Pothole labels

The dataset harmonization utility converts datasets into a unified JSON format for training.

---

## Dependencies

Major libraries include:

- Python 3.10
- PyTorch
- TorchVision
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- TorchMetrics
- Jupyter Notebook

---

## Research Motivation

Road surface anomalies significantly affect driving safety and maintenance costs. Manual inspection is time-consuming and difficult to scale. PatchFinders addresses this challenge by combining self-supervised visual representations from DINOv2 with lightweight detection heads and perspective-aware augmentation to improve robustness across varying viewpoints and environmental conditions.

---

## Future Improvements

- Multi-scale detection
- Real-time inference
- Domain adaptation
- Additional road damage categories
- Temporal video-based detection
- ONNX/TensorRT deployment
- Mobile deployment