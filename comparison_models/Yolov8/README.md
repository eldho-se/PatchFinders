# PatchFinders — Road Damage Detection

Cross-domain road damage detection from dashcam to pedestrian viewpoints.

## Setup
```bash
conda activate patchfinders
pip install -r requirements.txt
pip install -e .
```

## Usage

### Unified CLI (Recommended)
```bash
python main.py train --model m --epochs 100 --batch 16
python main.py evaluate --weights runs/train/yolov8m_baseline/weights/best.pt
python main.py predict --source data/processed/test/in_distribution/images
```

### Jupyter Evaluation Notebook
Run evaluation interactively in Jupyter:
- [main_notebook.ipynb](file:///Users/edwinsjohn/Downloads/Eldhose_patch_finder/main_notebook.ipynb)

### Module Direct Execution
```bash
python -m src.cli.train --model m --epochs 100 --batch 16
python -m src.cli.evaluate --weights runs/train/yolov8m_baseline/weights/best.pt
python -m src.cli.predict --source data/processed/test/in_distribution/images
```

## Repository Structure
```
.
├── configs/            # Dataset and model YAML configurations
├── data/               # Raw, processed, and augmented datasets
├── outputs/            # Predictions, evaluation metrics, and logs
├── runs/               # Training run checkpoints and logs
├── scripts/            # Setup and categorized analysis scripts
│   ├── setup.sh        # Shell setup script
│   ├── data/           # Dataset downloading, harmonization, and verification
│   ├── evaluation/     # OOD and AUROC evaluation scripts
│   └── visualization/  # Plotting and visualization tools
├── src/                # Core Python package (patchfinders)
│   ├── cli/            # CLI subcommands (train, evaluate, predict)
│   ├── data/           # Data processing and augmentation modules
│   ├── evaluation/     # Evaluator and OOD evaluation modules
│   ├── inference/      # Predictor module
│   ├── models/         # Model wrappers (YOLOv8)
│   ├── training/       # Training pipeline handlers
│   ├── utils/          # Logging, config parsers, and utilities
│   └── visualization/  # Plotting utilities
├── tests/              # Unit tests
├── weights/            # Pretrained model weights (.pt)
├── main.py             # Sole root CLI entry point
└── main_notebook.ipynb # Interactive evaluation Jupyter Notebook
```