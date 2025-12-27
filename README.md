# 6D Pose Estimation with Geometric Constraints

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SFR-Vision/6d-pose-estimation/blob/main/colab_notebook.ipynb)

Real-time 6D object pose estimation using YOLO detection and deep learning with geometric constraints on the LineMOD dataset.

## Features

- **Four Model Variants**: RGB, RGB-Geometric, RGBD, RGBD-Geometric
- **Geometric Translation**: Pinhole camera model reduces learned parameters
- **YOLOv8 Detection**: Fast object detection for real-time performance
- **LineMOD Dataset**: 13 object categories with ground truth 6D poses
- **ADD/ADD-S Metrics**: Standard pose estimation evaluation

## Model Architecture

| Model | Input | Translation | Learned Params |
|-------|-------|-------------|----------------|
| RGB | RGB only | Fully learned | 7 (quat + xyz) |
| RGB-Geometric | RGB + camera | Z learned, XY geometric | 5 (quat + z) |
| RGBD | RGB + Depth | Fully learned | 7 (quat + xyz) |
| RGBD-Geometric | RGB + Depth + camera | XYZ from depth sensor | 4 (quat only) |

**Key Insight**: Geometric models incorporate camera intrinsics as inductive bias for better translation estimation.

## Trained Model Results

| Model | Best Epoch | ADD-0.1d Accuracy |
|-------|------------|-------------------|
| RGB | 72 | 13.35% |
| RGB-Geometric | 52 | 23.20% |
| RGBD | 50 | 21.20% |
| RGBD-Geometric | 54 | 27.45% |

## Project Structure

```
Pose6D/
├── data/                       # Dataset classes
│   ├── dataset_rgb.py          # RGB dataset
│   └── dataset_rgbd.py         # RGBD dataset
├── models/                     # Neural networks
│   ├── pose_net_rgb.py         # RGB-only network
│   ├── pose_net_rgb_geometric.py
│   ├── pose_net_rgbd.py        # RGBD fusion network
│   ├── pose_net_rgbd_geometric.py
│   ├── pose_loss.py            # Training loss (geodesic)
│   └── add_loss.py             # ADD/ADD-S evaluation
├── scripts/
│   ├── training/               # Training scripts
│   │   ├── train_yolo.py
│   │   ├── train_rgb.py
│   │   ├── train_rgb_geometric.py
│   │   ├── train_rgbd.py
│   │   └── train_rgbd_geometric.py
│   ├── inference/              # Inference scripts
│   │   ├── inference_rgb.py
│   │   ├── inference_rgb_geometric.py
│   │   ├── inference_rgbd.py
│   │   └── inference_rgbd_geometric.py
│   ├── visualization/          # Visualization & comparison
│   │   ├── compare_all_models.py
│   │   ├── compare_visual.py
│   │   └── visualize_yolo.py
│   └── setup/                  # Setup utilities
│       ├── setup_data.py
│       ├── setup_weights.py
│       ├── prepare_yolo.py
│       └── package_weights.py
├── utils/                      # Utility functions
│   ├── camera.py
│   ├── mesh_utils.py
│   └── visualization.py
├── weights_*/                  # Trained model checkpoints
├── datasets/                   # LineMOD dataset
├── colab_notebook.ipynb        # Google Colab demo
└── results_viewer.ipynb        # Results visualization
```

## Installation

```bash
# Clone repository
git clone https://github.com/SFR-Vision/6d-pose-estimation.git
cd 6d-pose-estimation

# Install dependencies
pip install -r requirements.txt
# OR
conda env create -f environment.yml
conda activate pose6d
```

## Usage

### 1. Setup Dataset

```bash
# Download LineMOD dataset
python scripts/setup/setup_data.py

# Prepare YOLO dataset
python scripts/setup/prepare_yolo.py
```

### 2. Train Models

```bash
# Train YOLO detector (5 epochs)
python scripts/training/train_yolo.py

# Train pose models (75 epochs each)
python scripts/training/train_rgb.py
python scripts/training/train_rgb_geometric.py
python scripts/training/train_rgbd.py
python scripts/training/train_rgbd_geometric.py
```

### 3. Evaluate & Compare

```bash
# Compare all 4 models
python scripts/visualization/compare_all_models.py

# Visual comparison
python scripts/visualization/compare_visual.py
```

### 4. Run Inference

```bash
python scripts/inference/inference_rgb.py
python scripts/inference/inference_rgbd_geometric.py
```

## Training Configuration

- **Epochs**: 75
- **Batch Size**: 32
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=5)
- **Loss**: Geodesic rotation + L1 translation
- **Augmentation**: ColorJitter, RandomErasing, bbox jitter

## Dataset

**LineMOD** contains 13 objects:
- Ape, Benchvise, Camera, Can, Cat, Driller, Duck
- Eggbox*, Glue*, Holepuncher, Iron, Lamp, Phone

*Symmetric objects (use ADD-S metric)

## License

MIT License

## Acknowledgments

- LineMOD dataset creators
- Ultralytics YOLOv8
- PyTorch team
