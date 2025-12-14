# Scripts Directory

This directory contains all executable scripts for the 6D Pose Estimation project.

## Structure

```
scripts/
├── setup/              # Dataset and environment setup scripts
│   ├── setup_data.py          # Download LineMOD dataset from Google Drive
│   ├── setup_weights.py       # Download pre-trained model weights
│   └── prepare_yolo.py        # Convert dataset to YOLO format
│
├── training/           # Model training scripts
│   ├── train_yolo.py          # Train YOLO object detector
│   ├── train_rgb.py           # Train RGB-only pose model
│   └── train_rgbd.py          # Train RGB-D pose model
│
├── inference/          # Inference and evaluation scripts
│   ├── inference_rgb.py       # Run inference with RGB model
│   └── inference_rgbd.py      # Run inference with RGB-D model
│
└── visualization/      # Result visualization scripts
    ├── visualize_yolo.py      # Visualize YOLO detection results
    ├── visualize_rgb.py       # Visualize RGB model results
    └── visualize_rgbd.py      # Visualize RGB-D model with GT comparison
```

## Usage

### Setup
```bash
# Download dataset
python scripts/setup/setup_data.py

# Prepare YOLO dataset
python scripts/setup/prepare_yolo.py

# [Optional] Download pre-trained weights
python scripts/setup/setup_weights.py
```

### Training
```bash
# Train YOLO detector
python scripts/training/train_yolo.py

# Train RGB pose model
python scripts/training/train_rgb.py

# Train RGB-D pose model
python scripts/training/train_rgbd.py
```

### Inference
```bash
# RGB inference
python scripts/inference/inference_rgb.py

# RGB-D inference
python scripts/inference/inference_rgbd.py
```

### Visualization
```bash
# Visualize YOLO results
python scripts/visualization/visualize_yolo.py

# Visualize RGB model
python scripts/visualization/visualize_rgb.py

# Visualize RGB-D with ground truth
python scripts/visualization/visualize_rgbd.py
```

## Notes

All scripts automatically set the project root path, so they can be run from anywhere within the project directory.
