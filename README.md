# 6D Pose Estimation with RGB-D Fusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SFR-Vision/6d-pose-estimation/blob/main/colab_setup.ipynb)

Real-time 6D object pose estimation using YOLO detection and RGB-D deep learning on the LineMOD dataset.

## Features

- **RGB-D Fusion**: Combines RGB and depth information for improved accuracy
- **YOLO Detection**: Fast object detection for real-time performance
- **Pinhole Camera Model**: Geometric correction for accurate 3D pose estimation
- **LineMOD Dataset**: 13 object categories with ground truth poses

## Architecture

```
YOLO (Object Detection) → RGB-D PoseNet (Pose Estimation) → 6D Pose (Rotation + Translation)
```

- **RGB Backbone**: ResNet50 (pretrained on ImageNet)
- **Depth Backbone**: ResNet18 (single-channel input)
- **Fusion**: Concatenation + MLP
- **Output**: Quaternion (4D) + Translation (3D)

## Setup

### Local Installation

```bash
# Clone repository
git clone https://github.com/SFR-Vision/6d-pose-estimation.git
cd 6d-pose-estimation

# Install dependencies
pip install -r requirements.txt

# Download LineMOD dataset
# Place in: datasets/Linemod_preprocessed/
```

### Google Colab (Recommended)

**Option 1: Automated Pipeline (Easiest)**

1. Open `colab_setup.ipynb` in Google Colab
2. Click **Runtime → Run All**
3. Done! Everything runs automatically (dataset download, training, visualization)

**Option 2: Use Pre-trained Weights (Fastest)**

1. Open `colab_setup.ipynb` in Google Colab
2. Set `USE_PRETRAINED = True` in Step 3.5
3. Click **Runtime → Run All**
4. Skips ~3-4 hours of training, uses pre-trained models

**Option 3: Manual Setup**

```python
# Mount Google Drive (store dataset and models here)
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/SFR-Vision/6d-pose-estimation.git
%cd 6d-pose-estimation

# Install dependencies
!pip install -r requirements.txt

# Download dataset (example - adjust to your storage)
# Option 1: From Google Drive
!cp -r "/content/drive/MyDrive/LineMOD/datasets" .
!cp -r "/content/drive/MyDrive/LineMOD/weights_rgbd" .

# Option 2: Download from source
# !wget <dataset_url>
# !unzip linemod.zip -d datasets/
```

## Usage

### 1. Prepare YOLO Dataset

```bash
python data/prepare_yolo.py
```

### 2. Train YOLO (Object Detection)

```bash
python train_yolo.py
```

### 3. Train RGB-D Pose Model

```bash
python train_rgbd.py
```

### 4. Visualize Results

```bash
python visualize_rgbd.py
```

### 5. Run Inference

```bash
# Single image
python inference_rgbd.py path/to/image.png path/to/depth.png

# Auto mode (random test image)
python inference_rgbd.py
```

## Project Structure

```
Pose6D/
├── data/
│   ├── dataset.py          # RGB dataset loader
│   ├── dataset_rgbd.py     # RGB-D dataset loader
│   └── prepare_yolo.py     # YOLO data preparation
├── models/
│   ├── pose_net.py         # RGB-only pose network
│   ├── pose_net_rgbd.py    # RGB-D pose network
│   └── loss.py             # ADD loss function
├── train_rgbd.py           # Training script
├── inference_rgbd.py       # Inference script
├── visualize_rgbd.py       # Visualization script
└── requirements.txt        # Dependencies
```

## Results

- **RGB-D Model**: ~4.2 cm ADD error
- **RGB-Only Model**: ~6.5 cm ADD error
- **Improvement**: ~35% reduction in pose error with depth

## Dataset

**LineMOD** contains 13 objects with:
- RGB images (640×480)
- Depth maps (640×480)
- Ground truth 6D poses
- 3D object models (.ply)

Objects: Ape, Benchvise, Camera, Can, Cat, Driller, Duck, Eggbox, Glue, Holepuncher, Iron, Lamp, Phone

## Training Details

- **Epochs**: 100
- **Batch Size**: 16
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Loss**: ADD (Average Distance of Model Points)
- **Augmentation**: ColorJitter, RandomGrayscale, RandomErasing, geometric noise

## Citation

If you use this code, please cite:

```bibtex
@misc{pose6d2025,
  author = {SFR-Vision},
  title = {6D Pose Estimation with RGB-D Fusion},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SFR-Vision/6d-pose-estimation}
}
```

## License

MIT License

## Acknowledgments

- LineMOD dataset creators
- Ultralytics YOLOv8
- PyTorch team
