# 6D Pose Estimation with Geometric Constraints

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SFR-Vision/6d-pose-estimation/blob/main/colab_setup.ipynb)

Real-time 6D object pose estimation using YOLO detection and deep learning with geometric constraints on the LineMOD dataset.

## Features

- **Hybrid Architecture**: RGB features + Geometric camera constraints (5% better than RGB-only)
- **Three Model Variants**: RGB-only, RGB-D fusion, and Hybrid (geometry-aware)
- **YOLO Detection**: Fast object detection for real-time performance
- **Pinhole Camera Model**: Geometric X,Y computation reduces learned parameters
- **LineMOD Dataset**: 13 object categories with ground truth 6D poses
- **Pre-trained Weights**: Ready-to-use models via Google Colab

## Architecture

### Three Model Variants

**1. RGB-only**: Pure learning approach
```
RGB → ResNet50 → [Rotation Head, Translation Head] → 6D Pose
Learned: Rotation (4 params) + XYZ (3 params) = 7 parameters
```

**2. RGB-D**: Dual backbone fusion
```
RGB → ResNet50 ─┐
                 ├→ Concat → MLP → [Rotation, Translation]
Depth → ResNet50 ─┘
Learned: Rotation (4) + XYZ (3) = 7 parameters
```

**3. Hybrid** ⭐ **(Best - 5% improvement)**
```
RGB → ResNet50 → Rotation (quaternion) - LEARNED
RGB → Custom CNN → Z-distance - LEARNED
(Bbox, Camera, Z) → Pinhole Model → X,Y - GEOMETRIC (not learned)
Learned: Rotation (4) + Z (1) = 5 parameters
```

**Key Insight**: Hybrid achieves better accuracy with fewer learned parameters by incorporating camera geometry as inductive bias!

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

### Google Colab (Recommended) ⚡

**Automated "Run All" Pipeline** (Easiest - 20-30 minutes)

1. Open [`colab_setup.ipynb`](colab_setup.ipynb) in Google Colab
2. Click **Runtime → Run All**
3. Done! Everything runs automatically:
   - ✅ Downloads LineMOD dataset (~2 GB)
   - ✅ Downloads pre-trained RGB & Hybrid models (~250 MB)
   - ✅ Evaluates both models on test set
   - ✅ Generates comparison visualizations
   - ✅ Saves results to your Google Drive

**No configuration needed!** Default mode uses pre-trained weights.

**Training from Scratch** (Optional - 6-8 hours GPU)

1. In the notebook, set `USE_PRETRAINED = False`
2. Click **Runtime → Run All**
3. Models train automatically: YOLO → RGB → Hybrid

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

### 2. Train Models

```bash
# Train YOLO detector
python scripts/training/train_yolo.py

# Train RGB-only model
python scripts/training/train_rgb.py

# Train Hybrid model (RGB + Geometric constraints)
python scripts/training/train_hybrid.py

# [Optional] Train RGB-D model
python scripts/training/train_rgbd.py
```

### 3. Compare Models

```bash
# RGB vs Hybrid comparison with detailed metrics
python scripts/visualization/compare_rgb_vs_hybrid.py

# RGB vs RGB-D comparison
python scripts/visualization/compare_rgb_vs_rgbd.py
```

### 4. Visualize Individual Results

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

## Results

### Model Performance (LineMOD Test Set)

| Model | ADD Error | ADD-S @50mm | Parameters | Improvement |
|-------|-----------|-------------|------------|-------------|
| RGB-only | 50.3mm | 52.6% | 7 learned | Baseline |
| **Hybrid** ⭐ | **47.7mm** | **58.9%** | 5 learned | **+5.2%** |
| RGB-D | TBD | TBD | 7 learned | Pending |

**Key Finding**: Hybrid model achieves better accuracy with fewer learned parameters by incorporating camera geometry!

### ADD-S Accuracy Breakdown

Percentage of predictions below error threshold:

| Threshold | RGB-only | Hybrid | Improvement |
|-----------|----------|--------|-------------|
| < 20mm | 15.2% | 18.3% | +3.1% |
| < 30mm | 28.4% | 32.1% | +3.7% |
| < 50mm | 52.6% | 58.9% | +6.3% |
| < 100mm | 81.3% | 85.7% | +4.4% |

### Visualizations

![RGB vs Hybrid Comparison](comparison_results/example.jpg)

*Side-by-side comparison: Green (Ground Truth), Yellow (RGB), Magenta (Hybrid)*

## Project Structure

```
Pose6D/
├── colab_setup.ipynb       # Google Colab deployment (Run All)
├── data/
│   ├── dataset_rgb.py      # RGB-only dataset
│   ├── dataset_rgbd.py     # RGB-D dataset  
│   └── dataset_hybrid.py   # Hybrid dataset (RGB + camera info)
├── models/
│   ├── pose_net_rgb.py     # RGB-only network
│   ├── pose_net_rgbd.py    # RGB-D fusion network
│   ├── pose_net_hybrid.py  # Hybrid network (geometry-aware)
│   └── loss.py             # ADD loss function
├── scripts/
│   ├── setup/              # Dataset & weights download
│   ├── training/           # Model training scripts
│   ├── inference/          # Inference scripts
│   └── visualization/      # Comparison & visualization
├── weights_rgb/            # RGB model checkpoints
├── weights_hybrid/         # Hybrid model checkpoints
└── requirements.txt        # Dependencies
```

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
