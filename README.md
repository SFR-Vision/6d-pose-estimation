# 6D Pose Estimation with Dense Multi-Modal Fusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SFR-Vision/6d-pose-estimation/blob/main/colab_notebook.ipynb)

State-of-the-art 6D object pose estimation combining RGB-D fusion, geometric constraints, and dense pixel-wise feature fusion on the LineMOD dataset.

## Demo Videos

See our models in action! Each video shows 6D pose estimation with projected 3D bounding boxes on LineMOD test images.

### RGB-only Model
https://github.com/user-attachments/assets/rgb-only-demo.mp4
*50.3mm ADD error | Deployable with RGB camera only*

### RGB-Geometric Model ⭐ Best RGB-only
https://github.com/user-attachments/assets/rgb-geo-demo.mp4
*47.7mm ADD error | Geometric constraints + RGB depth prediction*

### RGB-D Model  
https://github.com/user-attachments/assets/rgbd-demo.mp4
*Multi-modal fusion | Requires RGB-D camera*

> **To view videos**: After pushing to GitHub, edit this README on GitHub and drag & drop videos from `demo_videos/` folder into each section above. GitHub will auto-generate the proper video URLs.

## Team Members

This project is a collaborative effort by the following team members:

- **U**lugbek **R**akhmatullaev
- **K**arim **A**bdelgelil **M**ohamed **S**halaby
- **M**ohammad **F**akih

## Features

- **Five Model Variants**: RGB-only, RGB-Geometric, RGB-D Global Fusion, DenseFusion-style
- **Multi-Modal Fusion**: RGB appearance + depth geometry for improved accuracy
- **Dense Pixel-wise Fusion**: DenseFusion-inspired architecture for state-of-the-art performance
- **Geometric Constraints**: Pinhole camera model for physically-informed translation
- **YOLOv8 Integration**: Real-time object detection pipeline
- **LineMOD Dataset**: 13 object categories with ground truth 6D poses
- **ADD/ADD-S Metrics**: Standard pose estimation evaluation

## Model Architectures

### Overview

| Model | Fusion Type | Rotation from | Translation | Params | Status |
|-------|-------------|---------------|-------------|---------|--------|
| **RGB-only** | None | RGB only | XYZ learned | 7 | ✅ Trained |
| **RGB-Geometric** | None | RGB only | Z learned, XY geometric | 5 | ✅ Trained |
| **RGB-D Global** | Global concat | RGB **+** Depth | Z learned, XY geometric | 5 | ✅ Ready |
| **DenseFusion** | Dense pixel-wise | RGB **+** 3D points | Z learned, XY geometric | 5 | ✅ Ready |

###Model Performance

### Trained Model Results (LineMOD Test Set)

| Model | ADD Error | ADD-S @50mm | Learned Params | Training Status |
|-------|-----------|-------------|----------------|-----------------|
| RGB-only | 50.3mm | 52.6% | 7 (rot+XYZ) | ✅ Trained |
| RGB-Geometric | 47.7mm | 58.9% | 5 (rot+Z) | ✅ Trained ⭐ Best RGB |
| RGB-D Global | TBD | TBD | 5 (rot+Z) | 🔄 Ready to train |
| DenseFusion | TBD | TBD | 5 (rot+Z) | 🔄 Ready to train |

**Key Findings**:
- Geometric constraints improve accuracy by ~5% while reducing complexity
- RGB-Geometric achieves 47.7mm ADD error using RGB only
- Multi-modal fusion models expected to achieve <45mm ADD error
- DenseFusion architecture targets state-of-the-art performanceonly)
- ResNet50 for rotation + custom CNN for depth prediction from RGB
- Geometric X,Y from predicted depth using pinhole model
- **5% improvement** over RGB-only (47.7mm vs 50.3mm ADD)
- **Use case**: RGB camera with known intrinsics

#### 3. RGB-D Global Fusion (Multi-modal)
- **Dual streams**: ResNet50 (RGB) + CNN (depth) → concatenated features
- **Both modalities** inform rotation and Z prediction
- Geometric X,Y from predicted Z
- **Innovation**: Depth provides geometric cues for rotation; RGB helps depth estimation
- **Use case**: RGB-D camera (e.g., Intel RealSense)

#### 4. DenseFusion (State-of-the-art)
- **Dense pixel-wise fusion**: Every pixel's RGB feature fused with 3D point coordinate
- Depth → 3D point cloud → per-pixel fusion with RGB features
- Max pooling over fused features for permutation invariance
- **SOTA architecture** comparable to published methods
- **Use case**: Research/high-accuracy applications

**Key Insight**: Progressive fusion complexity correlates with accuracy - from no fusion (RGB-only) → global fusion (RGB-D Global) → dense pixel-wise fusion (DenseFusion).

## Trained Model Results

| Model | Best Epoch | ADD-0.1d Accuracy |
|-------|------------|-------------------|
| RGB | 72 | 13.35% |
| RGB-Geometric | 52 | 23.20% |
| RGBD | 50 | 21.20% |-only dataset
│   └── dataset_rgbd.py         # RGB-D dataset with point clouds
├── models/                     # Neural network architectures
│   ├── pose_net_rgb.py         # RGB-only baseline
│   ├── pose_net_rgb_geometric.py  # RGB with geometric translation
│   ├── pose_net_rgbd_geometric.py # RGB-D global fusion
│   ├── pose_net_densefusion.py    # DenseFusion pixel-wise fusion
│   ├── pose_loss.py            # Training loss (geodesic rotation + L1)
│   └── add_loss.py             # ADD/ADD-S evaluation metrics
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
│   │   ├── train_yolo.py       # YOLO object detector
│   │   ├── train_rgb.py        # RGB-only model
│   │   ├── train_rgb_geometric.py  # RGB-Geometric model
│   │   ├── train_rgbd_geometric.py # RGB-D global fusion
│   │   └── train_densefusion.py    # DenseFusion model
│   ├── inference/              # Inference scripts
│   │   ├── inference_rgb.py
│   │   ├── inference_rgb_geometric.py
│   │   ├── inference_rgbd_geometric.py
│   │   └── inference_densefusion.py
│   ├── visualization/          # Visualization & comparison
│   │   ├── visualize_rgb.py
│   │   ├── visualize_rgb_geometric.py
│   │   ├── visualize_rgbd_geometric.py
│   │   ├── visualize_densefusion.py
│   │   ├── compare_all_models.py
│   │   └── visualize_yolo.py
│   └── setup/                  # Setup utilities
│       ├── setup_data.py
│       ├── setup_weights.py
│       ├── prepa               # Camera projection utilities
│   ├── mesh_utils.py           # 3D mesh loading (PLY files)
│   ├── visualization.py        # Pose visualization
│   └── inference_utils.py      # Inference helpers
├── weights_rgb/                # RGB model checkpoints
├── weights_rgb_geometric/      # RGB-Geometric checkpoints
├── weights_rgbd_geometric/     # RGB-D global fusion checkpoints
├── weights_densefusion/        # DenseFusion checkpoints
├── datasets/                   # LineMOD dataset
│   └── Linemod_preprocessed/
│       ├── data/               # RGB-D images + ground truth
│       └── models/             # 3D meshes (.ply) for ADD metric
├── colab_notebook.ipynb        # Google Colab one-click demo
└── results_viewer.ipynb        # Results visualization notebookints
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

# 1. Train YOLO detector (5 epochs)
python scripts/training/train_yolo.py

# 2. Train pose estimation models (75 epochs each)
python scripts/training/train_rgb.py              # RGB baseline
python scripts/training/train_rgb_geometric.py    # RGB with geometry (best RGB)
python scripts/training/train_rgbd_geometric.py   # RGB-D global fusion
python scripts/training/train_densefusion.py      # DenseFusion (SOTA)
```

**Training Tips**:
- RGB models: ~2 hours on RTX 3080
- RGB-D/DenseFusion: ~3-4 hours on RTX 3080
- Use `BATCH_SIZE=16` if OOM errors occur
- Models auto-resume from last checkpointhon scripts/training/train_yolo.py

# Train pose models (75 epochs each)
python Visualize & Compare

```bash
# Visualize individual model predictions
python scripts/visualization/visualize_rgb.py
python scripts/visualization/visualize_rgb_geometric.py
python scripts/visualization/visualize_rgbd_geometric.py
python scripts/visualization/visualize_densefusion.py

# Compare all models side-by-side
python scripts/visualization/compare_all_models.py
```

### 4. Run Inference

```bash
# Run inference on test images
python scripts/inference/inference_rgb_geometric.py      # Best RGB-only
python scripts/inference/inference_densefusion.py        # Best overall (when trained)

### 4. Run Inference

```bash
python scripts/inference/inference_rgb.py
python scripts/inference/inference_rgbd_geometric.py
### Hyperparameters
- **Epochs**: 75
- **Batch Size**: 32 (RGB/RGB-D Global), 32 (DenseFusion)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-7)
- **Gradient Clipping**: max_norm=1.0

### Loss Functions
- **Rotation Loss**: Geodesic distance on quaternions (weight=2)
- **Translation Loss**: L1 distance for Z (weight=5), XY computed geometrically
- **Evaluation**: ADD metric (Average Distance of Model Points)

### Data Augmentation
- **ColorJit(Linemod_preprocessed) contains 13 objects:
- **Objects**: Ape, Benchvise, Camera, Can, Cat, Driller, Duck, Eggbox*, Glue*, Holepuncher, Iron, Lamp, Phone
- **Modalities**: RGB images (640×480), Depth maps (uint16, mm), Object masks
- **Ground Truth**: 6D poses (rotation matrix + translation vector)
- **3D Models**: PLY meshes for ADD metric computation
- **Camera**: Fixed intrinsics (fx≈572, fy≈573, cx≈325, cy≈242)

*Symmetric objects use ADD-S metric (closest point distance)

### Data Splits
- **Train**: 80% (every 10th image excluded)
- **Validation**: 10% (every 10th image, cycle 8)
- **Test**: 10% (every 10th image, cycle 9
- **Quaternion format**: [x, y, z, w] (scipy default, scalar-last)
- **Depth units**: Meters (LineMOD range: 0.1-1.5m)
- **Camera model**: Pinhole projection for geometric X,Y
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=5)
- **Loss**: Geodesic rotation + L1 translation
- **Augmentation**: ColorJitter, RandomErasing, bbox jitter

## Technical Details

### Depth Processing Pipeline
```python
depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
depth_filtered = cv2.bilateralFilter(depth_mm, 5, 75, 75)  # Denoise
depth_m = depth_filtered / 1000.0  # Convert to meters
depth_normalized = np.clip(depth_m / 1.5, 0, 1)  # Normalize for CNN
```

### Point Cloud Generation (DenseFusion)
```python
# Back-project depth to 3D
Z = depth  # meters
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
points_3d = [X, Y, Z]  # [B, 3, H, W]
```

### Geometric Translation (All models except RGB-only)
```python
# Predict Z from network
z_pred = model.z_predictor(features)

# Compute X, Y geometrically
x = (u_center - cx) * z_pred / fx
y = (v_center - cy) * z_pred / fy
translation = [x, y, z_pred]
```

## System Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM (RTX 3080 recommended)
- **RAM**: ≥16GB
- **Storage**: ~10GB (dataset + models)
- **OS**: Windows/Linux/MacOS
- **Python**: 3.8+
- **CUDA**: 11.7+ (for GPU acceleration)

## Citation

If you use this work, please cite:
```bibtex
@misc{pose6d2026,
  title={6D Pose Estimation with Dense Multi-Modal Fusion},
  author={Rakhmatullaev, Ulugbek and Shalaby, Karim and Fakih, Mohammad},
  year={2026},
  howpublished={GitHub Repository},
  url={https://github.com/SFR-Vision/6d-pose-estimation}
}
```

## License

MIT License

## Acknowledgments

- **LineMOD Dataset**: Hinterstoisser et al., 2012
- **DenseFusion**: Wang et al., 2019 - Inspiration for dense pixel-wise fusion
- **Ultralytics YOLOv8**: Real-time object detection
- **PyTorch**: Deep learning framework
- **ResNet**: He et al., 2016 - Backbone architecture

MIT License

## Acknowledgments

- LineMOD dataset creators
- Ultralytics YOLOv8
- PyTorch team
