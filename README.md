# 6D Pose Estimation with RGB-D Fusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SFR-Vision/6d-pose-estimation/blob/main/colab_notebook.ipynb)

Deep learning-based 6D object pose estimation on the LineMOD dataset, comparing RGB-only and RGB-D fusion approaches.

## Team Members
 **Ulugbek Rakhmatullaev**,
 **Karim Abdelgelil Mohamed Shalaby**,
 **Mohammad Fakih** 

## Model Architectures

### RGB Model (4-channel input)
- **Input**: RGB (3) + Mask (1) = 4 channels
- **Backbone**: Modified ResNet50 (pretrained on ImageNet)
- **Outputs**: Rotation (quaternion) + Z-depth (learned), X,Y computed geometrically
- **Parameters**: 26.6M

### RGBD Model (5-channel input with z_sensor offset)
- **Input**: RGB (3) + Depth (1) + Mask (1) = 5 channels
- **Architecture**: Dual-stream fusion
  - RGB+Mask stream: ResNet50 → 2048 features
  - Depth+Mask stream: Custom CNN → 512 features
  - Fused features → Rotation head + Z-offset head
- **Z prediction**: `z_final = z_sensor + z_offset` (leverages depth sensor as prior)
- **Parameters**: 29.7M

## Results

| Model | ADD Error (mm) | ACC@2cm (%) | Improvement |
|-------|----------------|-------------|-------------|
| RGB | 31.5mm | 41.2% | Baseline |
| **RGBD** | **13.5mm** | **84.9%** | **57% better ADD, 2x accuracy** |

### Per-Object Results

| Object | RGB ADD | RGBD ADD | RGB ACC | RGBD ACC |
|--------|---------|----------|---------|----------|
| 01-ape | 31.9mm | 8.6mm | 38.2% | 98.4% |
| 02-benchvise | 30.3mm | 16.2mm | 29.8% | 68.6% |
| 04-camera | 29.7mm | 14.4mm | 33.3% | 79.2% |
| 05-can | 28.4mm | 12.9mm | 41.2% | 89.9% |
| 06-cat | 24.0mm | 10.5mm | 48.7% | **100%** |
| 08-driller | 34.9mm | 19.3mm | 29.7% | 68.6% |
| 09-duck | 26.5mm | 10.3mm | 84.0% | **100%** |
| 10-eggbox* | 31.9mm | 6.6mm | 74.4% | **100%** |
| 11-glue* | 35.5mm | 8.3mm | 41.8% | **100%** |
| 12-holepuncher | 30.2mm | 10.8mm | 35.8% | 94.3% |
| 13-iron | 38.1mm | 21.4mm | 17.4% | 48.7% |
| 14-lamp | 34.8mm | 14.6mm | 32.8% | 86.1% |
| 15-phone | 33.0mm | 21.5mm | 28.7% | 70.5% |

*Symmetric objects use ADD-S metric

## Project Structure

```
Pose6D/
├── data/                           # Dataset classes
│   ├── dataset_rgb.py              # RGB+Mask dataset (4ch)
│   └── dataset_rgbd.py             # RGB+Depth+Mask dataset (5ch)
├── models/                         # Neural network architectures
│   ├── pose_net_rgb.py             # RGB model
│   ├── pose_net_rgbd.py            # RGBD dual-stream model
│   ├── pose_loss.py                # Training loss (AutoWeighted/Geodesic)
│   └── add_loss.py                 # ADD evaluation metric
├── scripts/
│   ├── training/                   # Training scripts
│   │   ├── train_rgb.py            # Train RGB model
│   │   ├── train_rgbd.py           # Train RGBD model
│   │   └── train_yolo_seg.py       # Train YOLO segmentation
│   ├── inference/                  # Inference scripts
│   │   ├── inference_rgb.py        # RGB inference with visualization
│   │   └── inference_rgbd.py       # RGBD inference with visualization
│   ├── visualization/              # Visualization & analysis
│   │   ├── plot_training.py        # Generate training curves
│   │   ├── per_object_metrics.py   # Per-object ADD metrics
│   │   └── compare_all_models.py   # Side-by-side comparison
│   └── setup/                      # Setup utilities
│       ├── setup_data.py           # Download LineMOD dataset
│       └── setup_weights.py        # Download pretrained weights
├── utils/                          # Utility modules
│   ├── camera.py                   # Camera intrinsics
│   ├── mesh_utils.py               # 3D model loading
│   ├── visualization.py            # Drawing functions
│   └── inference_utils.py          # Inference helpers
├── weights_rgb/                    # RGB model weights & history
├── weights_rgbd/                   # RGBD model weights & history
├── datasets/Linemod_preprocessed/  # LineMOD dataset
├── report/                         # CVPR-style LaTeX report
├── colab_notebook.ipynb            # Google Colab demo
└── results_viewer.ipynb            # Local results visualization
```

## Installation

```bash
# Clone repository
git clone https://github.com/SFR-Vision/6d-pose-estimation.git
cd 6d-pose-estimation

# Create conda environment
conda env create -f environment.yml
conda activate pose6d

# OR use pip
pip install -r requirements.txt
```

## Usage

### 1. Setup Dataset & Weights
```bash
python scripts/setup/setup_data.py      # Download LineMOD dataset
python scripts/setup/setup_weights.py   # Download pretrained weights
```

### 2. Train Models
```bash
# Train RGB model (75 epochs, ~1.5 hours)
python scripts/training/train_rgb.py

# Train RGBD model (75 epochs, ~1.5 hours)
python scripts/training/train_rgbd.py

# Train YOLO segmentation (for object detection)
python scripts/training/train_yolo_seg.py
```

### 3. Run Inference
```bash
python scripts/inference/inference_rgb.py    # RGB model
python scripts/inference/inference_rgbd.py   # RGBD model
```

### 4. Visualize Results
```bash
# Generate training curves
python scripts/visualization/plot_training.py --all

# Per-object metrics
python scripts/visualization/per_object_metrics.py
```

## Technical Details

### Loss Function
**AutoWeightedPoseLoss** (Homoscedastic Uncertainty Weighting):
```
L = (1/2)e^(-s_r) * L_rot + (1/2)e^(-s_t) * L_trans + (1/2)(s_r + s_t)
```
- **L_rot**: Geodesic distance on quaternions (atan2 formulation)
- **L_trans**: L1 loss on Z-depth only
- **s_r, s_t**: Learned log-variance parameters (initialized to 0)
- **Note**: Total loss can become negative as training progresses (expected behavior)

### Geometric Translation (Pinhole Model)
```python
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```
Where (u,v) is bbox center, Z is predicted depth, and (fx,fy,cx,cy) are camera intrinsics.

### RGBD Z-Offset Prediction
Instead of predicting absolute depth, the RGBD model predicts an offset from the sensor measurement:
```python
z_sensor = median(depth[mask > 0.5])  # Sample from masked object pixels
z_final = z_sensor + z_offset         # Model predicts small correction (±10-30mm)
```

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 48 |
| Epochs | 75 |
| LR schedule | Warmup (5 epochs) + Cosine annealing |
| Gradient clipping | max_norm = 1.0 |

### Ablation Studies
- **ReduceLROnPlateau**: Underperformed vs Cosine (38.1% vs 41.2%)
- **Data augmentation**: Decreased accuracy (color jitter, bbox jitter broke geometric constraints)
- **Depth range**: Optimized to [0.3, 1.7]m based on GT analysis

## LineMOD Dataset

- **13 Objects**: ape, benchvise, camera, can, cat, driller, duck, eggbox*, glue*, holepuncher, iron, lamp, phone
- **Modalities**: RGB (640×480), Depth (uint16, mm), Masks
- **Camera**: fx≈572.4, fy≈573.6, cx≈325.3, cy≈242.0
- **Splits**: 80% train (12,637), 10% val (1,573), 10% test (1,573)

*Symmetric objects use ADD-S metric

## System Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM
- **RAM**: ≥16GB
- **Storage**: ~10GB (dataset + models)
- **Python**: 3.8+
- **CUDA**: 11.7+

## Citation

```bibtex
@misc{pose6d2026,
  title={6D Pose Estimation with RGB-D Fusion},
  author={Rakhmatullaev, Ulugbek and Shalaby, Karim and Fakih, Mohammad},
  year={2026},
  howpublished={GitHub Repository},
  url={https://github.com/SFR-Vision/6d-pose-estimation}
}
```

## Acknowledgments

- LineMOD Dataset: Hinterstoisser et al., 2012
- DenseFusion: Wang et al., 2019
- Ultralytics YOLOv8
- PyTorch

## License

MIT License
