# 6D Pose Estimation with RGB-D Fusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SFR-Vision/6d-pose-estimation/blob/main/colab_notebook.ipynb)

Deep learning-based 6D object pose estimation on the LineMOD dataset, comparing RGB-only and RGB-D fusion approaches.

## Team Members

| Name | Contribution |
|------|--------------|
| **Ulugbek Rakhmatullaev** | RGB model, training pipeline, evaluation |
| **Karim Abdelgelil Mohamed Shalaby** | RGBD model, depth processing, fusion architecture |
| **Mohammad Fakih** | YOLO integration, visualization, documentation |

## Model Architectures

### RGB Model (4-channel input)
- **Input**: RGB (3) + Mask (1) = 4 channels
- **Backbone**: Modified ResNet50 (pretrained on ImageNet)
- **Outputs**: Rotation (quaternion) + Z-depth (learned), X,Y computed geometrically
- **Parameters**: 29.7M

### RGBD Model (5-channel input with z_sensor offset)
- **Input**: RGB (3) + Depth (1) + Mask (1) = 5 channels
- **Architecture**: Dual-stream fusion
  - RGB+Mask stream: ResNet50 → 2048 features
  - Depth+Mask stream: Custom CNN → 512 features
  - Fused features → Rotation head + Z-offset head
- **Z prediction**: `z_final = z_sensor + z_offset` (leverages depth sensor as prior)
- **Parameters**: 35.2M

## Results

| Model | ADD Error (mm) | ACC@2cm (%) | Improvement |
|-------|----------------|-------------|-------------|
| RGB | 31.4mm | 43.2% | Baseline |
| **RGBD** | **13.2mm** | **85.7%** | **58% better ADD, 2x accuracy** |

### Per-Object Results

| Object | RGB ADD | RGBD ADD | RGB ACC | RGBD ACC |
|--------|---------|----------|---------|----------|
| 01-ape | 29.0mm | 8.7mm | 44.7% | 98.4% |
| 02-benchvise | 33.0mm | 16.2mm | 28.1% | 69.4% |
| 04-camera | 31.6mm | 14.3mm | 31.7% | 80.8% |
| 05-can | 29.9mm | 11.4mm | 45.4% | 93.3% |
| 06-cat | 24.0mm | 10.5mm | 53.8% | **100%** |
| 08-driller | 32.4mm | 19.6mm | 28.8% | 66.9% |
| 09-duck | 26.5mm | 10.5mm | 85.6% | **100%** |
| 10-eggbox* | 29.8mm | 6.6mm | 78.4% | **100%** |
| 11-glue* | 32.9mm | 8.8mm | 41.8% | 99.2% |
| 12-holepuncher | 32.5mm | 11.3mm | 43.1% | 92.7% |
| 13-iron | 38.6mm | 20.8mm | 16.5% | 53.9% |
| 14-lamp | 34.8mm | 14.2mm | 29.5% | 86.1% |
| 15-phone | 31.7mm | 19.5mm | 32.0% | 71.3% |

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
# Train RGB model (75 epochs, ~2 hours on RTX 3080)
python scripts/training/train_rgb.py

# Train RGBD model (75 epochs, ~3 hours on RTX 3080)
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
L = L_rot/(2σ²_rot) + L_trans/(2σ²_trans) + log(σ_rot) + log(σ_trans)
```
- L_rot: Geodesic distance on quaternions
- L_trans: L1 loss on Z-depth
- σ parameters learned automatically

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
z_final = z_sensor + z_offset         # Model predicts small correction
```

## LineMOD Dataset

- **13 Objects**: ape, benchvise, camera, can, cat, driller, duck, eggbox*, glue*, holepuncher, iron, lamp, phone
- **Modalities**: RGB (640×480), Depth (uint16, mm), Masks
- **Camera**: fx≈572, fy≈573, cx≈325, cy≈242
- **Splits**: 80% train, 10% val, 10% test

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
