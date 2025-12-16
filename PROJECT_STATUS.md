# Project Status - 6D Pose Estimation (LineMOD Dataset)

**Last Updated**: December 16, 2025  
**Current Phase**: Production Ready - Colab Deployment Complete

---

## 🎯 Project Goal

Estimate 6D pose (3D rotation + 3D translation) of objects from RGB-D images using the LineMOD dataset (13 objects).

**Performance Metric**: ADD (Average Distance of Model Points) - lower is better  
**Target**: < 5cm ADD error

---

## 📊 Final Results

| Model | Architecture | ADD Error (Test) | ADD-S Accuracy @50mm | Status |
|-------|-------------|------------------|---------------------|--------|
| **RGB-only** | ResNet50 | **50.3mm** | 52.6% | ✅ Trained (10 epochs) |
| **Hybrid** ⭐ | ResNet50 + Custom CNN + Pinhole | **47.7mm** | 58.9% | ✅ **Trained (Best)** |
| RGB-D | ResNet50 + ResNet50 | Not trained | - | ⏸️ Pending retraining |

**Key Finding**: Hybrid model achieves **5.2% better accuracy** than RGB-only by incorporating camera geometry!

---

## 🚀 Current Status - PRODUCTION READY ✅

**Completed**:
- ✅ RGB model trained (10 epochs) - 50.3mm test ADD error
- ✅ Hybrid model trained (full training) - 47.7mm test ADD error
- ✅ Comparison analysis with detailed metrics (ADD-S accuracy, distribution stats)
- ✅ Colab notebook with "Run All" automation
- ✅ Pre-trained weights uploaded to Google Drive
- ✅ Comprehensive visualizations (3D bounding boxes, side-by-side comparisons)
- ✅ `.github/copilot-instructions.md` for AI agent onboarding

**Ready for Deployment**:
- 🎯 Teammates can run `colab_setup.ipynb` → "Run All" → Get results in 20-30 minutes
- 📦 Pre-trained weights download automatically (RGB + Hybrid + YOLO)
- 📊 All visualizations generate automatically
- 💾 Results save to Google Drive automatically

**Future Work**:
- 🔄 Retrain RGB-D model with fixed augmentation (currently deprioritized)
- 🎓 Explore deeper architectures (ResNet101, EfficientNet)
- 🌍 Test on other datasets (YCB-Video, T-LESS)

---

## 🏗️ Architecture Overview

### Model 1: RGB-only (`pose_net_rgb.py`)
- **Input**: RGB image (224×224)
- **Backbone**: ResNet50 (pretrained)
- **Heads**: Rotation (4 outputs - quaternion) + Translation (3 outputs - XYZ)
- **Parameters**: ~25M
- **Files**: `dataset_rgb.py`, `train_rgb.py`, `inference_rgb.py`, `visualize_rgb.py`

### Model 2: RGB-D (`pose_net_rgbd.py`)
- **Input**: RGB (224×224) + Depth (224×224)
- **Backbone**: ResNet50 RGB + ResNet50 Depth (upgraded from ResNet18)
- **Fusion**: Concatenate features → 2048+2048 → 2048
- **Heads**: Rotation (4) + Translation (3)
- **Parameters**: ~50M
- **Files**: `dataset_rgbd.py`, `train_rgbd.py`, `inference_rgbd.py`, `visualize_rgbd.py`

### Model 3: Hybrid (`pose_net_hybrid.py`) ⭐ **BEST PERFORMER**
- **Input**: RGB (224×224) + Bbox Center + Camera Matrix (Note: Depth NOT used)
- **Rotation Branch**: ResNet50 (pretrained) → 2048 → 1024 → 512 → 4 (quaternion) - **LEARNED**
- **Z-Depth Branch**: Custom CNN (from RGB, not depth!)
  - 224×224×3 → Conv7+Pool → Conv5+Pool → Conv3+Pool → Conv3+Pool → GlobalAvgPool → 256 features
  - 256 → 128 → 64 → 1 (Z distance in meters) - **LEARNED**
- **X,Y Translation**: Pinhole camera model - **GEOMETRIC** (not learned)
  - `X = (u - cx) * Z / fx`
  - `Y = (v - cy) * Z / fy`
- **Parameters**: ~30M (5 learned outputs vs 7 for RGB-only)
- **Performance**: 47.7mm ADD error (5% better than RGB-only)
- **Files**: `dataset_hybrid.py`, `train_hybrid.py`, `compare_rgb_vs_hybrid.py`, `visualize_hybrid.py`

**Why Hybrid?**: Tests hypothesis that incorporating domain knowledge (camera geometry) improves over pure learning.

---

## 🐛 Critical Bug Fixed

**MAJOR BUG DISCOVERED & FIXED**:
- **Issue**: Original augmentation added rotation/translation **noise to labels WITHOUT modifying images**
- **Impact**: Model trained on corrupted labels → worse performance
- **Fix**: Removed rotation (±5°) and translation (±20mm) noise, kept only bbox jitter
- **Files affected**: `dataset_rgb.py`, `dataset_rgbd.py`, `dataset_hybrid.py`
- **Parameter renamed**: `augment_pose` → `augment_bbox` (more accurate name)

**Result**: All models need retraining with fixed augmentation after hybrid experiments.

---

## 🔧 Windows-Specific Fixes Applied

### Fix 1: Multiprocessing Bootstrap Error
- **Problem**: DataLoader workers re-executed main script on spawn
- **Solution**: Wrapped training loop in `if __name__ == '__main__':`
- **Location**: All `train_*.py` files

### Fix 2: OpenMP Library Conflict
- **Problem**: `OMP Error #15` - libomp.dll vs libiomp5md.dll conflict
- **Solution**: Added `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at top
- **Location**: All `train_*.py` files

### Fix 3: Camera Matrix Tensor Batching
- **Problem**: DataLoader didn't batch 3×3 camera matrices correctly
- **Solution**: Added dimension check in `pose_net_hybrid.py` to expand [3,3] → [B,3,3]
- **Location**: `models/pose_net_hybrid.py` line ~132

---

## 📁 Project Structure

```
Pose6D/
├── PROJECT_STATUS.md           ← YOU ARE HERE
├── README.md                   ← User-facing documentation
├── colab_setup.ipynb           ← Google Colab deployment (27 cells)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── dataset_rgb.py          ← RGB-only (5 outputs)
│   ├── dataset_rgbd.py         ← RGB-D (5 outputs)
│   └── dataset_hybrid.py       ← RGB-D + camera info (7 outputs)
│
├── models/
│   ├── pose_net_rgb.py         ← ResNet50
│   ├── pose_net_rgbd.py        ← ResNet50 + ResNet50
│   ├── pose_net_hybrid.py      ← ResNet50 + Custom CNN + Pinhole
│   └── loss.py                 ← ADD loss (with optional rot/trans weights)
│
├── scripts/
│   ├── setup/
│   │   ├── setup_data.py       ← Download LineMOD dataset
│   │   ├── setup_weights.py    ← Download pretrained weights
│   │   └── prepare_yolo.py     ← Convert to YOLO format
│   ├── training/
│   │   ├── train_yolo.py
│   │   ├── train_rgb.py
│   │   ├── train_rgbd.py
│   │   └── train_hybrid.py     ← CURRENTLY RUNNING
│   ├── inference/
│   │   ├── inference_rgb.py
│   │   └── inference_rgbd.py
│   └── visualization/
│       ├── visualize_yolo.py
│       ├── visualize_rgb.py
│       ├── visualize_rgbd.py
│       ├── compare_rgb_vs_rgbd.py
│       └── compare_rgb_vs_hybrid.py
│
├── weights_rgb/                ← RGB model checkpoints
├── weights_rgbd/               ← RGB-D model checkpoints
├── weights_hybrid/             ← Hybrid model checkpoints (TRAINING)
│   ├── best_pose_model.pth     ← Best validation loss
│   └── last_pose_model.pth     ← Latest epoch (for resuming)
│
└── datasets/
    └── Linemod_preprocessed/
        ├── data/               ← 13 object folders (01-15, no 03,07)
        │   ├── 01/
        │   │   ├── gt.yml      ← Ground truth poses
        │   │   ├── info.yml    ← Camera intrinsics (fx, fy, cx, cy)
        │   │   ├── train.txt
        │   │   ├── test.txt
        │   │   ├── rgb/
        │   │   ├── depth/
        │   │   └── mask/
        │   └── ... (02, 04-06, 08-15)
        └── models/
            ├── models_info.yml ← 3D model info (diameter, etc.)
            └── obj_*.ply       ← 3D point clouds for ADD metric
```

---

## 🔄 Data Pipeline

1. **YOLO Detection** → Bounding boxes around objects
2. **Crop RGB + Depth** → Extract 224×224 patches
3. **Load Ground Truth** → Quaternion (4D) + Translation (3D) from `gt.yml`
4. **Load Camera Info** → fx, fy, cx, cy from `info.yml` (hybrid model only)
5. **Augmentation** → ColorJitter + Bbox jitter (NO rotation/translation noise)
6. **Depth Processing**:
   - Bilateral filter: `cv2.bilateralFilter(depth_mm, 5, 75, 75)`
   - Convert to meters: `/ 1000.0`
   - Clip and normalize: `/ 1.5` (LineMOD max depth ~1.5m)

---

## 📋 TODO List (Priority Order)

### 🔥 Immediate (Training Campaign)
1. **Option A - Resume Hybrid**: `python scripts/training/train_hybrid.py` (will auto-resume from epoch 2)
2. **Option B - Fresh Start**: Systematic retraining RGB → RGB-D → Hybrid
3. After any training completes: Run comparison scripts to evaluate improvements

### 🎯 High Priority (Complete Retraining)
4. Train RGB model with fixed augmentation
   - Command: `python scripts/training/train_rgb.py`
   - Expected: ~5-5.5cm (down from 6.5cm buggy version)
5. Train RGB-D model with fixed augmentation
   - Command: `python scripts/training/train_rgbd.py`
   - Expected: ~3-3.5cm (down from 4.2cm buggy version)
6. Complete Hybrid training (100 epochs)
   - Command: `python scripts/training/train_hybrid.py`
   - Expected: ~3cm (best of all approaches)
7. Run full comparison: `python scripts/visualization/compare_rgb_vs_rgbd.py` and `compare_rgb_vs_hybrid.py`

### 📦 Medium Priority (Deployment)
8. Package new pre-trained weights
9. Upload to Google Drive
10. Update Colab notebook links
11. Update README.md with final results

### 🚀 Low Priority (Optimization)
12. Model optimization experiments:
    - Mixed precision training (2x faster)
    - Differential learning rates (faster/slower branches)
    - Switch ResNet50 → ResNet34 (30% smaller)
13. Model quantization (INT8) for deployment
14. ONNX export for cross-platform inference

---

## 🧪 Experiments Tried & Abandoned

### Geometric Pose Estimation (ABANDONED)
- **Idea**: Use depth + 2D-3D correspondence for pose via PnP
- **Result**: 15cm translation error (worse than 6.5cm learned)
- **Conclusion**: End-to-end learning superior for this task
- **Files deleted**: `geometric_pose.py`, etc.

### Depth Processing Fixes (APPLIED)
- **Issue 1**: Bilateral filter sigma too low (5) → changed to 75 (mm-scale values)
- **Issue 2**: Depth normalization range wrong (3m) → changed to 1.5m (LineMOD-specific)
- **Result**: Improved depth features

---

## 💡 Key Insights

1. **Hybrid Model Fast Convergence**: 4.3cm by epoch 2 because X,Y are geometrically computed from Z
2. **Domain Knowledge Helps**: Incorporating pinhole camera model reduces learning complexity
3. **Augmentation is Critical**: Wrong augmentation destroyed performance - fixed now
4. **Depth Matters**: RGB-D (4.2cm) beats RGB (6.5cm) by 35%
5. **Pure ADD Loss Works Best**: Separate rotation/translation weights (rot_weight=0.0, trans_weight=0.0) performed worse
6. **AI Agent Documentation**: Created `.github/copilot-instructions.md` for systematic knowledge transfer - covers architecture, conventions, Windows fixes, and development patterns

---

## 🛠️ How to Resume/Continue

### If Training Interrupted
```bash
cd "D:\MSc\Year2Semester1\Data Analysis and Artificial Intellegence\Projects\Pose6D"
python scripts/training/train_hybrid.py
```
- Will automatically resume from `weights_hybrid/last_pose_model.pth`

### If Starting Fresh Agent
1. Read this file (PROJECT_STATUS.md)
2. Check `weights_hybrid/` for checkpoints
3. Look at terminal output for last epoch number
4. Review `colab_setup.ipynb` for deployment details
5. Check `scripts/training/train_hybrid.py` for current hyperparameters

### To Visualize Results
```bash
# After training completes
python scripts/visualization/compare_rgb_vs_hybrid.py
```

---

## 📊 Dataset Info

- **Name**: LineMOD (Hinterstoisser et al.)
- **Objects**: 13 household objects (ape, benchvise, cam, can, cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)
- **Images**: ~1200 per object
- **Resolution**: RGB 640×480, Depth 640×480
- **Split**: ~80% train, ~20% val
- **Camera**: Fixed intrinsics (fx, fy, cx, cy in info.yml)
- **Pose Format**: Quaternion (wxyz) + Translation (xyz in meters)

---

## 🔗 Important References

- **Google Drive**: Pre-trained weights hosted (see colab_setup.ipynb)
- **Colab Badge**: "Open in Colab" button in README.md
- **ADD Metric**: Average Distance of 3D model points after pose transformation
- **LineMOD Paper**: Hinterstoisser et al., "Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes"

---

## 🚨 Known Issues & Solutions

| Issue | Solution | Status |
|-------|----------|--------|
| Windows multiprocessing error | Add `if __name__ == '__main__':` wrapper | ✅ Fixed |
| OpenMP library conflict | Set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` | ✅ Fixed |
| Camera matrix batching | Add dimension check in forward() | ✅ Fixed |
| Buggy augmentation | Remove rotation/translation noise | ✅ Fixed (need retrain) |
| Depth processing wrong | Bilateral sigma=75, max_depth=1.5m | ✅ Fixed |

---

## 📝 Notes for Agent

- **Current Priority**: Wait for hybrid training to complete, then compare approaches
- **Don't Retrain Yet**: RGB/RGB-D models need retraining with fixed augmentation, but wait until hybrid results are analyzed
- **Checkpoint Path**: `weights_hybrid/last_pose_model.pth` has epoch, model, optimizer, best_val_loss
- **Training Time**: ~100 epochs × 2 min/epoch = ~3-4 hours remaining
- **Next Script**: `compare_rgb_vs_hybrid.py` after training completes
- **User's Main Question**: Does hybrid (learned + geometric) beat pure learning (RGB-D)?

---

**END OF STATUS DOCUMENT**
