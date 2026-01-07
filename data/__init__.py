from .dataset_rgb import LineMODDatasetRGB
from .dataset_rgbd import LineMODDatasetRGBD

"""
Data module for 6D pose estimation.

This module provides dataset classes for loading LineMOD data with different input modalities:
- LineMODDatasetRGB: RGB-only input
- LineMODDatasetRGBD: RGB-D input
"""


__all__ = [
    'LineMODDatasetRGB',
    'LineMODDatasetRGBD',
]