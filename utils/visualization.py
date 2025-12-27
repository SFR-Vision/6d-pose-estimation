"""Utility functions for 3D to 2D projection and visualization."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def project_points(points_3d, rotation, translation, K):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: numpy array of shape (N, 3)
        rotation: quaternion (4,) or rotation matrix (3, 3)
        translation: numpy array of shape (3,)
        K: camera intrinsic matrix (3, 3)
    
    Returns:
        numpy array of shape (N, 2) with 2D pixel coordinates
    """
    if rotation.shape == (4,):
        r_mat = R.from_quat(rotation).as_matrix()
    else:
        r_mat = rotation
    
    p_cam = (r_mat @ points_3d.T).T + translation
    z = np.clip(p_cam[:, 2], 0.001, None)
    
    pts_2d = np.zeros((points_3d.shape[0], 2))
    pts_2d[:, 0] = (p_cam[:, 0] * K[0, 0] / z) + K[0, 2]
    pts_2d[:, 1] = (p_cam[:, 1] * K[1, 1] / z) + K[1, 2]
    return pts_2d.astype(int)


def draw_3d_box(img, pts_2d, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box edges on image.
    
    Args:
        img: OpenCV image (will be modified in-place)
        pts_2d: numpy array of shape (8, 2) with corner coordinates
        color: BGR color tuple
        thickness: line thickness
    """
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for s, e in edges:
        pt1 = (int(pts_2d[s][0]), int(pts_2d[s][1]))
        pt2 = (int(pts_2d[e][0]), int(pts_2d[e][1]))
        cv2.line(img, pt1, pt2, color, thickness)


def draw_axes(img, rotation, translation, K, scale=0.1):
    """
    Draw XYZ axes at object center.
    
    Args:
        img: OpenCV image (will be modified in-place)
        rotation: quaternion or rotation matrix
        translation: translation vector
        K: camera intrinsic matrix
        scale: length of axes in meters
    """
    origin = project_points(np.array([[0, 0, 0]]), rotation, translation, K)[0]
    x_axis = project_points(np.array([[scale, 0, 0]]), rotation, translation, K)[0]
    y_axis = project_points(np.array([[0, scale, 0]]), rotation, translation, K)[0]
    z_axis = project_points(np.array([[0, 0, scale]]), rotation, translation, K)[0]
    
    cv2.line(img, tuple(origin), tuple(x_axis), (0, 0, 255), 3)  # X = Red
    cv2.line(img, tuple(origin), tuple(y_axis), (0, 255, 0), 3)  # Y = Green
    cv2.line(img, tuple(origin), tuple(z_axis), (255, 0, 0), 3)  # Z = Blue
