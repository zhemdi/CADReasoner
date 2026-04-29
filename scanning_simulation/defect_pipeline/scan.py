from __future__ import annotations

import copy
from typing import List, Sequence, Set

import numpy as np
import open3d as o3d

from .config import MissingSurface, ScanConfig


def estimate_diameter(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    return float(np.linalg.norm(extent))


def _hpr_visible_indices(
    pcd: o3d.geometry.PointCloud,
    camera_pos: np.ndarray,
    hpr_radius: float,
) -> np.ndarray:
    _, idx = pcd.hidden_point_removal(camera_pos, hpr_radius)
    return np.asarray(idx, dtype=np.int64)


def _spherical_camera_positions(
    lookat: np.ndarray,
    radius: float,
    azimuth_deg: Sequence[float],
    elevation_deg: Sequence[float],
) -> List[np.ndarray]:
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))

    cams = []
    for a in az:
        for e in el:
            x = radius * np.cos(e) * np.cos(a)
            y = radius * np.cos(e) * np.sin(a)
            z = radius * np.sin(e)
            cams.append(lookat + np.array([x, y, z], dtype=np.float64))
    return cams


def angles_for_missing_surface(missing: MissingSurface, n_az: int, n_el: int) -> tuple[np.ndarray, np.ndarray]:
    if missing == "+x":
        az = np.linspace(90, 270, n_az, endpoint=False)
        el = np.linspace(-20, 80, n_el, endpoint=False)
    elif missing == "-x":
        az = np.linspace(-90, 90, n_az, endpoint=False)
        el = np.linspace(-90, 90, n_el, endpoint=False)
    elif missing == "+y":
        az = np.linspace(-180, 0, n_az, endpoint=False)
        el = np.linspace(-90, 90, n_el, endpoint=False)
    elif missing == "-y":
        az = np.linspace(0, 180, n_az, endpoint=False)
        el = np.linspace(-90, 90, n_el, endpoint=False)
    elif missing == "+z":
        n_az, n_el = n_el, n_az
        az = np.linspace(-180, 180, n_az, endpoint=False)
        el = np.linspace(-90, 0, n_el, endpoint=False) + 90 / (2 * n_el)
    elif missing == "-z":
        n_az, n_el = n_el, n_az
        az = np.linspace(-180, 180, n_az, endpoint=False)
        el = np.linspace(0, 90, n_el, endpoint=False)
    else:
        raise ValueError(f"Unknown missing surface: {missing}")
    return az, el


def _build_cameras(cfg: ScanConfig) -> List[np.ndarray]:
    lookat = np.asarray(cfg.lookat, dtype=np.float64).reshape(3)
    az, el = angles_for_missing_surface(cfg.missing_surface, cfg.n_az, cfg.n_el)
    return _spherical_camera_positions(lookat, cfg.radius, az, el)


# ------------------------------------------------------------------
# Fast path: returns only the unique visible indices as numpy array.
# No PointCloud copies, no Python set, no coloring.
# ------------------------------------------------------------------

def multi_camera_scan_indices(
    original_pcd: o3d.geometry.PointCloud,
    cfg: ScanConfig,
) -> np.ndarray:
    diameter = estimate_diameter(original_pcd)
    hpr_radius = cfg.hpr_factor * diameter
    camera_positions = _build_cameras(cfg)

    chunks: List[np.ndarray] = []
    for cam_pos in camera_positions:
        idx = _hpr_visible_indices(original_pcd, cam_pos, hpr_radius)
        chunks.append(idx)

    return np.unique(np.concatenate(chunks))


# ------------------------------------------------------------------
# Original interface (for notebooks / visualization)
# ------------------------------------------------------------------

def multi_camera_scan(
    original_pcd: o3d.geometry.PointCloud,
    cfg: ScanConfig,
    keep_scans: bool = False,
) -> tuple[List[o3d.geometry.PointCloud], Set[int], List[np.ndarray]]:
    diameter = estimate_diameter(original_pcd)
    hpr_radius = cfg.hpr_factor * diameter
    camera_positions = _build_cameras(cfg)

    pcds_colored: List[o3d.geometry.PointCloud] = []
    all_points: Set[int] = set()

    for i, cam_pos in enumerate(camera_positions):
        idx = _hpr_visible_indices(original_pcd, cam_pos, hpr_radius)
        all_points.update(idx.tolist())

        if keep_scans:
            scan_col = copy.deepcopy(original_pcd.select_by_index(idx))
            t = float(i) / max(1, (len(camera_positions) - 1))
            scan_col.paint_uniform_color([t, t, 0.0])
            pcds_colored.append(scan_col)

    return pcds_colored, all_points, camera_positions
