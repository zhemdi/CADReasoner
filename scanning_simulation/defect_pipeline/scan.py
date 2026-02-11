from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Sequence, Set

import numpy as np
import open3d as o3d

from .config import MissingSurface, ScanConfig


@dataclass
class ScanResult:
    pcd_visible: o3d.geometry.PointCloud
    visible_indices: np.ndarray
    camera_positions: List[np.ndarray]
    scans_colored: List[o3d.geometry.PointCloud]


def estimate_diameter(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    return float(np.linalg.norm(extent))


def single_camera_scan(
    pcd: o3d.geometry.PointCloud,
    camera_pos,
    diameter: float,
    hpr_factor: float = 100.0,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    camera_pos = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    hpr_radius = hpr_factor * diameter
    _, idx = pcd.hidden_point_removal(camera_pos, hpr_radius)
    idx = np.asarray(idx, dtype=np.int64)
    return pcd.select_by_index(idx), idx


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
    """
    таблица miss_surf_agles.
    """

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
        az = np.linspace(-180, 180, n_az, endpoint=False)
        el = np.linspace(-90, 0, n_el, endpoint=False)
    elif missing == "-z":
        az = np.linspace(-180, 180, n_az, endpoint=False)
        el = np.linspace(0, 90, n_el, endpoint=False)
    else:
        raise ValueError(f"Unknown missing surface: {missing}")
    return az, el


def multi_camera_scan(
    original_pcd: o3d.geometry.PointCloud,
    cfg: ScanConfig,
    keep_scans: bool = False,
) -> tuple[List[o3d.geometry.PointCloud], Set[int], List[np.ndarray]]:
    lookat = np.asarray(cfg.lookat, dtype=np.float64).reshape(3)
    diameter = estimate_diameter(original_pcd)

    az, el = angles_for_missing_surface(cfg.missing_surface, cfg.n_az, cfg.n_el)
    camera_positions = _spherical_camera_positions(
        lookat=lookat,
        radius=cfg.radius,
        azimuth_deg=az,
        elevation_deg=el,
    )

    pcds_colored: List[o3d.geometry.PointCloud] = []
    all_points: Set[int] = set()

    for i, cam_pos in enumerate(camera_positions):
        scan_pcd, idx = single_camera_scan(
            original_pcd,
            camera_pos=cam_pos,
            diameter=diameter,
            hpr_factor=cfg.hpr_factor,
        )
        all_points.update(idx.tolist())

        if keep_scans:
            scan_col = copy.deepcopy(scan_pcd)
            t = float(i) / max(1, (len(camera_positions) - 1))
            scan_col.paint_uniform_color([t, t, 0.0])
            pcds_colored.append(scan_col)

    return pcds_colored, all_points, camera_positions
