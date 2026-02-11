from __future__ import annotations

from typing import List

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .config import DefectConfig
from .io import bbox_diag


def find_random_centers_trimesh(
    mesh: trimesh.Trimesh,
    radius: float,
    k: int,
    rng: np.random.Generator,
    candidates_mult: int = 50,
    min_center_dist_factor: float = 2.0,
) -> List[np.ndarray]:
    if k <= 0:
        return []

    n_candidates = max(k * candidates_mult, k)
    pts, _ = trimesh.sample.sample_surface(mesh, n_candidates, seed=rng)

    selected: List[np.ndarray] = []
    tree = None
    min_dist = min_center_dist_factor * radius

    for p in pts:
        if not selected:
            selected.append(p)
            tree = cKDTree(np.array(selected))
            if len(selected) >= k:
                break
            continue

        dist, _ = tree.query(p)
        if float(dist) >= min_dist:
            selected.append(p)
            tree = cKDTree(np.array(selected))
            if len(selected) >= k:
                break

    return selected


def make_hole(mesh: trimesh.Trimesh, center: np.ndarray, hole_radius: float) -> trimesh.Trimesh:
    verts = mesh.vertices
    faces = mesh.faces
    face_centers = verts[faces].mean(axis=1)
    dist = np.linalg.norm(face_centers - center[None, :], axis=1)
    mask = dist > hole_radius
    return trimesh.Trimesh(vertices=verts, faces=faces[mask], process=False)


def add_local_gaussian_bump(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
    sigma: float,
    amplitude: float,
) -> trimesh.Trimesh:
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals

    d = np.linalg.norm(verts - center[None, :], axis=1)
    mask = d < radius
    if not np.any(mask):
        return mesh.copy()

    denom = (sigma * radius) + 1e-12
    gauss = np.exp(-0.5 * (d[mask] / denom) ** 2)
    verts[mask] += normals[mask] * (gauss * amplitude)[:, None]
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def add_random_noise_trimesh(mesh: trimesh.Trimesh, rng: np.random.Generator, noise_level: float) -> trimesh.Trimesh:
    out = mesh.copy()
    if noise_level > 0:
        out.vertices += rng.uniform(-noise_level, noise_level, out.vertices.shape)
    return out


def inject_local_defects(mesh: trimesh.Trimesh, cfg: DefectConfig, rng: np.random.Generator) -> trimesh.Trimesh:
    mesh = mesh.copy()

    diag = bbox_diag(mesh)
    bump_radius = diag * cfg.bump_radius_ratio
    hole_radius = float(rng.uniform(cfg.hole_radius_range[0] * bump_radius,
                                   cfg.hole_radius_range[1] * bump_radius))

    n_centers = cfg.n_holes + cfg.n_gaussian_bumps
    if n_centers <= 0:
        return add_random_noise_trimesh(mesh, rng=rng, noise_level=cfg.noise_level)

    try:
        mesh.fill_holes()
    except Exception:
        pass

    centers = find_random_centers_trimesh(
        mesh,
        radius=hole_radius,
        k=n_centers,
        rng=rng,
        candidates_mult=cfg.candidates_mult,
        min_center_dist_factor=cfg.min_center_dist_factor,
    )

    if len(centers) < n_centers:
        raise RuntimeError(f"Only {len(centers)} centers found, needed {n_centers}. "
                           f"Try increasing candidates_mult or decreasing min_center_dist_factor/radius.")

    for i, c in enumerate(centers):
        mesh = make_hole(mesh, center=c, hole_radius=hole_radius)
        if i >= cfg.n_holes:
            mesh = add_local_gaussian_bump(
                mesh,
                center=c,
                radius=bump_radius,
                sigma=cfg.sigma,
                amplitude=cfg.amplitude,
            )

    mesh = add_random_noise_trimesh(mesh, rng=rng, noise_level=cfg.noise_level)
    return mesh
