from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


def load_and_normalize_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    mn, mx = mesh.bounds
    center = (mn + mx) * 0.5
    scale = (mx - mn).max()
    mesh.vertices = (mesh.vertices - center) / (scale + 1e-12)
    return mesh


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_mesh_trimesh(mesh: trimesh.Trimesh, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def bbox_diag(mesh: trimesh.Trimesh) -> float:
    bbox = mesh.bounds
    return float(np.linalg.norm(bbox[1] - bbox[0]))
