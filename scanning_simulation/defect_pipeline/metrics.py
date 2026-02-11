from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import trimesh


@dataclass
class MeshStats:
    triangles: int
    vertices: int
    area: float


def mesh_stats_trimesh(mesh: trimesh.Trimesh) -> MeshStats:
    return MeshStats(
        triangles=int(len(mesh.faces)),
        vertices=int(len(mesh.vertices)),
        area=float(mesh.area),
    )


def count_triangles_o3d(mesh: o3d.geometry.TriangleMesh) -> int:
    return int(len(np.asarray(mesh.triangles)))
