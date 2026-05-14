from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, overload

import numpy as np
import open3d as o3d

from .config import PoissonConfig


@dataclass
class PoissonResult:
    mesh: o3d.geometry.TriangleMesh
    densities: Optional[np.ndarray]


def reconstruct_poisson(
    pcd: o3d.geometry.PointCloud,
    cfg: PoissonConfig,
    keep_densities: bool = True,
) -> o3d.geometry.TriangleMesh | PoissonResult:
    """
    When keep_densities=False returns bare TriangleMesh (saves memory).
    When keep_densities=True returns PoissonResult with densities.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=cfg.depth,
        width=cfg.width,
        scale=cfg.scale,
        linear_fit=cfg.linear_fit,
        n_threads=cfg.n_threads,
    )

    if cfg.density_quantile is not None:
        dens = np.asarray(densities, dtype=np.float64)
        thr = np.quantile(dens, cfg.density_quantile)
        mesh.remove_vertices_by_mask((dens < thr).tolist())
    else:
        dens = None

    mesh.compute_vertex_normals()

    if not keep_densities:
        return mesh

    if dens is None:
        dens = np.asarray(densities, dtype=np.float64)
    return PoissonResult(mesh=mesh, densities=dens)
