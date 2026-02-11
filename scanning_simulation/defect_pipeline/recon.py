from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .config import PoissonConfig


@dataclass
class PoissonResult:
    mesh: o3d.geometry.TriangleMesh
    densities: np.ndarray


def reconstruct_poisson(pcd: o3d.geometry.PointCloud, cfg: PoissonConfig) -> PoissonResult:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=cfg.depth,
        width=cfg.width,
        scale=cfg.scale,
        linear_fit=cfg.linear_fit,
        n_threads=cfg.n_threads,
    )
    dens = np.asarray(densities, dtype=np.float64)

    # опционально: фильтр низкой плотности
    if cfg.density_quantile is not None:
        thr = np.quantile(dens, cfg.density_quantile)
        mesh.remove_vertices_by_mask((dens < thr).tolist())

    mesh.compute_vertex_normals()
    return PoissonResult(mesh=mesh, densities=dens)
