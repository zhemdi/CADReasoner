from __future__ import annotations

import numpy as np
import open3d as o3d

from .config import CleanupConfig


def remove_unsupported_triangles(
    mesh: o3d.geometry.TriangleMesh,
    support_pcd: o3d.geometry.PointCloud,
    max_dist: float,
) -> o3d.geometry.TriangleMesh:
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    centers = verts[tris].mean(axis=1)

    centers_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    d = np.asarray(centers_pcd.compute_point_cloud_distance(support_pcd))
    tri_mask = (d > max_dist).tolist()

    mesh.remove_triangles_by_mask(tri_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def cleanup_poisson_mesh(
    mesh: o3d.geometry.TriangleMesh,
    support_pcd: o3d.geometry.PointCloud,
    cfg: CleanupConfig,
) -> o3d.geometry.TriangleMesh:
    nn = np.asarray(support_pcd.compute_nearest_neighbor_distance())
    tau = cfg.tau_nn_mult * nn.mean()
    mesh = remove_unsupported_triangles(mesh, support_pcd, max_dist=tau)
    mesh.compute_vertex_normals()
    return mesh
