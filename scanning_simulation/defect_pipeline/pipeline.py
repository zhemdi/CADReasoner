from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh

from .cleanup import cleanup_poisson_mesh
from .config import PipelineConfig
from .defects_local import inject_local_defects
from .io import load_and_normalize_mesh
from .metrics import mesh_stats_trimesh, count_triangles_o3d
from .recon import reconstruct_poisson
from .scan import multi_camera_scan, multi_camera_scan_indices


def make_o3d_pc(mesh: trimesh.Trimesh, pc_size: int) -> o3d.geometry.PointCloud:
    pts, face_idx = trimesh.sample.sample_surface(mesh, pc_size)

    faces = mesh.faces[face_idx]
    tri_xyz = mesh.vertices[faces]
    tri_vn = mesh.vertex_normals[faces]

    A = tri_xyz[:, 0, :]
    B = tri_xyz[:, 1, :]
    C = tri_xyz[:, 2, :]
    v0 = B - A
    v1 = C - A
    v2 = pts - A

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-18

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    nrm = tri_vn[:, 0, :] * u[:, None] + tri_vn[:, 1, :] * v[:, None] + tri_vn[:, 2, :] * w[:, None]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pts)
    o3d_pc.normals = o3d.utility.Vector3dVector(nrm)
    return o3d_pc


@dataclass
class PipelineResult:
    mesh_defected: trimesh.Trimesh
    mesh_recon_o3d: o3d.geometry.TriangleMesh
    original_mesh: trimesh.Trimesh
    original_pcd: o3d.geometry.PointCloud
    scanned_pcd: o3d.geometry.PointCloud
    camera_positions: list[np.ndarray]
    timings: Dict[str, float]
    stats: Dict[str, Any]


class DefectPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.inp.seed)

    # ------------------------------------------------------------------
    # Full run — keeps all intermediate data (for notebooks / debugging)
    # ------------------------------------------------------------------

    def run_from_path(self, mesh_path: str) -> PipelineResult:
        original_mesh = load_and_normalize_mesh(mesh_path)
        return self.run(original_mesh)

    def run(self, mesh: trimesh.Trimesh) -> PipelineResult:
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        start_triangles = len(mesh.faces)
        original_pcd = make_o3d_pc(mesh, self.cfg.inp.pc_size)
        timings["pc_sample"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        pcds_colored, all_points, camera_positions = multi_camera_scan(
            original_pcd, cfg=self.cfg.scan, keep_scans=False,
        )
        idx = np.fromiter(all_points, dtype=np.int64)
        scanned_pcd = original_pcd.select_by_index(idx)
        timings["scan_hpr"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        poisson = reconstruct_poisson(scanned_pcd, self.cfg.poisson)
        mesh_poisson = poisson.mesh
        timings["poisson"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        mesh_clean = cleanup_poisson_mesh(mesh_poisson, scanned_pcd, self.cfg.cleanup)
        timings["cleanup"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        mesh_tm = trimesh.Trimesh(
            vertices=np.asarray(mesh_clean.vertices),
            faces=np.asarray(mesh_clean.triangles),
            process=False,
        )
        timings["to_trimesh"] = time.perf_counter() - t4

        t5 = time.perf_counter()
        mesh_def = inject_local_defects(mesh_tm, self.cfg.defect, rng=self.rng)
        timings["local_defects"] = time.perf_counter() - t5

        t6 = time.perf_counter()
        mesh_def = self._simplify(mesh_def, start_triangles)
        timings["simplify"] = time.perf_counter() - t6

        stats = {
            "start_triangles": int(start_triangles),
            "recon_triangles": int(count_triangles_o3d(mesh_clean)),
            "final": mesh_stats_trimesh(mesh_def).__dict__,
            "config_seed": self.cfg.inp.seed,
        }

        return PipelineResult(
            mesh_defected=mesh_def,
            mesh_recon_o3d=mesh_clean,
            original_mesh=mesh,
            original_pcd=original_pcd,
            scanned_pcd=scanned_pcd,
            camera_positions=camera_positions,
            timings=timings,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Lightweight run — returns only (defected_mesh, timings, stats).
    # Frees every intermediate object ASAP. Use for batch processing.
    # ------------------------------------------------------------------

    def run_for_export(
        self, mesh_path: str,
    ) -> Tuple[trimesh.Trimesh, Dict[str, float], Dict[str, Any]]:
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        mesh = load_and_normalize_mesh(mesh_path)
        start_triangles = len(mesh.faces)
        original_pcd = make_o3d_pc(mesh, self.cfg.inp.pc_size)
        del mesh
        timings["pc_sample"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        visible_idx = multi_camera_scan_indices(original_pcd, cfg=self.cfg.scan)
        scanned_pcd = original_pcd.select_by_index(visible_idx)
        del original_pcd, visible_idx
        timings["scan_hpr"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        mesh_poisson = reconstruct_poisson(scanned_pcd, self.cfg.poisson, keep_densities=False)
        timings["poisson"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        mesh_clean = cleanup_poisson_mesh(mesh_poisson, scanned_pcd, self.cfg.cleanup)
        del mesh_poisson, scanned_pcd
        timings["cleanup"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        recon_tri_count = int(count_triangles_o3d(mesh_clean))
        mesh_tm = trimesh.Trimesh(
            vertices=np.asarray(mesh_clean.vertices),
            faces=np.asarray(mesh_clean.triangles),
            process=False,
        )
        del mesh_clean
        timings["to_trimesh"] = time.perf_counter() - t4

        t5 = time.perf_counter()
        mesh_def = inject_local_defects(mesh_tm, self.cfg.defect, rng=self.rng)
        del mesh_tm
        timings["local_defects"] = time.perf_counter() - t5

        t6 = time.perf_counter()
        mesh_def = self._simplify(mesh_def, start_triangles)
        timings["simplify"] = time.perf_counter() - t6

        stats = {
            "start_triangles": int(start_triangles),
            "recon_triangles": recon_tri_count,
            "final": mesh_stats_trimesh(mesh_def).__dict__,
            "config_seed": self.cfg.inp.seed,
        }

        return mesh_def, timings, stats

    # ------------------------------------------------------------------

    def _simplify(self, mesh_def: trimesh.Trimesh, start_triangles: int) -> trimesh.Trimesh:
        if self.cfg.simplify.coef_triangle_size is None:
            return mesh_def
        coef = float(self.cfg.simplify.coef_triangle_size)
        _m = o3d.geometry.TriangleMesh()
        _m.vertices = o3d.utility.Vector3dVector(mesh_def.vertices)
        _m.triangles = o3d.utility.Vector3iVector(mesh_def.faces)
        tri_cnt = count_triangles_o3d(_m)
        target = int(start_triangles * coef) if tri_cnt > start_triangles * coef else tri_cnt
        _m = _m.simplify_quadric_decimation(target)
        _m.compute_vertex_normals()
        return trimesh.Trimesh(
            vertices=np.asarray(_m.vertices),
            faces=np.asarray(_m.triangles),
            process=False,
        )
