from __future__ import annotations

import numpy as np
import open3d as o3d


def _make_sphere(center, radius=0.03):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(np.asarray(center, dtype=np.float64))
    s.compute_vertex_normals()
    return s


def visualize_debug(
    pcd: o3d.geometry.PointCloud,
    lookat=(0, 0, 0),
    camera_positions=None,
    show_rays: bool = True,
    show_axes: bool = True,
    camera_sphere_radius: float = 0.03,
    axes_size: float = 0.5,
):
    geoms = []
    geoms.append(pcd)

    lookat = np.asarray(lookat, dtype=np.float64).reshape(3)

    if show_axes:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=[0, 0, 0]))
        look_s = _make_sphere(lookat, radius=camera_sphere_radius * 1.2)
        look_s.paint_uniform_color([0.0, 1.0, 0.0])
        geoms.append(look_s)

    if camera_positions is not None:
        cams = [np.asarray(c, dtype=np.float64).reshape(3) for c in camera_positions]

        for i, cpos in enumerate(cams):
            s = _make_sphere(cpos, radius=camera_sphere_radius)
            t = float(i) / max(1, (len(cams) - 1))
            s.paint_uniform_color([t, 0.2, 1.0 - t])
            geoms.append(s)

        if show_rays:
            points = [lookat]
            points.extend(cams)
            points = np.asarray(points, dtype=np.float64)

            lines = []
            for i in range(1, len(points)):
                lines.append([0, i])

            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
            )
            ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]]), (len(lines), 1)))
            geoms.append(ls)

    o3d.visualization.draw_geometries(geoms)
