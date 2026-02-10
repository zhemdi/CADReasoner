import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv


class Plotter:
    def __init__(self, scale_gt: bool, scale_pred: bool):
        self.scale_gt = scale_gt
        self.scale_pred = scale_pred
        self.plotter = None
        self.iso_plotter = None

        self.view_img_size = 14 * 17
        self.rows = 4
        self.cols = 2
        self.mesh_render_img_size = self.view_img_size * 2

        self.views = {
            '-Z': self.minus_z_view,
            '+Z': self.plus_z_view,
            '+Y': self.plus_y_view,
            '-Y': self.minus_y_view,
            '+X': self.plus_x_view,
            '-X': self.minus_x_view,
        }

        self.align_coordinates = True
        self.show_axes = False

        self.cmap_gt = pv.LookupTable(
            values=np.array([[0, c, 0, 255] for c in range(0, 256)]),
            scalar_range=(0, 255), ramp="linear",
        )

        self.cmap_pred = pv.LookupTable(
            values=np.array([[c, 0, 0, 255] for c in range(0, 256)]),
            scalar_range=(0, 255), ramp="linear",
        )

        self.reload()

    def get_img(
        self,
        gt_mesh_path,
        pred_mesh_path,
    ):
        image = self._get_img(gt_mesh_path, self.cmap_gt, color=(0, 255, 0),  scale=self.scale_gt)
        if pred_mesh_path:
            pred_img = self._get_img(pred_mesh_path, self.cmap_pred, color=(255, 0, 0), scale=self.scale_pred)
            gt_r, gt_g, gt_b = image.split()
            pred_r, pred_g, pred_b = pred_img.split()
            image = Image.merge("RGB", (pred_r, gt_g, gt_b))
        return image

    def _get_img(
        self,
        mesh_path,
        cmap,
        color=None,
        scale=True,
    ):
        mesh = pv.read(mesh_path)

        if scale:
            mesh.translate([-0.5 * (mesh.bounds.x_min + mesh.bounds.x_max),
                            -0.5 * (mesh.bounds.y_min + mesh.bounds.y_max),
                            -0.5 * (mesh.bounds.z_min + mesh.bounds.z_max)], inplace=True)
            max_span = max(mesh.bounds.x_max - mesh.bounds.x_min, mesh.bounds.y_max - mesh.bounds.y_min, mesh.bounds.z_max - mesh.bounds.z_min)
            mesh.scale(200. / max_span, inplace=True)

        mesh.point_data.update(self.get_scalars(mesh))
        mesh_actor = self.plotter.add_mesh(
            mesh, reset_camera=False, color=None, scalars=None, cmap=cmap, show_scalar_bar=False
        )
        mesh_actor.use_bounds = False

        view_images = []
        for view_name, set_view_func in self.views.items():
            set_view_func(mesh)
            if view_name in ("Iso", "-Iso"):
                self.plotter.disable_parallel_projection()
            else:
                self.plotter.enable_parallel_projection()
                self.plotter.zoom_camera(1.7)  # 1.73 - perfect

            img_array = self.plotter.screenshot(return_img=True)
            pil_img = Image.fromarray(img_array)
            pil_img.thumbnail((self.view_img_size, self.view_img_size), resample=Image.Resampling.BILINEAR)
            if self.align_coordinates and view_name in ("-Z", "+Y", "+X", "Iso"):
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            view_images.append(pil_img)

        self.remove_meshes(mesh_actor)

        mesh_actor = self.iso_plotter.add_mesh(mesh, reset_camera=False, color=color)
        mesh_actor.use_bounds = False

        for view_name in ("Iso", "-Iso"):
            # self.plotter.disable_parallel_projection()
            if view_name == "Iso":
                self.iso_plotter.view_isometric()
            else:
                self.iso_plotter.view_isometric(negative=True)
            self.iso_plotter.zoom_camera(1.1)

            img_array = self.iso_plotter.screenshot(return_img=True)
            pil_img = Image.fromarray(img_array)
            pil_img.thumbnail((self.view_img_size, self.view_img_size), resample=Image.Resampling.BILINEAR)
            if self.align_coordinates and view_name in ("-Z", "+Y", "+X", "Iso"):
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            view_images.append(pil_img)

        _success = self.iso_plotter.remove_actor(mesh_actor, reset_camera=False, render=False)
        if not _success:
            self.reload()

        padding = 0
        total_width = round(self.cols * self.view_img_size + (self.cols - 1) * padding)
        total_height = round(self.rows * self.view_img_size + (self.rows - 1) * padding)
        collage = Image.new('RGB', (total_width, total_height), color="white")
        for i, img in enumerate(view_images):
            row = i // self.cols
            col = i % self.cols
            x_offset = col * (img.width + padding)
            y_offset = row * (img.height + padding)
            collage.paste(img, (x_offset, y_offset))

        return collage

    def reload(self):
        plotter = pv.Plotter(
            off_screen=True, window_size=(self.mesh_render_img_size, self.mesh_render_img_size), lighting='none')
        plotter.set_background('black')
        self.plotter = plotter

        plotter = pv.Plotter(
            off_screen=True, window_size=(self.mesh_render_img_size, self.mesh_render_img_size))
        plotter.set_background('black')
        self.iso_plotter = plotter

        x_min, x_max = -100, 100
        y_min, y_max = -100, 100
        z_min, z_max = -100, 100
        lim_points = [(x, y, z) for x in (x_min, x_max) for y in (y_min, y_max) for z in (z_min, z_max)]
        self.plotter.add_points(np.array(lim_points, dtype=float), color=(1, 1, 1), opacity=0, point_size=1)

        self.iso_plotter.add_points(np.array(lim_points, dtype=float), color=(1, 1, 1), opacity=0, point_size=1)

    def get_scalars(self, mesh):
        result = {}

        shift = 100
        scale = 255 / 200

        x_coords = mesh.points[:, 0]
        result["+X"] = (x_coords + shift) * scale
        result["-X"] = 255 - result["+X"]

        y_coords = mesh.points[:, 1]
        result["+Y"] = (y_coords + shift) * scale
        result["-Y"] = 255 - result["+Y"]

        z_coords = mesh.points[:, 2]
        result["+Z"] = (z_coords + shift) * scale
        result["-Z"] = 255 - result["+Z"]

        return result

    def remove_meshes(self, mesh):
        # clear from meshes
        _success = self.plotter.remove_actor(mesh, reset_camera=False, render=False)
        if not _success:
            self.reload()

    def minus_z_view(self, mesh):
        self.plotter.view_xy(negative=True)
        mesh.set_active_scalars("-Z")

    def plus_z_view(self, mesh):
        self.plotter.view_xy()
        mesh.set_active_scalars("+Z")

    def plus_y_view(self, mesh):
        self.plotter.view_xz(negative=True)
        mesh.set_active_scalars("+Y")

    def minus_y_view(self, mesh):
        self.plotter.view_xz()
        mesh.set_active_scalars("-Y")

    def plus_x_view(self, mesh):
        self.plotter.view_yz()
        mesh.set_active_scalars("+X")

    def minus_x_view(self, mesh):
        self.plotter.view_yz(negative=True)
        mesh.set_active_scalars("-X")
