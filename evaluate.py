import multiprocessing
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm


def compute_chamfer_distance(gt_mesh, pred_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume

        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except:
        pass


def compute_cd_iou(arg):
    gt_mesh_path, pred_mesh_path = arg
    cd, iou = None, None

    try:
        pred_mesh = trimesh.load_mesh(pred_mesh_path)
        center = (pred_mesh.bounds[0] + pred_mesh.bounds[1]) / 2.0
        pred_mesh.apply_translation(-center)
        extent = np.max(pred_mesh.extents)
        if extent > 1e-7:
            pred_mesh.apply_scale(1.0 / extent)
        pred_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        gt_mesh = trimesh.load_mesh(gt_mesh_path)
        center = (gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0
        gt_mesh.apply_translation(-center)
        extent = np.max(gt_mesh.extents)
        if extent > 1e-7:
            gt_mesh.apply_scale(1.0 / extent)
        gt_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        try:
            cd = compute_chamfer_distance(gt_mesh, pred_mesh)
        except:
            pass
        try:
            iou = compute_iou(gt_mesh, pred_mesh)
        except:
            pass

    except:
        pass

    return dict(cd=cd, iou=iou)


def compute_metrics(gt_pred_pairs):
    with multiprocessing.Pool(processes=8) as pool:
        per_sample_metrics = list(
            tqdm(pool.imap_unordered(compute_cd_iou, gt_pred_pairs), total=len(gt_pred_pairs))
        )
    cds = [m["cd"] for m in per_sample_metrics if m["cd"] is not None]
    ious = [m["iou"] for m in per_sample_metrics if m["iou"] is not None]

    cd = float(np.median(cds) * 1000)
    iou = float(np.mean(ious))
    ir_cd = (len(gt_pred_pairs) - len(cds)) / len(gt_pred_pairs)
    ir_iou = (len(gt_pred_pairs) - len(ious)) / len(gt_pred_pairs)

    return dict(cd=cd, iou=iou, ir_cd=ir_cd, ir_iou=ir_iou)


def run(gt_dirs, pred_dirs, output_dirs, best_from_all_prev_iters):
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    for gt_dir, pred_dir, output_dir in tqdm(zip(gt_dirs, pred_dirs, output_dirs), total=len(pred_dirs)):
        gt_dir = Path(gt_dir)
        pred_dir = Path(pred_dir)

        results = []
        num_iters = len(list(pred_dir.glob("*")))

        for iter_num in range(1, num_iters + 1):
            gt_pred_pairs = []

            for gt_mesh_path in gt_dir.glob("*.stl"):
                if best_from_all_prev_iters:
                    predicts = [stl for iter_dir in range(1, iter_num + 1)
                                for stl in (pred_dir / str(iter_dir) / str(gt_mesh_path.stem)).glob("*.stl")]
                else:
                    predicts = [stl for iter_dir in range(iter_num, iter_num + 1)
                                for stl in (pred_dir / str(iter_dir) / str(gt_mesh_path.stem)).glob("*.stl")]

                if predicts:
                    best_pred_stl = min(predicts, key=lambda x: float(x.stem) if is_float(x.stem) else float("inf"))
                else:
                    best_pred_stl = None

                gt_pred_pairs.append((gt_mesh_path, best_pred_stl))

            metrics = compute_metrics(gt_pred_pairs)

            results.append({
                'dataset': gt_dir.stem,
                'iter_num': iter_num,
                'cd': metrics['cd'],
                'iou': metrics['iou'],
            })

            print(gt_dir.stem, f"iter_num: {iter_num}", metrics)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Test dataset directory with .stl files.')
    parser.add_argument('--pred_dir', type=str, help='Directory with test predictions.')
    parser.add_argument('--best_from_all_prev_iters', default=True, type=lambda x: x.lower() == 'true',
                        help='Whether to select the best predictions (closest by CD to the target) from all previous '
                             'iterations or only from the current iteration.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    pred_dirs = [args.pred_dir]
    gt_dirs = [args.dataset]
    output_dirs = [str(Path(pred_dir).parent) for pred_dir in pred_dirs]

    run(
        gt_dirs=gt_dirs,
        pred_dirs=pred_dirs,
        output_dirs=output_dirs,
        best_from_all_prev_iters=args.best_from_all_prev_iters
    )
