import numpy as np
import trimesh
from trimesh.proximity import closest_point
from scipy.spatial import cKDTree

def farthest_point_sampling(points: np.ndarray, m: int) -> np.ndarray:
    """Farthest point sampling: returns the indices of the m most spread-out points."""
    N = len(points)
    m = min(m, N)
    idxs = np.empty(m, dtype=np.int64)
    idxs[0] = np.random.randint(0, N)
    dists = np.full(N, np.inf)

    for i in range(1, m):
        last = points[idxs[i-1]]
        d = np.linalg.norm(points - last, axis=1)
        dists = np.minimum(dists, d)
        idxs[i] = np.argmax(dists)
    return idxs

def compute_far_points(gt_points, pred_points, thresh_percentile=95):
    """
    Returns:
        gt_far, gt_close, pred_far, pred_close
    where "far" = points whose distance to the nearest opposite point exceeds the percentile threshold.
    """
    tree_gt = cKDTree(gt_points)
    tree_pred = cKDTree(pred_points)

    gt2pred_d, _ = tree_pred.query(gt_points, k=1)
    pred2gt_d, _ = tree_gt.query(pred_points, k=1)

    thr_gt = np.percentile(gt2pred_d, thresh_percentile)
    thr_pred = np.percentile(pred2gt_d, thresh_percentile)

    gt_far_mask = gt2pred_d > thr_gt
    pred_far_mask = pred2gt_d > thr_pred

    gt_far = gt_points[gt_far_mask]
    gt_close = gt_points[~gt_far_mask]
    pred_far = pred_points[pred_far_mask]
    pred_close = pred_points[~pred_far_mask]

    return gt_far, gt_close, pred_far, pred_close, thr_gt, thr_pred


def closest_points_on_mesh(pred_mesh: trimesh.Trimesh, query_pts: np.ndarray):
    try:
        # requires rtree + libspatialindex
        from trimesh.proximity import closest_point
        cp, dists, _ = closest_point(pred_mesh, query_pts)
        return cp, dists
    except Exception:
        # fallback: KD-tree over the vertices
        verts = np.asarray(pred_mesh.vertices)
        kdt = cKDTree(verts)
        dists, idx = kdt.query(query_pts, k=1)
        cp = verts[idx]
        return cp, dists

def make_gt_vector_features(
    gt_points: np.ndarray,
    pred_mesh: trimesh.Trimesh | None,
    m: int = 256,
):
    """
    Returns:
      features: (k,6) = [x,y,z, vx,vy,vz]
      pts_sel: (k,3)
      targets: (k,3) nearest points on the mesh, or the bbox centre
      dists:   (k,)
    where k = min(m, number of available points)
    """

    # 1) pick GT points (preferring gt_far)
    idx = farthest_point_sampling(gt_points, m)
    pts_sel = gt_points[idx]

    # 2) target for the displacement vector
    if pred_mesh is None:
        mn, mx = gt_points.min(0), gt_points.max(0)
        center = (mn + mx) * 0.5
        targets = np.repeat(center[None, :], len(pts_sel), axis=0)
        dists = np.linalg.norm(targets - pts_sel, axis=1)
    else:
        # nearest points on the predicted mesh surface / vertices
        targets, dists = closest_points_on_mesh(pred_mesh, pts_sel)

    # 3) build the features
    vectors = targets - pts_sel
    features = np.hstack([pts_sel, vectors])
    return features, pts_sel, targets, dists


import numpy as np
import trimesh


def make_pc_far(gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh | None,
                m_each: int = 128, thresh_percentile: float = 90) -> np.ndarray:
    """
      the first m_each come from GT->pred, the next from pred->GT.
      if pred_mesh is None: use GT->centre and centre->GT instead.
    """

    if pred_mesh is None:
            pts, _ = trimesh.sample.sample_surface(gt_mesh, 2 * m_each)
            zero = np.zeros(3, dtype=pts.dtype)
            s1 = pts[:m_each]
            v1 = -pts[:m_each]
            s2 = np.repeat(zero[None, :], m_each, axis=0)
            v2 = pts[m_each:]
            return np.vstack([np.hstack([s1, v1]), np.hstack([s2, v2])]).astype(np.float32)


    # sample a dense set of points
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, 30000)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, 30000)

    # locate the mismatched regions
    gt_far, _, pred_far, _, _, _ = compute_far_points(gt_points, pred_points, thresh_percentile)

    # farthest point sampling for an even selection
    gt_far = gt_far[farthest_point_sampling(gt_far, min(m_each, len(gt_far)))]
    pred_far = pred_far[farthest_point_sampling(pred_far, min(m_each, len(pred_far)))]

    # vectors from GT->pred and back
    feat_gt2pred, _, _, _ = make_gt_vector_features(gt_far, pred_mesh, m=len(gt_far))
    feat_pred2gt, _, _, _ = make_gt_vector_features(pred_far, gt_mesh, m=len(pred_far))

    
    assert len(feat_gt2pred.shape) == 2
    assert len(feat_pred2gt.shape) == 2
    assert feat_gt2pred.shape[1] == 6
    assert feat_pred2gt.shape[1] == 6
    assert feat_pred2gt.shape[0] == m_each
    assert feat_gt2pred.shape[0] == m_each

    # just in case
    # pad both halves to the same length
    def pad(arr, target):
        if len(arr) < target:
            pad_n = target - len(arr)
            arr = np.vstack([arr, np.zeros((pad_n, arr.shape[1]))])
        return arr[:target]

    feat_gt2pred = pad(feat_gt2pred, m_each)
    feat_pred2gt = pad(feat_pred2gt, m_each)

    return np.vstack([feat_gt2pred, feat_pred2gt]).astype(np.float32)
