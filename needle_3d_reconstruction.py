"""
Stereo 3D Needle Midpoint Reconstruction

Pipeline:
  1. Run YOLO + SAM3 on left and right images.
  2. Extract 2D skeleton from each mask.
  3. Undistort skeleton points.
  4. Epipolar matching using stereo calibration.
  5. Triangulate matched pairs into a 3D point cloud.
  6. Fit a 3D circle (PCA plane + 2D LSQ circle + RANSAC).
  7. Recover the arc midpoint as the gripping target.
  8. Reproject and visualize; save result to .npz.

Output: 3D midpoint in the left camera frame (mm), plus plane normal and radius.
"""

import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import skeletonize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Stereo 3D needle midpoint reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--left',  required=True, help='Path to the left image')
    p.add_argument('--right', required=True, help='Path to the right image')
    p.add_argument('--weights', required=True, help='Path to YOLO weights (.pt)')
    p.add_argument('--calib',   default=None,
                   help='Path to stereo_calib.npz (K1,K2,D1,D2,R,T,img_w,img_h). '
                        'If omitted, built-in 1920x1080 calibration is used.')
    p.add_argument('--output-dir', default='outputs_3d',
                   help='Directory for output files')
    p.add_argument('--conf',        type=float, default=0.45, help='YOLO confidence threshold')
    p.add_argument('--box-expand',  type=float, default=1.1,  help='YOLO box expand ratio')
    p.add_argument('--epipolar-thresh', type=float, default=3.0,
                   help='Max epipolar distance (px) for skeleton matching')
    p.add_argument('--ransac-thresh',   type=float, default=0.5,
                   help='RANSAC inlier threshold for 3D circle fit (mm)')
    p.add_argument('--ransac-iters',    type=int,   default=300,
                   help='Number of RANSAC iterations')
    p.add_argument('--no-vis', action='store_true',
                   help='Skip saving visualisation plots')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

# Built-in calibration from stereo_calibration_new.yaml
_DEFAULT_IMG_W, _DEFAULT_IMG_H = 1920, 1080

_DEFAULT_K1 = np.array([[1342.1035839657784, 0.0,               920.5005859206799],
                         [   0.0,            1340.9475695814524, 538.1550519345003],
                         [   0.0,               0.0,               1.0           ]], dtype=np.float64)

_DEFAULT_K2 = np.array([[1351.9176425849425, 0.0,                1049.8154931878398],
                         [   0.0,            1349.8914218044931,  539.0721823602935],
                         [   0.0,               0.0,                1.0            ]], dtype=np.float64)

_DEFAULT_D1 = np.array([-0.02274402282338767,  0.081934076877251946, -0.0071887339138836806,
                          0.0043763375152072346, -0.030801832698880989], dtype=np.float64)
_DEFAULT_D2 = np.array([-0.00032074923748045745,  0.086133513136658554, -0.0034036936374620646,
                          0.010804767877325474,   -0.11458384021854233], dtype=np.float64)

_DEFAULT_R = np.array([[ 0.9999361032042462,  0.0026727319508472271, -0.010983897879437153],
                        [-0.0026642644411264756,  0.99999614236457202,  0.00078546226037539118],
                        [ 0.010985954837643046, -0.0007561480633086141,  0.99993936668000605]], dtype=np.float64)

_DEFAULT_T = np.array([[-0.004468698370539214],
                        [-0.0001984338841315943],
                        [-0.00043762260979519782]], dtype=np.float64)


def load_calibration(path=None):
    """Load calibration from .npz, or return built-in defaults."""
    if path is not None:
        d = np.load(path)
        return (d['K1'], d['K2'], d['D1'], d['D2'],
                d['R'], d['T'].reshape(3, 1),
                int(d['img_w']), int(d['img_h']))
    return (_DEFAULT_K1, _DEFAULT_K2, _DEFAULT_D1, _DEFAULT_D2,
            _DEFAULT_R, _DEFAULT_T, _DEFAULT_IMG_W, _DEFAULT_IMG_H)

# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------

def keep_large_components(mask, min_ratio=0.15):
    """Keep all connected components whose area >= min_ratio * largest CC area."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return mask.astype(bool)
    threshold = areas.max() * min_ratio
    keep = np.zeros_like(m, dtype=bool)
    for i, a in enumerate(areas, start=1):
        if a >= threshold:
            keep |= (labels == i)
    return keep


def order_skeleton(skel):
    """Walk skeleton pixels along the curve. Returns (N, 2) array of (u, v) in order."""
    skel = skel.astype(np.uint8)
    pix = set(map(tuple, np.argwhere(skel)))  # (y, x) pairs
    if not pix:
        return np.empty((0, 2), dtype=np.float32)

    nbrs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def neighbors(p):
        y, x = p
        return [(y+dy, x+dx) for dy, dx in nbrs8 if (y+dy, x+dx) in pix]

    endpoints = [p for p in pix if len(neighbors(p)) == 1]
    start = endpoints[0] if endpoints else next(iter(pix))

    chain = [start]
    visited = {start}
    while True:
        cur = chain[-1]
        cands = [q for q in neighbors(cur) if q not in visited]
        if not cands:
            break
        cands.sort(key=lambda q: (abs(q[0]-cur[0]) + abs(q[1]-cur[1]), q))
        nxt = cands[0]
        chain.append(nxt)
        visited.add(nxt)

    return np.array([(x, y) for (y, x) in chain], dtype=np.float32)


def extract_skeleton(mask):
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    # Crop to tight bounding box of the foreground before skeletonizing,
    # then offset the resulting pixel coordinates back to original image space.
    rows = np.any(cleaned, axis=1)
    cols = np.any(cleaned, axis=0)
    if not rows.any():
        return np.empty((0, 2), dtype=np.float32), cleaned, np.zeros_like(cleaned)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    cropped = cleaned[r0:r1+1, c0:c1+1]

    skel_crop = skeletonize(cropped)
    ordered_crop = order_skeleton(skel_crop)

    # Translate crop-relative (x, y) back to full-image coordinates
    if len(ordered_crop):
        ordered = ordered_crop + np.array([c0, r0], dtype=np.float32)
    else:
        ordered = ordered_crop

    # Reconstruct full-size skeleton image for callers that use it directly
    skel = np.zeros_like(cleaned)
    skel[r0:r1+1, c0:c1+1] = skel_crop

    return ordered, cleaned, skel

# ---------------------------------------------------------------------------
# Undistortion
# ---------------------------------------------------------------------------

def undistort_pixels(pts, K, D):
    """Remove lens distortion. Input/output: (N, 2) pixel coordinates."""
    if len(pts) == 0:
        return pts
    pts_in = pts.reshape(-1, 1, 2).astype(np.float64)
    pts_out = cv2.undistortPoints(pts_in, K, D, P=K)
    return pts_out.reshape(-1, 2).astype(np.float32)

# ---------------------------------------------------------------------------
# Epipolar matching
# ---------------------------------------------------------------------------

def skew(v):
    v = v.ravel()
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]], dtype=np.float64)


def fundamental_from_KRT(K1, K2, R, T):
    E = skew(T) @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def epipolar_match(pts_L, pts_R, F, max_dist_px=3.0):
    """Match each left skeleton point to the nearest right point along its epipolar line."""
    if len(pts_L) == 0 or len(pts_R) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    ones = np.ones((len(pts_L), 1), dtype=np.float64)
    pL_h = np.hstack([pts_L.astype(np.float64), ones])
    lines = (F @ pL_h.T).T
    norms = np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    lines = lines / norms

    pR = pts_R.astype(np.float64)
    dists = np.abs(lines[:, [0]] * pR[:, 0][None, :] +
                   lines[:, [1]] * pR[:, 1][None, :] +
                   lines[:, [2]])

    best_idx = dists.argmin(axis=1)
    best_d   = dists[np.arange(len(pts_L)), best_idx]
    keep = best_d <= max_dist_px
    return pts_L[keep], pts_R[best_idx[keep]]

# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def projection_matrices(K1, K2, R, T):
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T])
    return P1, P2


def triangulate(P1, P2, pts1, pts2):
    if len(pts1) == 0:
        return np.empty((0, 3))
    X4 = cv2.triangulatePoints(P1, P2,
                                pts1.T.astype(np.float64),
                                pts2.T.astype(np.float64))
    return (X4[:3] / X4[3]).T

# ---------------------------------------------------------------------------
# 3D circle fit
# ---------------------------------------------------------------------------

def fit_plane_pca(pts):
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, Vt[0], Vt[1], Vt[2]


def fit_circle_2d(p, q):
    A = np.column_stack([p, q, np.ones_like(p)])
    b = -(p**2 + q**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    pc, qc = -D / 2.0, -E / 2.0
    r2 = pc**2 + qc**2 - F
    if r2 <= 0:
        return None
    return float(pc), float(qc), float(np.sqrt(r2))


def fit_circle_3d(pts, ransac_iters=300, ransac_thresh=0.5, rng=None):
    pts = np.asarray(pts, dtype=np.float64)
    n_pts = len(pts)
    if n_pts < 5:
        return None
    rng = rng or np.random.default_rng(0)

    def _fit_once(subset):
        c, e1, e2, n = fit_plane_pca(subset)
        p = (subset - c) @ e1
        q = (subset - c) @ e2
        circ = fit_circle_2d(p, q)
        if circ is None:
            return None
        pc, qc, r = circ
        C3 = c + pc * e1 + qc * e2
        return C3, r, n, e1, e2

    def _residuals(model, all_pts):
        C3, r, n, _e1, _e2 = model
        d_plane  = (all_pts - C3) @ n
        proj     = all_pts - np.outer(d_plane, n)
        d_inplane = np.linalg.norm(proj - C3, axis=1) - r
        return np.sqrt(d_plane**2 + d_inplane**2)

    best_inliers = None
    best_model   = None
    sample_size  = max(5, int(0.1 * n_pts))

    for _ in range(ransac_iters):
        idx = rng.choice(n_pts, size=sample_size, replace=False)
        m = _fit_once(pts[idx])
        if m is None:
            continue
        res = _residuals(m, pts)
        inliers = res < ransac_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_model   = m

    if best_model is None:
        return None

    final = _fit_once(pts[best_inliers])
    if final is None:
        final = best_model
    C3, r, n, e1, e2 = final
    res = _residuals(final, pts[best_inliers])
    rms = float(np.sqrt(np.mean(res**2)))

    return {
        'center':  C3,
        'radius':  r,
        'normal':  n / np.linalg.norm(n),
        'e1':      e1,
        'e2':      e2,
        'inliers': best_inliers,
        'rms_mm':  rms,
        'n_in':    int(best_inliers.sum()),
        'n_total': n_pts,
    }

# ---------------------------------------------------------------------------
# Arc midpoint
# ---------------------------------------------------------------------------

def arc_midpoint_3d(pts3d, circle):
    inliers = pts3d[circle['inliers']]
    C  = circle['center']
    r  = circle['radius']
    e1 = circle['e1']
    e2 = circle['e2']
    n  = circle['normal']

    centered = inliers - C
    angles = np.arctan2(centered @ e2, centered @ e1)

    sorted_a = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_a, [sorted_a[0] + 2 * np.pi]]))
    gap_idx = int(np.argmax(gaps))

    a_end   = float(sorted_a[gap_idx])
    a_start = float(sorted_a[(gap_idx + 1) % len(sorted_a)])
    a_end_u = a_end if a_end > a_start else a_end + 2 * np.pi
    a_mid   = 0.5 * (a_start + a_end_u)
    sweep   = a_end_u - a_start

    mid_3d  = C + r * (np.cos(a_mid)   * e1 + np.sin(a_mid)   * e2)
    end1_3d = C + r * (np.cos(a_start) * e1 + np.sin(a_start) * e2)
    end2_3d = C + r * (np.cos(a_end)   * e1 + np.sin(a_end)   * e2)

    return {
        'midpoint_3d':   mid_3d,
        'endpoint1_3d':  end1_3d,
        'endpoint2_3d':  end2_3d,
        'arc_sweep_deg': float(np.degrees(sweep)),
        'mid_angle':     a_mid,
        'plane_normal':  n,
    }


def arc_midpoint_from_matches(pts3d, trim_frac=0.05):
    """Find midpoint by greedy nearest-neighbour chain, trimming noisy ends,
    then locating the geometric arc-length midpoint."""
    pts = np.asarray(pts3d, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return None

    # Start from the point furthest from the centroid (one endpoint of the arc)
    centroid = pts.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))

    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    for _ in range(n - 1):
        last = pts[order[-1]]
        dists = np.linalg.norm(pts - last, axis=1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True

    sorted_pts = pts[order]

    # Trim noisy skeleton ends before computing midpoint
    trim = max(1, int(n * trim_frac))
    trimmed = sorted_pts[trim:-trim] if len(sorted_pts) > 2 * trim + 1 else sorted_pts

    # Arc-length midpoint: find the point at 50% cumulative arc length
    seg_lens = np.linalg.norm(np.diff(trimmed, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    half = cum_len[-1] / 2.0
    mid_idx = int(np.searchsorted(cum_len, half, side='left'))
    mid_idx = np.clip(mid_idx, 0, len(trimmed) - 1)
    mid_3d = trimmed[mid_idx]

    return {
        'midpoint_3d':  mid_3d,
        'endpoint1_3d': trimmed[0],
        'endpoint2_3d': trimmed[-1],
        'n_pts':        len(trimmed),
    }

# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def project(P, X3):
    Xh = np.array([X3[0], X3[1], X3[2], 1.0])
    x  = P @ Xh
    return x[0] / x[2], x[1] / x[2]

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_segmentation_plot(img_L, img_R, mask_L, mask_R, out_path):
    """Overlay segmentation masks on the original images for visual verification."""
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, mask, title in zip(axes,
                                     [img_L, img_R],
                                     [mask_L, mask_R],
                                     ['Left segmentation', 'Right segmentation']):
        overlay = img.copy().astype(np.float32)
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        alpha = 0.4
        overlay[mask > 0] = (1 - alpha) * overlay[mask > 0] + alpha * green[mask > 0]
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'saved segmentation plot -> {out_path}')


def save_detection_plot(result_img_L, result_img_R, out_path):
    """Show YOLO detection results for left and right images side-by-side."""
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, title in zip(axes,
                               [result_img_L, result_img_R],
                               ['Left detection', 'Right detection']):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'saved detection plot -> {out_path}')


def save_skeleton_plot(img_L, img_R, skel_L, skel_R, out_path):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, skel, title in zip(axes,
                                     [img_L, img_R],
                                     [skel_L, skel_R],
                                     ['Left view', 'Right view']):
        ax.imshow(img)
        if len(skel) > 0:
            ax.scatter(skel[:, 0], skel[:, 1],
                       c=np.arange(len(skel)), cmap='viridis', s=2)
        ax.set_title(f'{title} skeleton (color = arc order)')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'saved skeleton plot -> {out_path}')


def save_epipolar_plot(img_L, img_R, matches_L, matches_R, F, img_w, out_path):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_L); axes[0].set_title('Left'); axes[0].axis('off')
    axes[1].imshow(img_R); axes[1].set_title('Right + epipolar lines'); axes[1].axis('off')

    n_show = min(8, len(matches_L))
    if n_show > 0:
        show_idx = np.linspace(0, len(matches_L) - 1, n_show).astype(int)
        colors = plt.cm.tab10(np.arange(n_show) % 10)
        for k, idx in enumerate(show_idx):
            uL, vL = matches_L[idx]
            uR, vR = matches_R[idx]
            axes[0].scatter([uL], [vL], c=[colors[k]], s=40, edgecolors='white')
            a, b, c = F @ np.array([uL, vL, 1.0])
            x = np.array([0, img_w])
            if abs(b) > 1e-6:
                y = -(a * x + c) / b
                axes[1].plot(x, y, color=colors[k], linewidth=1)
            axes[1].scatter([uR], [vR], c=[colors[k]], s=40, edgecolors='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'saved epipolar plot -> {out_path}')


def save_reprojection_plot(img_L, img_R, skel_L_ud, skel_R_ud,
                           uL_mid, vL_mid, uR_mid, vR_mid,
                           pts3d, P1, P2, out_path):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_L); axes[0].set_title('Left: reprojected 3D points + midpoint')
    axes[1].imshow(img_R); axes[1].set_title('Right: reprojected 3D points + midpoint')
    for ax in axes:
        ax.axis('off')

    # original skeleton (undistorted) for reference
    if len(skel_L_ud):
        axes[0].scatter(skel_L_ud[:, 0], skel_L_ud[:, 1], s=1, c='cyan', alpha=0.4,
                        label='skeleton')
    if len(skel_R_ud):
        axes[1].scatter(skel_R_ud[:, 0], skel_R_ud[:, 1], s=1, c='cyan', alpha=0.4,
                        label='skeleton')

    # reproject all triangulated 3D points back onto each image
    if len(pts3d) > 0:
        Xh = np.hstack([pts3d, np.ones((len(pts3d), 1))]).T  # (4, N)
        for ax, P in [(axes[0], P1), (axes[1], P2)]:
            proj = P @ Xh           # (3, N)
            u = proj[0] / proj[2]
            v = proj[1] / proj[2]
            ax.scatter(u, v, s=4, c='lime', alpha=0.7, label='reprojected 3D pts')

    # midpoint
    axes[0].scatter([uL_mid], [vL_mid], s=200, marker='+', c='red', linewidths=3,
                    label='midpoint')
    axes[1].scatter([uR_mid], [vR_mid], s=200, marker='+', c='red', linewidths=3,
                    label='midpoint')
    for ax in axes:
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'saved reprojection plot -> {out_path}')


# def save_3d_plot(pts3d, circle, arc, out_path):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     inl  = circle['inliers']
#     outl = ~inl
#     ax.scatter(pts3d[inl, 0],  pts3d[inl, 1],  pts3d[inl, 2],
#                c='royalblue', s=8, label=f'inliers (n={int(inl.sum())})')
#     if outl.any():
#         ax.scatter(pts3d[outl, 0], pts3d[outl, 1], pts3d[outl, 2],
#                    c='lightgray', s=8, label=f'outliers (n={int(outl.sum())})')
#
#     C, r, e1, e2 = circle['center'], circle['radius'], circle['e1'], circle['e2']
#     thetas = np.linspace(0, 2 * np.pi, 200)
#     circ_pts = C[None, :] + r * (np.outer(np.cos(thetas), e1) +
#                                   np.outer(np.sin(thetas), e2))
#     ax.plot(circ_pts[:, 0], circ_pts[:, 1], circ_pts[:, 2],
#             c='orange', linewidth=1, alpha=0.7,
#             label=f'fitted circle (r={r:.2f} mm)')
#
#     M = arc['midpoint_3d']
#     ax.scatter([M[0]], [M[1]], [M[2]], c='red', s=120, marker='*',
#                label=f'3D midpoint  Z={M[2]:.2f} mm')
#
#     ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
#     ax.set_title('3D needle reconstruction (left camera frame)')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     print(f'saved 3D plot -> {out_path}')


def save_3d_plot(pts3d, arc, out_path):
    """3D scatter of triangulated needle points + chain midpoint."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pts_mm = pts3d * 1000
    ax.scatter(pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2],
               c='royalblue', s=8, alpha=0.6, label=f'triangulated pts (n={len(pts3d)})')

    M = arc['midpoint_3d'] * 1000
    ax.scatter([M[0]], [M[1]], [M[2]], c='red', s=200, marker='*',
               label=f'3D midpoint  Z={M[2]:.2f} mm')

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title('3D needle reconstruction (left camera frame)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'saved 3D plot -> {out_path}')



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Calibration ---
    K1, K2, D1, D2, R, T, IMG_W, _ = load_calibration(args.calib)
    print('Calibration loaded.')
    print('K1 =\n', K1)
    print('K2 =\n', K2)
    print('R  =\n', R)
    print('T  =\n', T.ravel(), '  (units: m)')

    # --- Load detection models (once) ---
    sys.path.insert(0, str(Path(__file__).parent))
    from detection_to_segmentation import load_yolo, load_sam3, run_pipeline

    yolo_model = load_yolo(args.weights)
    sam3_model, sam3_processor = load_sam3()

    # --- Run YOLO + SAM3 ---
    print('\n[1/7] Running YOLO + SAM3 on left image ...')
    left_result = run_pipeline(
        args.left, args.weights,
        conf=args.conf, expand=args.box_expand, fit_arc=False,
        output_path=str(output_dir),
        yolo_model=yolo_model,
        sam3_model=sam3_model, sam3_processor=sam3_processor)

    print('[1/7] Running YOLO + SAM3 on right image ...')
    right_result = run_pipeline(
        args.right, args.weights,
        conf=args.conf, expand=args.box_expand, fit_arc=False,
        output_path=str(output_dir),
        yolo_model=yolo_model,
        sam3_model=sam3_model, sam3_processor=sam3_processor)

    assert left_result is not None and right_result is not None, \
        'YOLO failed to detect the needle in one or both views'

    mask_L = left_result['masks'][0]
    mask_R = right_result['masks'][0]
    print(f'  mask_L: {mask_L.shape}, {int(mask_L.sum())} px')
    print(f'  mask_R: {mask_R.shape}, {int(mask_R.sum())} px')

    img_L = cv2.cvtColor(cv2.imread(args.left),  cv2.COLOR_BGR2RGB)
    img_R = cv2.cvtColor(cv2.imread(args.right), cv2.COLOR_BGR2RGB)

    if not args.no_vis:
        save_detection_plot(left_result['result_img'], right_result['result_img'],
                            output_dir / 'detection.png')
        save_segmentation_plot(img_L, img_R, mask_L, mask_R,
                               output_dir / 'segmentation.png')

    # --- Skeleton extraction (left only) ---
    print('\n[2/7] Extracting left skeleton; collecting right mask pixels ...')
    skel_L, *_ = extract_skeleton(mask_L)
    ys_R, xs_R = np.where(mask_R)
    mask_R_pts = np.column_stack([xs_R, ys_R]).astype(np.float32)
    print(f'  left  skeleton:   {len(skel_L):4d} ordered pixels')
    print(f'  right mask pixels:{len(mask_R_pts):4d}')

    if not args.no_vis:
        save_skeleton_plot(img_L, img_R, skel_L, mask_R_pts,
                           output_dir / 'skeleton.png')

    # --- Undistort ---
    print('\n[3/7] Undistorting points ...')
    skel_L_ud    = undistort_pixels(skel_L,    K1, D1)
    mask_R_pts_ud = undistort_pixels(mask_R_pts, K2, D2)

    # --- Epipolar matching ---
    print('\n[4/7] Epipolar matching ...')
    F = fundamental_from_KRT(K1, K2, R, T)
    matches_L, matches_R = epipolar_match(
        skel_L_ud, mask_R_pts_ud, F, max_dist_px=args.epipolar_thresh)
    print(f'  kept {len(matches_L)} / {len(skel_L_ud)} left points')

    if not args.no_vis:
        save_epipolar_plot(img_L, img_R, matches_L, matches_R, F, IMG_W,
                           output_dir / 'epipolar.png')

    # --- Triangulation ---
    print('\n[5/7] Triangulating ...')
    P1, P2 = projection_matrices(K1, K2, R, T)
    pts3d  = triangulate(P1, P2, matches_L, matches_R)
    valid  = (pts3d[:, 2] > 0) & (pts3d[:, 2] < 1.0)
    pts3d  = pts3d[valid]
    print(f'  {len(pts3d)} valid 3D points')
    print(f'  Z range: {pts3d[:, 2].min():.4f} .. {pts3d[:, 2].max():.4f} m')
    print(f'  centroid: {pts3d.mean(axis=0)}')


    # --- 3D circle fit (disabled: most points are outliers, direct chain midpoint is used instead) ---
    # print('\n[6/7] Fitting 3D circle (RANSAC) ...')
    # circle = fit_circle_3d(pts3d,
    #                        ransac_iters=args.ransac_iters,
    #                        ransac_thresh=args.ransac_thresh)
    # assert circle is not None, '3D circle fit failed — not enough valid 3D points'
    # print(f"  center  : {circle['center']}  mm")
    # print(f"  radius  : {circle['radius']:.3f} mm")
    # print(f"  normal  : {circle['normal']}")
    # print(f"  inliers : {circle['n_in']} / {circle['n_total']}")
    # print(f"  RMS res : {circle['rms_mm']:.3f} mm")
    # if circle['rms_mm'] > 1.0:
    #     print('  WARNING: RMS residual > 1 mm — check calibration or matches')
    # if circle['n_in'] / circle['n_total'] < 0.5:
    #     print('  WARNING: inlier ratio < 50% — check skeleton/matching')

    # --- Arc midpoint via nearest-neighbour chain (no circle fit) ---
    print('\n[6/7] Computing 3D arc midpoint (nearest-neighbour chain) ...')
    arc_match = arc_midpoint_from_matches(pts3d)

    print('\n=== RESULT ===')
    print(f"3D MIDPOINT: {arc_match['midpoint_3d']}  m")
    print(f"  depth Z    : {arc_match['midpoint_3d'][2]:.4f} m")
    print(f"  n_pts      : {arc_match['n_pts']}")
    print(f"  endpoint 1 : {arc_match['endpoint1_3d']}")
    print(f"  endpoint 2 : {arc_match['endpoint2_3d']}")

    # --- Visualisation ---
    if not args.no_vis:
        uL_mid, vL_mid = project(P1, arc_match['midpoint_3d'])
        uR_mid, vR_mid = project(P2, arc_match['midpoint_3d'])
        save_reprojection_plot(img_L, img_R, skel_L_ud, mask_R_pts_ud,
                               uL_mid, vL_mid, uR_mid, vR_mid,
                               pts3d, P1, P2,
                               output_dir / 'reprojection.png')
        save_3d_plot(pts3d, arc_match,
                     output_dir / '3d_reconstruction.png')
    # --- Save result ---
    out = {
        'midpoint_m':   arc_match['midpoint_3d'],
        'endpoint1_m':  arc_match['endpoint1_3d'],
        'endpoint2_m':  arc_match['endpoint2_3d'],
        'n_pts':        arc_match['n_pts'],
    }
    npz_path = output_dir / 'needle_3d.npz'
    np.savez(npz_path, **out)
    print(f'\nsaved -> {npz_path}')
    for k, v in out.items():
        print(f'  {k:14s}: {v}')


if __name__ == '__main__':
    main()
