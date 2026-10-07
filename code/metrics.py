"""Standalone 3D reconstruction metrics for ISAR point clouds."""

import json
from pathlib import Path

import numpy as np
import torch


def _robust_extent(points, lower_percentile=1.0, upper_percentile=99.0):
    """Return percentile-robust axis extents for an ``Nx3`` point cloud."""
    lower = np.percentile(points, lower_percentile, axis=0)
    upper = np.percentile(points, upper_percentile, axis=0)
    return np.maximum(upper - lower, 0.0)


def _knn_components(points, radius, neighbors=8,
                    min_component_fraction=0.005):
    """Approximate radius-graph components without constructing an NxN graph."""
    from scipy.spatial import cKDTree

    n_points = int(len(points))
    k = min(max(int(neighbors), 1) + 1, n_points)
    distances, indices = cKDTree(points).query(points, k=k)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    parent = np.arange(n_points, dtype=np.int64)
    component_size = np.ones(n_points, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root == right_root:
            return
        if component_size[left_root] < component_size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_size[left_root] += component_size[right_root]

    within_radius = distances <= float(radius)
    for point_i in range(n_points):
        for neighbor_i in indices[point_i, within_radius[point_i]]:
            union(point_i, neighbor_i)

    roots = np.fromiter(
        (find(i) for i in range(n_points)),
        dtype=np.int64,
        count=n_points,
    )
    _, labels = np.unique(roots, return_inverse=True)
    sizes = np.bincount(labels)
    threshold = max(
        10, int(np.ceil(float(min_component_fraction) * n_points)))
    significant = np.flatnonzero(sizes >= threshold)
    if significant.size == 0:
        significant = np.asarray([int(np.argmax(sizes))], dtype=np.int64)
    neighbor_counts = within_radius.sum(axis=1)
    return {
        "labels": labels,
        "sizes": sizes,
        "significant": significant,
        "threshold": threshold,
        "isolated_ratio": float((neighbor_counts == 0).mean()),
        "sparse_ratio": float((neighbor_counts < 3).mean()),
        "largest_component_ratio": float(sizes.max() / n_points),
        "small_component_ratio": float(
            sizes[sizes < threshold].sum() / n_points),
    }


def _shape_structure_metrics(pred, gt, d_gt_pred, diag,
                             radius_frac=0.02, neighbors=8,
                             min_component_fraction=0.005,
                             lower_percentile=1.0,
                             upper_percentile=99.0):
    """Annotation-free size, fragmentation, and component-proxy metrics."""
    eps = 1e-9
    gt_extent = _robust_extent(
        gt, lower_percentile, upper_percentile)
    pred_extent = _robust_extent(
        pred, lower_percentile, upper_percentile)
    aabb_rel = (pred_extent - gt_extent) / np.maximum(gt_extent, eps)

    # Evaluate oriented size in the GT PCA frame.  Both clouds use the same
    # frame so a rotated prediction cannot hide a compressed dimension.
    gt_center = gt.mean(axis=0)
    pred_center = pred.mean(axis=0)
    covariance = np.cov(gt - gt_center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    gt_pca_extent = _robust_extent(
        (gt - gt_center) @ basis, lower_percentile, upper_percentile)
    pred_pca_extent = _robust_extent(
        (pred - gt_center) @ basis, lower_percentile, upper_percentile)
    pca_rel = (
        (pred_pca_extent - gt_pca_extent)
        / np.maximum(gt_pca_extent, eps))

    radius = float(radius_frac) * float(diag)
    if radius <= 0:
        raise ValueError("structure_radius_frac must be positive")
    pred_graph = _knn_components(
        pred, radius, neighbors, min_component_fraction)
    gt_graph = _knn_components(
        gt, radius, neighbors, min_component_fraction)

    component_rows = []
    component_means = []
    component_recalls = []
    recovery_threshold = 0.02 * float(diag)
    for component_id in gt_graph["significant"]:
        mask = gt_graph["labels"] == component_id
        component_distance = d_gt_pred[mask]
        mean_distance = float(component_distance.mean())
        recall = float((component_distance <= recovery_threshold).mean())
        component_means.append(mean_distance)
        component_recalls.append(recall)
        component_rows.append({
            "component": int(component_id),
            "n_gt": int(mask.sum()),
            "fraction": float(mask.mean()),
            "Completeness": mean_distance,
            "R_2pct": recall,
        })

    pred_count = int(len(pred_graph["significant"]))
    gt_count = int(len(gt_graph["significant"]))
    centroid_error = float(np.linalg.norm(pred_center - gt_center))
    result = {
        "HD95": None,  # populated by reconstruction_metrics_3d
        "Centroid_error": centroid_error,
        "Centroid_error_norm": centroid_error / float(diag),
        "GT_AABB_extent_x": float(gt_extent[0]),
        "GT_AABB_extent_y": float(gt_extent[1]),
        "GT_AABB_extent_z": float(gt_extent[2]),
        "Pred_AABB_extent_x": float(pred_extent[0]),
        "Pred_AABB_extent_y": float(pred_extent[1]),
        "Pred_AABB_extent_z": float(pred_extent[2]),
        "AABB_extent_rel_error_x": float(aabb_rel[0]),
        "AABB_extent_rel_error_y": float(aabb_rel[1]),
        "AABB_extent_rel_error_z": float(aabb_rel[2]),
        "AABB_extent_abs_rel_error_mean": float(np.abs(aabb_rel).mean()),
        "AABB_compression_mean": float(np.maximum(-aabb_rel, 0.0).mean()),
        "AABB_expansion_mean": float(np.maximum(aabb_rel, 0.0).mean()),
        "AABB_diag_rel_error": float(
            np.linalg.norm(pred_extent)
            / max(float(np.linalg.norm(gt_extent)), eps) - 1.0),
        "AABB_volume_ratio": float(
            np.prod(pred_extent) / max(float(np.prod(gt_extent)), eps)),
        "GT_PCA_extent_1": float(gt_pca_extent[0]),
        "GT_PCA_extent_2": float(gt_pca_extent[1]),
        "GT_PCA_extent_3": float(gt_pca_extent[2]),
        "Pred_PCA_extent_1": float(pred_pca_extent[0]),
        "Pred_PCA_extent_2": float(pred_pca_extent[1]),
        "Pred_PCA_extent_3": float(pred_pca_extent[2]),
        "PCA_extent_rel_error_1": float(pca_rel[0]),
        "PCA_extent_rel_error_2": float(pca_rel[1]),
        "PCA_extent_rel_error_3": float(pca_rel[2]),
        "PCA_extent_abs_rel_error_mean": float(np.abs(pca_rel).mean()),
        "PCA_compression_mean": float(np.maximum(-pca_rel, 0.0).mean()),
        "PCA_expansion_mean": float(np.maximum(pca_rel, 0.0).mean()),
        "PCA_diag_rel_error": float(
            np.linalg.norm(pred_pca_extent)
            / max(float(np.linalg.norm(gt_pca_extent)), eps) - 1.0),
        "PCA_volume_ratio": float(
            np.prod(pred_pca_extent)
            / max(float(np.prod(gt_pca_extent)), eps)),
        "structure_radius": radius,
        "structure_radius_frac": float(radius_frac),
        "structure_neighbors": int(neighbors),
        "component_min_points": int(gt_graph["threshold"]),
        "Pred_isolated_ratio": pred_graph["isolated_ratio"],
        "GT_isolated_ratio": gt_graph["isolated_ratio"],
        "Isolated_ratio_excess": max(
            0.0, pred_graph["isolated_ratio"] - gt_graph["isolated_ratio"]),
        "Pred_sparse_ratio": pred_graph["sparse_ratio"],
        "GT_sparse_ratio": gt_graph["sparse_ratio"],
        "Sparse_ratio_excess": max(
            0.0, pred_graph["sparse_ratio"] - gt_graph["sparse_ratio"]),
        "Pred_component_count": pred_count,
        "GT_component_count": gt_count,
        "Component_count_gap": pred_count - gt_count,
        "Component_count_excess": max(0, pred_count - gt_count),
        "Pred_largest_component_ratio":
            pred_graph["largest_component_ratio"],
        "GT_largest_component_ratio": gt_graph["largest_component_ratio"],
        "Largest_component_ratio_gap": max(
            0.0,
            gt_graph["largest_component_ratio"]
            - pred_graph["largest_component_ratio"]),
        "Pred_small_component_ratio": pred_graph["small_component_ratio"],
        "GT_small_component_ratio": gt_graph["small_component_ratio"],
        "Small_component_ratio_excess": max(
            0.0,
            pred_graph["small_component_ratio"]
            - gt_graph["small_component_ratio"]),
        "Component_Completeness_macro": float(np.mean(component_means)),
        "Component_Completeness_worst": float(np.max(component_means)),
        "Component_R_2pct_macro": float(np.mean(component_recalls)),
        "Component_recovery_rate_2pct": float(
            np.mean(np.asarray(component_recalls) >= 0.8)),
        "per_component": component_rows,
    }
    return result


def infer_meters_per_world_unit(scene_dir):
    """Infer the physical scale for a processed satellite scene."""
    scene_dir = Path(scene_dir)
    meta_path = scene_dir / "nas_full_points_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scale_to_world = float(meta.get("scale_to_world", 0.0))
        if scale_to_world > 0:
            return 1.0 / scale_to_world

    camera_path = scene_dir / "cameras.json"
    if camera_path.is_file():
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        ortho_scale = float(camera.get("ortho_scale", 0.0))
        range_resolution_m = float(camera.get("range_resolution_m", 0.0))
        if ortho_scale > 0 and range_resolution_m > 0:
            return ortho_scale * range_resolution_m
    return None


@torch.no_grad()
def build_gt_pointcloud(scene_dir, frames, T_w2c_ref, device,
                        n_frames_use=80, voxel=0.01):
    """Aggregate per-frame camera-space ``xyz_gt`` into canonical space."""
    xyz_dir = Path(scene_dir) / "xyz_gt"
    if not xyz_dir.is_dir():
        return None
    n_frames = len(frames)
    step = max(1, n_frames // max(n_frames_use, 1))
    points = []
    for i in range(0, n_frames, step):
        path = xyz_dir / f"{i:04d}.npy"
        if not path.exists():
            continue
        xyz = np.load(str(path))
        valid = ~np.isnan(xyz).any(-1)
        xyz = xyz[valid]
        if xyz.shape[0] == 0:
            continue
        c2w = np.asarray(frames[i]["c2w"], dtype=np.float64)
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        points.append((c2w @ xyz_h.T).T[:, :3])
    if not points:
        return None

    world = np.concatenate(points, axis=0)
    T = (T_w2c_ref.detach().cpu().numpy() if torch.is_tensor(T_w2c_ref)
         else np.asarray(T_w2c_ref)).astype(np.float64)
    world_h = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1)
    canonical = (T @ world_h.T).T[:, :3]
    if voxel and voxel > 0:
        _, unique = np.unique(
            np.floor(canonical / voxel).astype(np.int64),
            axis=0,
            return_index=True,
        )
        canonical = canonical[unique]
    return torch.from_numpy(canonical).float().to(device)


@torch.no_grad()
def build_full_nas_pointcloud(scene_dir, T_w2c_ref, device):
    """Load the complete NAS surface cloud and convert it to canonical space."""
    path = Path(scene_dir) / "nas_full_points.npy"
    if not path.is_file():
        return None

    points = np.load(str(path)).astype(np.float64)
    valid = np.isfinite(points).all(axis=-1)
    points = points[valid]
    if points.shape[0] == 0:
        return None

    transform = (T_w2c_ref.detach().cpu().numpy()
                 if torch.is_tensor(T_w2c_ref)
                 else np.asarray(T_w2c_ref)).astype(np.float64)
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    canonical = (transform @ points_h.T).T[:, :3]
    return torch.from_numpy(canonical).float().to(device)


@torch.no_grad()
def reconstruction_metrics_3d(pred_xyz, gt_xyz,
                              tau_fracs=(0.01, 0.02, 0.05, 0.1),
                              meters_per_unit=None,
                              tau_meters=(0.1, 0.3, 0.5, 0.7),
                              pred_confidence=None,
                              surface_normal_neighbors=16,
                              confidence_bins=10,
                              reference_diag=None,
                              compute_structure_metrics=False,
                              structure_radius_frac=0.02,
                              structure_neighbors=8,
                              min_component_fraction=0.005,
                              robust_extent_percentiles=(1.0, 99.0)):
    """Compute squared CD, F-scores, and paper-compatible surface metrics.

    Accuracy is prediction-to-GT mean Euclidean distance; Completeness is
    GT-to-prediction; Overall is their mean.  The legacy ``R2G/G2R/CD``
    fields remain mean *squared* world-unit distances for backward
    compatibility.  ``CD_world2`` is the explicit name for that raw value;
    ``CD_m2`` is the paper-facing metric when physical scale is available.

    The v17 paper diagnostics additionally report prediction/GT distance
    percentiles, local point-to-plane surface residuals, fixed-metre outlier
    rates, and optional confidence calibration/ranking metrics.

    ``compute_structure_metrics`` adds annotation-free robust size/compression,
    scatter/connectivity, and geometric component-proxy diagnostics.  These
    are intended for complete-shape GT, rather than visible/hidden subsets.

    When ``meters_per_unit`` is supplied, the result also contains explicitly
    named, unsquared ``R_to_G_mean_m/G_to_R_mean_m/CD_mean_m`` values and
    precision/recall/F1 at the fixed metric thresholds used by ISAR-GS.
    """
    from scipy.spatial import cKDTree

    pred = pred_xyz.detach().cpu().numpy() if torch.is_tensor(pred_xyz) else np.asarray(pred_xyz)
    gt = gt_xyz.detach().cpu().numpy() if torch.is_tensor(gt_xyz) else np.asarray(gt_xyz)
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    if len(pred) < 10 or len(gt) < 10:
        return None

    diag = (
        float(reference_diag)
        if reference_diag is not None
        else float(np.linalg.norm(gt.max(0) - gt.min(0)) + 1e-9))
    if diag <= 0:
        raise ValueError("reference_diag must be positive")
    gt_tree = cKDTree(gt)
    d_pred_gt, nearest_gt_idx = gt_tree.query(pred)
    d_gt_pred, _ = cKDTree(pred).query(gt)
    r2g = float((d_pred_gt ** 2).mean())
    g2r = float((d_gt_pred ** 2).mean())
    cd = 0.5 * (r2g + g2r)
    accuracy = float(d_pred_gt.mean())
    completeness = float(d_gt_pred.mean())
    overall = 0.5 * (accuracy + completeness)

    percentiles = (50, 90, 95, 99)
    r_to_g_percentiles = {
        f"R_to_G_p{p}": float(np.percentile(d_pred_gt, p))
        for p in percentiles
    }
    g_to_r_percentiles = {
        f"G_to_R_p{p}": float(np.percentile(d_gt_pred, p))
        for p in percentiles
    }
    hd95 = max(r_to_g_percentiles["R_to_G_p95"],
               g_to_r_percentiles["G_to_R_p95"])

    # Local GT PCA converts visual "thick panels" into a reproducible
    # point-to-plane residual.  It is unsigned because NAS point normals are
    # not oriented consistently.
    p2plane = np.zeros(len(pred), dtype=np.float64)
    normal_k = min(max(int(surface_normal_neighbors), 3), len(gt))
    if normal_k >= 3:
        nearest_gt = gt[nearest_gt_idx]
        _, normal_neighbor_idx = gt_tree.query(nearest_gt, k=normal_k)
        local = gt[normal_neighbor_idx]
        centered = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum(
            "nki,nkj->nij", centered, centered) / float(normal_k)
        normal = np.linalg.eigh(covariance)[1][..., 0]
        p2plane = np.abs(np.einsum(
            "ni,ni->n", pred - nearest_gt, normal))

    result = {
        "CD_world2": cd,
        # Deprecated compatibility alias. New tables and terminal summaries
        # must use CD_m2 (or CD_world2 for internal scale-free debugging).
        "CD": cd,
        "CD_norm": cd / (diag ** 2),
        "R2G": r2g,
        "G2R": g2r,
        "R_to_G_mean": accuracy,
        "G_to_R_mean": completeness,
        "CD_mean": overall,
        "Accuracy": accuracy,
        "Completeness": completeness,
        "Overall": overall,
        "n_pred": int(len(pred)),
        "n_gt": int(len(gt)),
        "gt_diag": diag,
        "HD95": hd95,
        "HD95_norm": hd95 / diag,
        **r_to_g_percentiles,
        **g_to_r_percentiles,
        "P2Plane_mean": float(p2plane.mean()),
        "P2Plane_p90": float(np.percentile(p2plane, 90)),
        "P2Plane_p95": float(np.percentile(p2plane, 95)),
        "per_tau": [],
        "per_tau_m": [],
    }
    if meters_per_unit is not None:
        meters_per_unit = float(meters_per_unit)
        if meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be positive")
        result.update({
            "meters_per_unit": meters_per_unit,
            "Accuracy_m": accuracy * meters_per_unit,
            "Completeness_m": completeness * meters_per_unit,
            "Overall_m": overall * meters_per_unit,
            "R_to_G_mean_m": accuracy * meters_per_unit,
            "G_to_R_mean_m": completeness * meters_per_unit,
            "CD_mean_m": overall * meters_per_unit,
            "CD_m2": cd * (meters_per_unit ** 2),
            "HD95_m": hd95 * meters_per_unit,
            **{
                f"{key}_m": value * meters_per_unit
                for key, value in {
                    **r_to_g_percentiles,
                    **g_to_r_percentiles,
                }.items()
            },
            "P2Plane_mean_m": float(p2plane.mean()) * meters_per_unit,
            "P2Plane_p90_m": float(np.percentile(p2plane, 90)) * meters_per_unit,
            "P2Plane_p95_m": float(np.percentile(p2plane, 95)) * meters_per_unit,
        })

    if compute_structure_metrics:
        lower_percentile, upper_percentile = robust_extent_percentiles
        if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
            raise ValueError(
                "robust_extent_percentiles must satisfy "
                "0 <= lower < upper <= 100")
        structure = _shape_structure_metrics(
            pred, gt, d_gt_pred, diag,
            radius_frac=structure_radius_frac,
            neighbors=structure_neighbors,
            min_component_fraction=min_component_fraction,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        structure["HD95"] = hd95
        if meters_per_unit is not None:
            structure["Centroid_error_m"] = (
                structure["Centroid_error"] * meters_per_unit)
            structure["Component_Completeness_macro_m"] = (
                structure["Component_Completeness_macro"]
                * meters_per_unit)
            structure["Component_Completeness_worst_m"] = (
                structure["Component_Completeness_worst"]
                * meters_per_unit)
            structure["structure_radius_m"] = (
                structure["structure_radius"] * meters_per_unit)
            for family in ("AABB", "PCA"):
                for owner in ("GT", "Pred"):
                    axes = ("x", "y", "z") if family == "AABB" else (1, 2, 3)
                    for axis in axes:
                        key = f"{owner}_{family}_extent_{axis}"
                        structure[f"{key}_m"] = (
                            structure[key] * meters_per_unit)
            for component in structure["per_component"]:
                component["Completeness_m"] = (
                    component["Completeness"] * meters_per_unit)
        result.update(structure)

    for frac in tau_fracs:
        tau = frac * diag
        precision = float((d_pred_gt < tau).mean())
        recall = float((d_gt_pred < tau).mean())
        result["per_tau"].append({
            "tau_frac": frac,
            "tau": tau,
            "P": precision,
            "R": recall,
            "F1": 2 * precision * recall / (precision + recall + 1e-9),
        })

    if meters_per_unit is not None:
        for tau_m in tau_meters:
            tau_m = float(tau_m)
            if tau_m <= 0:
                raise ValueError("tau_meters values must be positive")
            tau_world = tau_m / meters_per_unit
            precision = float((d_pred_gt <= tau_world).mean())
            recall = float((d_gt_pred <= tau_world).mean())
            result["per_tau_m"].append({
                "tau_m": tau_m,
                "tau_world": tau_world,
                "P": precision,
                "R": recall,
                "F1": 2 * precision * recall / (precision + recall + 1e-9),
                "R_to_G_outlier_rate": float((d_pred_gt > tau_world).mean()),
            })

    if pred_confidence is not None:
        confidence = (pred_confidence.detach().cpu().numpy()
                      if torch.is_tensor(pred_confidence)
                      else np.asarray(pred_confidence))
        confidence = confidence.astype(np.float64).reshape(-1)
        if confidence.shape[0] != pred.shape[0]:
            raise ValueError(
                "pred_confidence length must equal the prediction count")
        confidence = np.clip(confidence, 0.0, 1.0)
        correctness = (d_pred_gt <= 0.02 * diag).astype(np.float64)
        if np.std(confidence) > 0 and np.std(d_pred_gt) > 0:
            pearson = float(np.corrcoef(confidence, -d_pred_gt)[0, 1])
        else:
            pearson = 0.0
        from scipy.stats import spearmanr
        spearman_result = spearmanr(confidence, -d_pred_gt)
        spearman = float(spearman_result.statistic)
        if not np.isfinite(spearman):
            spearman = 0.0
        n_bins = max(int(confidence_bins), 2)
        ece = 0.0
        for lower, upper in zip(
                np.linspace(0.0, 1.0, n_bins + 1)[:-1],
                np.linspace(0.0, 1.0, n_bins + 1)[1:]):
            mask = ((confidence >= lower) &
                    (confidence < upper if upper < 1.0 else confidence <= upper))
            if mask.any():
                ece += (float(mask.mean())
                        * abs(float(confidence[mask].mean())
                              - float(correctness[mask].mean())))
        top_k = max(1, int(round(0.2 * len(confidence))))
        top_idx = np.argsort(-confidence)[:top_k]
        confidence_metrics = {
            "confidence_correctness_pearson": pearson,
            "confidence_correctness_spearman": spearman,
            "confidence_ece_2pct": float(ece),
            "confidence_brier_2pct": float(
                np.mean((confidence - correctness) ** 2)),
            "confidence_top20_R_to_G_mean": float(d_pred_gt[top_idx].mean()),
            "confidence_top20_P_2pct": float(correctness[top_idx].mean()),
            "confidence_mean": float(confidence.mean()),
        }
        if meters_per_unit is not None:
            confidence_metrics["confidence_top20_R_to_G_mean_m"] = (
                confidence_metrics["confidence_top20_R_to_G_mean"]
                * meters_per_unit)
        result.update(confidence_metrics)
    return result
