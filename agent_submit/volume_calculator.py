"""3D back-projection and OBB-based volume estimation.

Uses PCA for oriented bounding box computation (no Open3D dependency).
Falls back to Open3D if available.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PipelineConfig

# Try to import Open3D, but it's optional
# OSError can occur on headless servers missing libX11
try:
    import open3d as o3d
    HAS_OPEN3D = True
except (ImportError, OSError):
    HAS_OPEN3D = False


class VolumeCalculator:
    """Sample depth points from masks, back-project to 3D, compute OBB volume."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def sample_points_from_mask(
        self,
        mask: np.ndarray,
        n_points: int = 9,
    ) -> List[Tuple[int, int]]:
        """Sample strategic points from a binary mask.

        Strategy: centroid + 4 extremal + 4 grid (25%/75%) + random fill.
        Returns list of (x, y) pixel coordinates.
        """
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return []

        points = set()

        # 1. Centroid
        cx, cy = int(xs.mean()), int(ys.mean())
        if mask[cy, cx]:
            points.add((cx, cy))
        else:
            dists = (xs - cx) ** 2 + (ys - cy) ** 2
            nearest = dists.argmin()
            points.add((int(xs[nearest]), int(ys[nearest])))

        # 2. Extremal points (top, bottom, left, right)
        points.add((int(xs[ys.argmin()]), int(ys.min())))   # top
        points.add((int(xs[ys.argmax()]), int(ys.max())))   # bottom
        points.add((int(xs.min()), int(ys[xs.argmin()])))   # left
        points.add((int(xs.max()), int(ys[xs.argmax()])))   # right

        # 3. Grid points at 25%/75% positions
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        y_range = y_max - y_min
        x_range = x_max - x_min

        for yf in [0.25, 0.75]:
            for xf in [0.25, 0.75]:
                target_y = int(y_min + y_range * yf)
                target_x = int(x_min + x_range * xf)
                dists = (xs - target_x) ** 2 + (ys - target_y) ** 2
                nearest = dists.argmin()
                points.add((int(xs[nearest]), int(ys[nearest])))

        # 4. Fill remaining with random mask points
        points_list = list(points)
        rng = np.random.default_rng(42)
        attempts = 0
        while len(points_list) < n_points and attempts < 100:
            idx = rng.integers(len(xs))
            p = (int(xs[idx]), int(ys[idx]))
            if p not in points:
                points.add(p)
                points_list.append(p)
            attempts += 1

        return points_list[:n_points]

    def backproject_to_3d(
        self,
        points_2d,
        depths,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """Back-project 2D pixel coords + metric depths to 3D points.

        Accepts either list-of-tuples or numpy arrays for fast vectorized operation.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

        Returns: (N, 3) array of 3D points. Invalid depths are filtered out.
        """
        pts = np.asarray(points_2d)          # (N, 2) — columns: u, v
        z = np.asarray(depths, dtype=float)  # (N,)

        valid = np.isfinite(z) & (z > 0) & (z <= 100.0)
        if not valid.any():
            return np.empty((0, 3))

        u = pts[valid, 0].astype(float)
        v = pts[valid, 1].astype(float)
        z = z[valid]

        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        return np.stack([X, Y, z], axis=1)

    @staticmethod
    def filter_points_by_depth(
        points_3d: np.ndarray,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
    ) -> np.ndarray:
        """Remove 3D points with extreme depth values based on the Z axis only.

        This keeps the mask silhouette intact in X/Y while suppressing
        background leakage and depth spikes near object boundaries.
        """
        if len(points_3d) < 4:
            return points_3d

        z_vals = points_3d[:, 2]
        lo, hi = np.percentile(z_vals, [lower_pct, upper_pct])
        mask = (z_vals >= lo) & (z_vals <= hi)
        filtered = points_3d[mask]
        return filtered if len(filtered) >= 4 else points_3d

    def filter_points_robust(
        self,
        points_3d: np.ndarray,
        lower_pct: float = 1.0,
        upper_pct: float = 99.0,
    ) -> np.ndarray:
        """Remove 3D outliers using depth percentiles plus robust XYZ filtering."""
        if len(points_3d) < 8:
            return points_3d

        depth_filtered = self.filter_points_by_depth(
            points_3d, lower_pct=lower_pct, upper_pct=upper_pct
        )
        if len(depth_filtered) < 8:
            return depth_filtered

        median = np.median(depth_filtered, axis=0)
        mad = np.median(np.abs(depth_filtered - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-4)
        zscore = np.abs((depth_filtered - median) / scale)

        xy_score = np.sqrt(zscore[:, 0] ** 2 + zscore[:, 1] ** 2)
        xy_keep = xy_score <= self.config.robust_point_filter_mad_threshold * np.sqrt(2.0)
        z_keep = zscore[:, 2] <= self.config.robust_point_filter_mad_threshold
        filtered = depth_filtered[xy_keep & z_keep]

        min_keep = max(4, int(len(points_3d) * self.config.robust_point_filter_min_keep_ratio))
        return filtered if len(filtered) >= min_keep else depth_filtered

    def compute_obb_volume(
        self,
        points_3d: np.ndarray,
        object_category: str = "",
        mode: str = "obb",
    ) -> Dict:
        """Compute oriented bounding box volume from 3D points.

        Uses Open3D if available, otherwise falls back to PCA.
        Applies category-aware depth correction for single-view limitation.
        """
        if len(points_3d) < 4:
            return {
                "volume_m3": 0.0,
                "volume_cm3": 0.0,
                "dimensions_m": [0.0, 0.0, 0.0],
                "correction_applied": 1.0,
                "error": f"insufficient 3D points ({len(points_3d)})",
            }

        correction = self.config.depth_correction_factors.get(
            object_category.lower(),
            self.config.default_depth_correction,
        )

        if mode == "camera_aligned":
            min_xyz, max_xyz = self._robust_axis_aligned_bounds(points_3d)
            dims = np.maximum(max_xyz - min_xyz, 0.01)
            dims[2] *= correction
        elif mode == "hybrid":
            min_xyz, max_xyz = self._robust_axis_aligned_bounds(points_3d)
            camera_dims = np.maximum(max_xyz - min_xyz, 0.01)

            if HAS_OPEN3D:
                try:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_3d)
                    obb = pcd.get_oriented_bounding_box()
                    obb_dims = np.sort(np.array(obb.extent))
                except Exception:
                    obb_dims = np.sort(self._pca_extents(points_3d))
            else:
                obb_dims = np.sort(self._pca_extents(points_3d))

            # Hybrid for upright indoor appliances:
            # width  <- max(camera X span, OBB middle span)
            # height <- max(camera Y span, OBB largest span)
            # depth  <- max(camera Z span, OBB smallest span)
            depth_dim = max(float(camera_dims[2]), float(obb_dims[0]))
            width_dim = max(float(camera_dims[0]), float(obb_dims[1]))
            height_dim = max(float(camera_dims[1]), float(obb_dims[2]))
            dims = np.array([depth_dim, width_dim, height_dim], dtype=np.float32)
            dims[0] *= correction
        else:
            # Compute OBB extents
            if HAS_OPEN3D:
                try:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_3d)
                    obb = pcd.get_oriented_bounding_box()
                    extent = np.array(obb.extent)
                except Exception:
                    extent = self._pca_extents(points_3d)
            else:
                extent = self._pca_extents(points_3d)

            # Sort dimensions: smallest first
            dims = np.sort(extent)
            # Apply depth correction to the smallest dimension
            dims[0] *= correction

        volume = float(np.prod(dims))
        return {
            "volume_m3": volume,
            "volume_cm3": volume * 1e6,
            "dimensions_m": dims.tolist(),
            "correction_applied": correction,
            "estimation_mode": mode,
        }

    def compute_obb_geometry(
        self,
        points_3d: np.ndarray,
        object_category: str = "",
        robust: bool = False,
    ) -> Optional[Dict]:
        """Return AABB geometry aligned to camera axes for stable visualization.

        Uses camera-axis-aligned bounding box (X=left/right, Y=up/down, Z=depth)
        so that the 2D projection closely matches the segmentation mask bounds.
        Depth correction is applied to the Z (depth) axis.

        Returns dict with keys: center (3,), extents (3,), corners (8,3), points_3d.
        """
        if len(points_3d) < 4:
            return None

        correction = self.config.depth_correction_factors.get(
            object_category.lower(), self.config.default_depth_correction
        )

        # Axis-aligned extents in camera space
        if robust:
            min_xyz, max_xyz = self._robust_axis_aligned_bounds(points_3d)
        else:
            min_xyz = points_3d.min(axis=0)
            max_xyz = points_3d.max(axis=0)
        extents = np.maximum(max_xyz - min_xyz, 0.01)  # (X, Y, Z)

        # Apply depth correction to Z (camera depth axis)
        extents[2] *= correction

        center = (min_xyz + max_xyz) / 2.0
        # Re-center Z after correction
        center[2] = points_3d[:, 2].mean()

        half = extents / 2.0
        corners_3d = np.array([
            [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] + half[2]],
        ])

        return {
            "center": center,
            "extents": extents,
            "corners": corners_3d,
            "points_3d": points_3d,
        }

    @staticmethod
    def compute_marker_scale_factor(
        marker_points_3d: np.ndarray,
        known_width_mm: float,
        known_height_mm: float,
        known_depth_mm: Optional[float] = None,
    ) -> float:
        """Compute scale correction factor from a known-size reference marker.

        Compares estimated OBB dimensions against the marker's real-world size.
        Returns a multiplicative factor to apply to all 3D coordinates.

        For flat markers (CreditCard, A4), pass only width/height — the thinnest
        PCA axis is ignored. For 3D markers (e.g., Microwave), also pass
        ``known_depth_mm`` so all three PCA extents are matched to known dims.
        """
        if len(marker_points_3d) < 4:
            return 1.0

        # PCA to get principal extents
        centered = marker_points_3d - marker_points_3d.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        projected = centered @ eigenvectors
        extents = projected.max(axis=0) - projected.min(axis=0)
        extents_sorted = np.sort(extents)[::-1]  # descending

        if known_depth_mm is not None:
            # 3D marker: match all three extents to known dims (descending)
            known_sorted = sorted(
                [known_width_mm, known_height_mm, known_depth_mm], reverse=True
            )
            scales = []
            for i in range(3):
                est_mm = extents_sorted[i] * 1000.0
                if est_mm > 1e-6:
                    scales.append(known_sorted[i] / est_mm)
            return float(np.mean(scales)) if scales else 1.0

        # Flat marker: ignore thinnest axis, use top-2 extents
        est_w_mm = extents_sorted[0] * 1000  # m → mm
        est_h_mm = extents_sorted[1] * 1000

        known_sorted = sorted([known_width_mm, known_height_mm], reverse=True)
        scale_w = known_sorted[0] / est_w_mm if est_w_mm > 0 else 1.0
        scale_h = known_sorted[1] / est_h_mm if est_h_mm > 0 else 1.0
        scale_factor = (scale_w + scale_h) / 2.0

        return float(scale_factor)

    def aggregate_marker_scale_factors(self, scales: List[float]) -> float:
        """Aggregate multiple marker scale estimates robustly."""
        valid = np.asarray(
            [
                s for s in scales
                if np.isfinite(s)
                and self.config.marker_scale_min <= s <= self.config.marker_scale_max
            ],
            dtype=np.float32,
        )
        if len(valid) == 0:
            return 1.0
        if len(valid) == 1:
            return float(valid[0])

        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        if mad > 1e-6:
            robust = valid[np.abs(valid - median) <= 2.5 * 1.4826 * mad]
            if len(robust) > 0:
                valid = robust
        return float(np.median(valid))

    def _robust_axis_aligned_bounds(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lower = self.config.robust_extent_lower_pct
        upper = self.config.robust_extent_upper_pct
        min_xyz = np.percentile(points_3d, lower, axis=0)
        max_xyz = np.percentile(points_3d, upper, axis=0)
        return min_xyz, max_xyz

    @staticmethod
    def _pca_extents(points_3d: np.ndarray) -> np.ndarray:
        """PCA-based oriented bounding box extent estimation.

        Projects points onto principal axes and computes extent along each axis.
        """
        centered = points_3d - points_3d.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Project to principal axes
        projected = centered @ eigenvectors
        extents = projected.max(axis=0) - projected.min(axis=0)
        return np.maximum(extents, 0.01)  # Floor at 1cm
