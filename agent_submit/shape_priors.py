"""Automatic box-geometry estimators for volume refinement."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config import PipelineConfig


@dataclass
class ShapePriorEstimate:
    applied: bool
    mode: str
    dimensions_m: list[float]
    volume_m3: float
    volume_cm3: float
    correction_applied: float
    obb_geometry: Optional[Dict]
    metadata: Dict


class AutomaticShapePriorEstimator:
    """Estimate box-like dimensions from observed geometry only.

    This module intentionally avoids category-specific hand-tuned ratios.
    It uses only the observed mask/3D/depth cues from the current input.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def estimate(
        self,
        object_category: str,
        mask: np.ndarray,
        box: np.ndarray,
        points_3d: np.ndarray,
        intrinsics: Dict[str, float],
        depth_map: Optional[np.ndarray],
        video_mode: bool,
    ) -> Optional[ShapePriorEstimate]:
        if not video_mode or len(points_3d) < 16:
            return None

        rectangularity = self._rectangularity(mask, box)
        if not self._should_use_box_prior(rectangularity, points_3d):
            return None

        dims_xyz, metadata = self._estimate_box_dimensions(
            mask, box, points_3d, intrinsics, depth_map
        )
        correction = 1.0
        dims_xyz[2] *= correction

        dims_sorted = np.sort(dims_xyz.astype(np.float32))
        volume = float(np.prod(dims_sorted))
        geometry = self._build_camera_aligned_geometry(points_3d, dims_xyz)

        return ShapePriorEstimate(
            applied=True,
            mode="box_geometry",
            dimensions_m=dims_sorted.tolist(),
            volume_m3=volume,
            volume_cm3=volume * 1e6,
            correction_applied=correction,
            obb_geometry=geometry,
            metadata=metadata,
        )

    def _should_use_box_prior(
        self,
        rectangularity: float,
        points_3d: np.ndarray,
    ) -> bool:
        min_xyz = np.percentile(points_3d, self.config.robust_extent_lower_pct, axis=0)
        max_xyz = np.percentile(points_3d, self.config.robust_extent_upper_pct, axis=0)
        spans = np.maximum(max_xyz - min_xyz, 1e-3)
        horiz_ratio = min(spans[0], spans[2]) / max(spans[0], spans[2])
        vertical_ratio = spans[1] / max(spans[0], spans[2])
        return rectangularity >= 0.72 and horiz_ratio >= 0.35 and vertical_ratio >= 0.8

    def _estimate_box_dimensions(
        self,
        mask: np.ndarray,
        box: np.ndarray,
        points_3d: np.ndarray,
        intrinsics: Dict[str, float],
        depth_map: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Dict]:
        x0, y0, x1, y1 = box.astype(float)
        bbox_w = max(x1 - x0, 1.0)
        bbox_h = max(y1 - y0, 1.0)

        min_xyz = np.percentile(points_3d, self.config.robust_extent_lower_pct, axis=0)
        max_xyz = np.percentile(points_3d, self.config.robust_extent_upper_pct, axis=0)
        spans = np.maximum(max_xyz - min_xyz, 1e-3)  # X, Y, Z in camera space

        z_med = float(np.median(points_3d[:, 2]))
        proj_w = bbox_w * z_med / max(intrinsics["fx"], 1e-6)
        proj_h = bbox_h * z_med / max(intrinsics["fy"], 1e-6)

        width_x = max(float(spans[0]), float(proj_w))
        observed_depth = float(spans[2])
        depth_z = observed_depth

        height_candidates = [float(spans[1]), float(proj_h)]
        height_from_depth = self._estimate_height_from_mask_and_depth(
            mask=mask,
            box=box,
            depth_map=depth_map,
            intrinsics=intrinsics,
        )
        if height_from_depth is not None:
            height_candidates.append(float(height_from_depth))

        height_y = max(height_candidates)

        dims_xyz = np.array([width_x, height_y, depth_z], dtype=np.float32)
        metadata = {
            "estimator": "category_agnostic_box_geometry",
            "rectangularity": self._rectangularity(mask, box),
            "projected_width_m": proj_w,
            "projected_height_m": proj_h,
            "observed_spans_xyz_m": spans.tolist(),
            "median_depth_m": z_med,
            "depth_ratio_prior": None,
            "height_ratio_prior": None,
            "height_from_depth_m": height_from_depth,
            "height_candidates_m": [float(v) for v in height_candidates],
        }
        return dims_xyz, metadata

    def _estimate_height_from_mask_and_depth(
        self,
        mask: np.ndarray,
        box: np.ndarray,
        depth_map: Optional[np.ndarray],
        intrinsics: Dict[str, float],
    ) -> Optional[float]:
        if depth_map is None:
            return None

        x0, y0, x1, y1 = box.astype(int)
        bbox_w = max(x1 - x0, 1)
        bbox_h = max(y1 - y0, 1)
        if bbox_h < 16 or bbox_w < 16:
            return None

        row_counts = mask[y0:y1 + 1].sum(axis=1)
        row_widths = []
        for y in range(y0, y1 + 1):
            xs = np.where(mask[y])[0]
            if len(xs) == 0:
                row_widths.append(0)
            else:
                row_widths.append(int(xs.max() - xs.min() + 1))
        row_widths = np.asarray(row_widths, dtype=np.int32)

        min_body_width = max(4, int(bbox_w * 0.45))
        body_rows = np.where(row_widths >= min_body_width)[0]
        if len(body_rows) < 6:
            return None

        top_y = y0 + int(body_rows[0])
        bottom_y = y0 + int(body_rows[-1])
        band = max(4, int(bbox_h * 0.03))

        top_point = self._sample_band_point(mask, depth_map, intrinsics, top_y, band, prefer="top")
        bottom_point = self._sample_band_point(mask, depth_map, intrinsics, bottom_y, band, prefer="bottom")
        if top_point is None or bottom_point is None:
            return None

        # Use camera-up (Y) component primarily; Euclidean distance as fallback lower bound.
        delta = bottom_point - top_point
        height_y = abs(float(delta[1]))
        height_euclidean = float(np.linalg.norm(delta))
        return max(height_y, height_euclidean * 0.9)

    @staticmethod
    def _sample_band_point(
        mask: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: Dict[str, float],
        center_y: int,
        band: int,
        prefer: str,
    ) -> Optional[np.ndarray]:
        h, w = mask.shape
        ys = np.arange(max(0, center_y - band), min(h, center_y + band + 1))
        if prefer == "bottom":
            ys = ys[::-1]

        candidates = []
        for y in ys:
            xs = np.where(mask[y])[0]
            if len(xs) < 4:
                continue
            x_left, x_right = int(xs.min()), int(xs.max())
            crop_left = int(x_left + 0.25 * (x_right - x_left))
            crop_right = int(x_left + 0.75 * (x_right - x_left))
            crop_right = max(crop_right, crop_left + 1)
            sample_xs = xs[(xs >= crop_left) & (xs <= crop_right)]
            if len(sample_xs) == 0:
                sample_xs = xs

            sample_depths = depth_map[y, sample_xs].astype(np.float32)
            valid = np.isfinite(sample_depths) & (sample_depths > 0.05) & (sample_depths <= 20.0)
            if not valid.any():
                continue

            x = float(np.median(sample_xs[valid]))
            z = float(np.median(sample_depths[valid]))
            X = (x - intrinsics["cx"]) * z / max(intrinsics["fx"], 1e-6)
            Y = (float(y) - intrinsics["cy"]) * z / max(intrinsics["fy"], 1e-6)
            candidates.append(np.array([X, Y, z], dtype=np.float32))

        if not candidates:
            return None
        return np.median(np.stack(candidates, axis=0), axis=0)

    @staticmethod
    def _rectangularity(mask: np.ndarray, box: np.ndarray) -> float:
        x0, y0, x1, y1 = box.astype(float)
        bbox_area = max((x1 - x0) * (y1 - y0), 1.0)
        return float(mask.sum() / bbox_area)

    @staticmethod
    def _build_camera_aligned_geometry(points_3d: np.ndarray, dims_xyz: np.ndarray) -> Dict:
        center = np.median(points_3d, axis=0).astype(np.float32)
        half = dims_xyz / 2.0
        corners_3d = np.array([
            [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] + half[2]],
        ], dtype=np.float32)

        return {
            "center": center,
            "extents": dims_xyz.astype(np.float32),
            "corners": corners_3d,
            "points_3d": points_3d,
        }
