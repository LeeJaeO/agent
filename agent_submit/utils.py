"""Utility functions: camera intrinsics, arrow rendering, focal length normalization, visualization."""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw, ExifTags


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def estimate_camera_intrinsics(image: Image.Image) -> Dict[str, float]:
    """Estimate camera intrinsics from EXIF or use sensible defaults.

    Returns dict with keys: fx, fy, cx, cy.
    """
    W, H = image.size

    # Try EXIF focal length
    try:
        exif = image.getexif()
        if exif:
            # Tag 0x920A = FocalLength
            focal_length_mm = exif.get(0x920A)
            if focal_length_mm is not None:
                focal_length_mm = float(focal_length_mm)
                # Approximate: assume 36mm sensor width (full-frame equiv)
                fx = focal_length_mm * W / 36.0
                return {"fx": fx, "fy": fx, "cx": W / 2.0, "cy": H / 2.0}
    except Exception:
        pass

    # Default: assume ~28mm equivalent on phone (common indoor lens)
    fx = max(W, H) * 1.2
    return {"fx": fx, "fy": fx, "cx": W / 2.0, "cy": H / 2.0}


# ---------------------------------------------------------------------------
# Focal length normalization (from DepthLM: utils/datasets.py:91-110)
# ---------------------------------------------------------------------------

def normalize_focal_length(
    image: Image.Image,
    fx: float,
    target_fl: float = 1000.0,
) -> Tuple[Image.Image, float]:
    """Resize image so that focal length becomes target_fl.

    Returns (resized_image, scale_factor).
    """
    scale_factor = target_fl / fx
    new_w = int(image.width * scale_factor)
    new_h = int(image.height * scale_factor)
    return image.resize((new_w, new_h), Image.BILINEAR), scale_factor


# ---------------------------------------------------------------------------
# Arrow marker rendering (from DepthLM: utils/datasets.py:791-811)
# ---------------------------------------------------------------------------

def render_arrow(
    image: Image.Image,
    x: int,
    y: int,
    cross_size: int = 5,
) -> Optional[Image.Image]:
    """Draw a red --> arrow pointing at pixel (x, y).

    Returns a copy of the image with the arrow, or None if the point is
    too close to the border.
    """
    if not (cross_size <= x < image.width - cross_size
            and cross_size <= y < image.height - cross_size):
        return None

    img = image.copy()
    # Horizontal line (shaft)
    for dx in range(1, cross_size + 1):
        img.putpixel((x - dx, y), (255, 0, 0))
    # Arrowhead
    for dy in range(1, cross_size // 2 + 1):
        img.putpixel((x - dy - 1, y + dy), (255, 0, 0))
        img.putpixel((x - dy - 1, y - dy), (255, 0, 0))
    return img


# ---------------------------------------------------------------------------
# Logistics recommendation
# ---------------------------------------------------------------------------

STANDARD_BOX_VOLUME_M3 = 0.06  # ~40x30x50cm moving box

def generate_logistics_recommendation(
    object_volumes: List[Dict],
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Generate Korean moving logistics recommendation."""
    if thresholds is None:
        thresholds = {
            "다마스": 3.0,
            "1톤 트럭": 8.0,
            "2.5톤 트럭": 15.0,
            "5톤 트럭": 30.0,
        }

    total_m3 = sum(v.get("volume_m3", 0) for v in object_volumes)
    total_liters = total_m3 * 1000
    n_boxes = max(1, int(np.ceil(total_m3 / STANDARD_BOX_VOLUME_M3)))

    # Determine truck size
    truck = "5톤 트럭"
    for name, limit in thresholds.items():
        if total_m3 < limit:
            truck = name
            break

    lines = [
        "=" * 50,
        "  이사 물류 추천 (Moving Logistics Recommendation)",
        "=" * 50,
    ]
    for v in object_volumes:
        name = v.get("object", "unknown")
        vol = v.get("volume_m3", 0)
        dims = v.get("dimensions_m", [0, 0, 0])
        dim_str = " x ".join(f"{d*100:.0f}cm" for d in sorted(dims, reverse=True))
        lines.append(f"  {name:20s}  {vol*1e6:>10,.0f} cm³  ({dim_str})")

    lines.append("-" * 50)
    lines.append(f"  총 부피 (Total):      {total_m3:.3f} m³  ({total_liters:,.0f} L)")
    lines.append(f"  예상 박스 수:          약 {n_boxes}개 (40x30x50cm 기준)")
    lines.append(f"  추천 차량:             {truck}")
    lines.append("=" * 50)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_depth_map(
    image: Image.Image,
    depth_map: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Save depth map as a color heatmap side-by-side with the original image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    valid = depth_map[depth_map > 0.01]
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0

    im = axes[1].imshow(depth_map, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1].set_title("Depth Map (meters)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="depth (m)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_per_object_mask(
    image: Image.Image,
    mask: np.ndarray,
    object_name: str,
    score: float,
    save_path: Optional[str] = None,
) -> None:
    """Save a single object's segmentation mask overlaid on the image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Binary mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Mask: {object_name} (score={score:.2f})")
    axes[1].axis("off")

    # Overlay
    img_np = np.array(image).copy()
    overlay = img_np.copy()
    overlay[mask] = [255, 0, 0]
    blended = (img_np * 0.6 + overlay * 0.4).astype(np.uint8)
    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_objects_on_depth(
    image: Image.Image,
    results: List[Dict],
    depth_map: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Save a single image: colorized depth map + mask overlays + name/volume labels for all detected objects."""
    fig, ax = plt.subplots(figsize=(14, 10))

    valid = depth_map[depth_map > 0.01]
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0
    im = ax.imshow(depth_map, cmap="turbo", vmin=vmin, vmax=vmax)

    palette = plt.cm.tab20.colors  # 20 distinct colors
    n_drawn = 0
    for i, res in enumerate(results):
        if "error" in res:
            continue
        mask = res.get("mask")
        if mask is None:
            continue

        color = palette[i % len(palette)]
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[mask] = (*color, 0.45)
        ax.imshow(rgba)

        box = res.get("box")
        if box is not None:
            x0, y0, x1, y1 = [float(v) for v in box]
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor=color, linewidth=1.5,
            )
            ax.add_patch(rect)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = float(xs.mean()), float(ys.mean())
            name = res.get("object", "?")
            vol_cm3 = float(res.get("volume_cm3", 0.0))
            label = f"{name}\n{vol_cm3:,.0f} cm³"
            ax.text(
                cx, cy, label,
                color="white", fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65, ec="none"),
            )
        n_drawn += 1

    ax.set_title(f"Depth map + {n_drawn} segmented object(s)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="depth (m)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def visualize_object_depth(
    image: Image.Image,
    mask: np.ndarray,
    depth_map: np.ndarray,
    object_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Save depth visualization for a single masked object region."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Masked depth
    masked_depth = np.where(mask, depth_map, np.nan)
    valid = depth_map[mask & (depth_map > 0.01)]
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0

    axes[0].imshow(image)
    axes[0].imshow(masked_depth, cmap="turbo", alpha=0.7, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{object_name} — Depth Overlay")
    axes[0].axis("off")

    # Depth histogram
    if len(valid) > 0:
        axes[1].hist(valid, bins=50, color="steelblue", edgecolor="white")
        axes[1].set_xlabel("Depth (m)")
        axes[1].set_ylabel("Pixel count")
        axes[1].set_title(f"{object_name} — Depth Distribution\n"
                          f"mean={valid.mean():.2f}m, std={valid.std():.2f}m")
    else:
        axes[1].text(0.5, 0.5, "No valid depth", ha="center", va="center")
        axes[1].set_title(f"{object_name} — No Data")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_single_object_3d(
    points_3d: np.ndarray,
    obb_geometry: Optional[Dict],
    object_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Save 3D point cloud + OBB for a single object."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points_3d[:, 0], points_3d[:, 2], -points_3d[:, 1],
               c="steelblue", s=1, alpha=0.3)

    if obb_geometry is not None:
        corners = obb_geometry["corners"]
        for a, b in _OBB_EDGES:
            xs = [corners[a, 0], corners[b, 0]]
            ys = [corners[a, 2], corners[b, 2]]
            zs = [-corners[a, 1], -corners[b, 1]]
            ax.plot(xs, ys, zs, c="red", linewidth=1.5)

        extents = obb_geometry["extents"]
        vol_cm3 = float(np.prod(extents)) * 1e6
        ax.set_title(f"{object_name}\n"
                     f"{extents[0]*100:.0f} x {extents[1]*100:.0f} x {extents[2]*100:.0f} cm  "
                     f"({vol_cm3:,.0f} cm³)")
    else:
        ax.set_title(object_name)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth Z (m)")
    ax.set_zlabel("Height (m)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_results(
    image: Image.Image,
    results: List[Dict],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Overlay segmentation masks and volume annotations on image."""
    img = np.array(image).copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (255, 128, 0),
    ]

    for i, res in enumerate(results):
        if "error" in res:
            continue
        mask = res.get("mask")
        if mask is None:
            continue

        color = colors[i % len(colors)]
        alpha_mask = mask.astype(np.float32) * 0.4
        for c in range(3):
            img[:, :, c] = (
                img[:, :, c] * (1 - alpha_mask) + color[c] * alpha_mask
            ).astype(np.uint8)

        # Draw bounding box and label with PIL to avoid OpenCV dependency.
        box = res.get("box")
        name = res.get("object", "?")
        vol = res.get("volume_m3", 0)
        label = f"{name}: {vol*1e6:,.0f} cm3"
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            tx, ty = int(box[0]), max(int(box[1]) - 8, 15)
        else:
            ys, xs = np.where(mask)
            tx, ty = int(xs.min()), max(int(ys.min()) - 8, 15)
        draw.text((tx, ty), label, fill=color)
        img = np.array(img_pil)

    if save_path:
        Image.fromarray(img).save(save_path)
    return img


# OBB edge indices: 12 edges of a cuboid (corner indices 0-7)
_OBB_EDGES = [
    (0,1),(1,2),(2,3),(3,0),  # bottom face
    (4,5),(5,6),(6,7),(7,4),  # top face
    (0,4),(1,5),(2,6),(3,7),  # vertical edges
]


def project_3d_to_2d(
    pts_3d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Project (N,3) 3D camera-space points to (N,2) pixel coordinates."""
    u = fx * pts_3d[:, 0] / pts_3d[:, 2] + cx
    v = fy * pts_3d[:, 1] / pts_3d[:, 2] + cy
    return np.stack([u, v], axis=1)


def visualize_3d_obb_on_image(
    image: Image.Image,
    results: List[Dict],
    intrinsics: Dict[str, float],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Project 3D OBB corners onto the 2D image and draw cuboid wireframes."""
    img = np.array(image).copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 255, 0),
    ]
    fx = intrinsics["fx"]; fy = intrinsics["fy"]
    cx = intrinsics["cx"]; cy = intrinsics["cy"]

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for i, res in enumerate(results):
        obb_geo = res.get("obb_geometry")
        if obb_geo is None:
            continue
        corners_3d = obb_geo["corners"]  # (8, 3)

        # Filter out corners behind the camera
        if np.any(corners_3d[:, 2] <= 0):
            continue

        corners_2d = project_3d_to_2d(corners_3d, fx, fy, cx, cy)
        color = colors[i % len(colors)]

        # Draw 12 edges
        for a, b in _OBB_EDGES:
            p1 = tuple(corners_2d[a].astype(int))
            p2 = tuple(corners_2d[b].astype(int))
            draw.line([p1, p2], fill=color, width=2)

        # Label at centroid projection
        center_2d = project_3d_to_2d(
            obb_geo["center"].reshape(1, 3), fx, fy, cx, cy
        )[0]
        dims = obb_geo["extents"]
        label = f"{res.get('object','?')}: {np.prod(dims)*1e6:,.0f}cm3"
        draw.text(tuple(center_2d.astype(int)), label, fill=color)

    img = np.array(img_pil)

    if save_path:
        Image.fromarray(img).save(save_path)
    return img


def visualize_3d_pointcloud(
    results: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """Render 3D point clouds with OBB wireframes using matplotlib (headless)."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["red", "green", "blue", "orange", "purple", "cyan"]

    has_data = False
    for i, res in enumerate(results):
        obb_geo = res.get("obb_geometry")
        if obb_geo is None:
            continue
        pts = obb_geo["points_3d"]   # (N, 3)
        corners = obb_geo["corners"] # (8, 3)
        c = colors[i % len(colors)]
        name = res.get("object", f"obj{i}")

        # Scatter point cloud
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1],
                   c=c, s=40, alpha=0.8, label=name)

        # Draw OBB edges
        for a, b in _OBB_EDGES:
            xs = [corners[a, 0], corners[b, 0]]
            ys = [corners[a, 2], corners[b, 2]]  # Z → Y axis (depth forward)
            zs = [-corners[a, 1], -corners[b, 1]]  # -Y → Z (up)
            ax.plot(xs, ys, zs, c=c, linewidth=1.5)

        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth Z (m)")
    ax.set_zlabel("Height (m)")
    ax.legend()
    ax.set_title("3D OBB Point Cloud")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
