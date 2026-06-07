"""Central configuration for the volume estimation pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class PipelineConfig:
    # Model paths
    translation_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    depth_backend: str = "unidepth"  # "unidepth", "depth_anything_v2", "anycalib_unidepth", or "anycalib_moca3d"
    segmentation_backend: str = "sam3"  # "sam3", "openworldsam", or "sam3_fallback"
    sam3_checkpoint: str = "facebook/sam3"
    sam3_bpe_path: str = os.environ.get("SAM3_BPE_PATH", os.path.join(os.environ.get("HOME", ""), "data-vol1/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    robust_point_filter_mad_threshold: float = 3.5
    robust_point_filter_min_keep_ratio: float = 0.2
    robust_extent_lower_pct: float = 2.0
    robust_extent_upper_pct: float = 98.0
    marker_scale_min: float = 0.7
    marker_scale_max: float = 1.5

    # Device / dtype
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # SAM3
    sam3_resolution: int = 1008
    sam3_confidence_threshold: float = 0.5

    # UniDepthV2
    unidepth_backbone: str = "vitl14"  # "vitl14", "vitb14", or "vits14"

    # AnyCalib + MoCA3D cuboid backend
    anycalib_root: str = os.environ.get("ANYCALIB_ROOT", "/home/irteam/data-vol1/AnyCalib")
    anycalib_model_id: str = os.environ.get("ANYCALIB_MODEL_ID", "anycalib_pinhole")
    anycalib_cam_id: str = os.environ.get("ANYCALIB_CAM_ID", "pinhole")
    moca3d_root: str = os.environ.get("MOCA3D_ROOT", "/home/irteam/data-vol1/MoCA3D")
    moca3d_checkpoint_path: str = os.environ.get(
        "MOCA3D_CHECKPOINT_PATH",
        "/home/irteam/data-vol1/MoCA3D/checkpoints/moca3d.safetensors",
    )
    moca3d_hf_repo: str = os.environ.get("MOCA3D_HF_REPO", "jeoncwcw/MoCA3D")
    moca3d_dinov3_checkpoint_path: str = os.environ.get(
        "MOCA3D_DINOV3_CHECKPOINT_PATH",
        "/home/irteam/data-vol1/MoCA3D/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    moca3d_dinov3_hf_repo: str = os.environ.get(
        "MOCA3D_DINOV3_HF_REPO",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    moca3d_dinov3_hf_filename: str = os.environ.get(
        "MOCA3D_DINOV3_HF_FILENAME",
        "model.safetensors",
    )
    moca3d_input_size: int = 512

    # Camera intrinsics defaults
    default_focal_length_px: float = 1000.0

    # Depth correction factors (single-view limitation)
    depth_correction_factors: Dict[str, float] = field(default_factory=lambda: {})
    default_depth_correction: float = 1.0

    # Reference marker for scale calibration (known real-world size)
    # For 3D markers (e.g., Microwave), include "depth_mm" to use all three axes
    # for scale estimation; otherwise only the two largest PCA extents are matched.
    marker_definitions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "CreditCard": {
            "width_mm": 85.60,
            "height_mm": 53.98,
            "prompt": "credit card",
        },
        "A4": {
            "width_mm": 297.0,
            "height_mm": 210.0,
            "prompt": "A4 paper",
        },
        "Microwave": {
            "width_mm": 435.0,
            "height_mm": 395.0,
            "depth_mm": 255.0,
            "prompt": "microwave oven",
        },
    })

    # Logistics thresholds (cubic meters)
    logistics_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "다마스": 3.0,
        "1톤 트럭": 8.0,
        "2.5톤 트럭": 15.0,
        "5톤 트럭": 30.0,
    })
