"""Depth estimation wrappers.

Supports five backends:
  1. DepthAnythingV2 (depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf)
     — dense metric depth map in one forward pass (open model)
  2. Depth Anything 3 (depth-anything/DA3METRIC-LARGE)
     — dense metric depth, newer model (open model, DEFAULT)
  3. DepthLM (facebook/DepthLM) — point-wise VLM-based metric depth (gated model)
  4. Hybrid (DA3 + DepthLM) — pixel-wise average of both models
  5. UniDepthV2 — dense metric depth + camera intrinsics jointly estimated
"""

import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import PipelineConfig
from utils import normalize_focal_length, render_arrow


# ===================================================================
# Depth Anything V2 Metric Indoor — dense metric depth (DEFAULT)
# ===================================================================

class DepthAnythingV2Estimator:
    """Dense metric depth using Depth Anything V2 Metric Indoor.

    Generates a full depth map in one forward pass, then samples depths
    at requested pixel locations. Much faster than per-point VLM queries.
    """

    MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.processor = None
        self._depth_cache: Optional[np.ndarray] = None
        self._cache_image_id: Optional[int] = None

    def load(self):
        """Load the Depth Anything V2 model."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        print(f"  Loading DepthAnythingV2 from {self.MODEL_ID}")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float32,
        ).to(self.config.device)
        self.model.eval()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._depth_cache = None
        self._cache_image_id = None
        torch.cuda.empty_cache()

    def get_depth_map(self, image: Image.Image) -> np.ndarray:
        """Compute dense metric depth map (H, W) in meters. Cached per image."""
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return self._depth_cache

        inputs = self.processor(images=image, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth  # (1, H', W')

        # Resize to original image size
        depth_resized = F.interpolate(
            depth.unsqueeze(0),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        depth_np = depth_resized.cpu().numpy()
        self._depth_cache = depth_np
        self._cache_image_id = img_id
        return depth_np

    def estimate_depth_at_point(
        self,
        image: Image.Image,
        x: int,
        y: int,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Get metric depth at pixel (x, y) from the dense depth map."""
        depth_map = self.get_depth_map(image)
        h, w = depth_map.shape
        if 0 <= y < h and 0 <= x < w:
            val = float(depth_map[y, x])
            if 0.01 <= val <= 100.0:
                return val
        return None

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        """Estimate depth at multiple points (single forward pass)."""
        depth_map = self.get_depth_map(image)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths


# ===================================================================
# Depth Anything 3 Metric — dense metric depth (DEFAULT)
# ===================================================================

class DepthAnything3Estimator:
    """Dense metric depth using Depth Anything 3 (DA3METRIC-LARGE).

    Generates a full depth map in one forward pass, then samples depths
    at requested pixel locations. Requires cuDNN SDPA to be disabled.
    """

    MODEL_ID = "depth-anything/DA3METRIC-LARGE"

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self._depth_cache: Optional[np.ndarray] = None
        self._cache_image_id: Optional[int] = None

    def load(self):
        """Load the Depth Anything 3 model."""
        # Workaround: disable cuDNN SDPA to avoid execution plan errors
        torch.backends.cuda.enable_cudnn_sdp(False)

        from depth_anything_3.api import DepthAnything3

        print(f"  Loading DepthAnything3 from {self.MODEL_ID}")
        self.model = DepthAnything3.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.config.device)
        self.model.eval()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._depth_cache = None
        self._cache_image_id = None
        torch.cuda.empty_cache()

    def get_depth_map(self, image: Image.Image) -> np.ndarray:
        """Compute dense metric depth map (H, W) in meters. Cached per image."""
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return self._depth_cache

        with torch.no_grad():
            prediction = self.model.inference([image])

        # prediction.depth shape: (1, H', W')
        depth_np = prediction.depth[0]  # (H', W')

        # Resize to original image size if needed
        if depth_np.shape != (image.height, image.width):
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
            depth_resized = F.interpolate(
                depth_tensor,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            depth_np = depth_resized.numpy()

        self._depth_cache = depth_np
        self._cache_image_id = img_id
        return depth_np

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        """Estimate depth at multiple points (single forward pass)."""
        depth_map = self.get_depth_map(image)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths


# ===================================================================
# DepthLM — point-wise VLM metric depth (gated model)
# ===================================================================

DEPTH_PROMPT = (
    "Given this image, how far is the point pointed by the red arrow "
    "from the camera? Output the thinking process in <think> </think> "
    "and final answer (the meter number only, without the unit) in "
    "<answer> </answer> tags."
)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


class DepthLMEstimator:
    """Point-wise depth estimation using DepthLM (Pixtral/LLaVA 12B) via vLLM.

    Uses vLLM for high-throughput batched inference with continuous batching.
    Supports dense depth map generation via grid sampling + scipy interpolation.
    Requires access to facebook/DepthLM on HuggingFace.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.llm = None
        self.sampling_params = None
        self._depth_cache: Optional[np.ndarray] = None
        self._cache_image_id: Optional[int] = None

    def load(self):
        """Load DepthLM via vLLM for fast batched inference."""
        from vllm import LLM, SamplingParams

        print(f"  Loading DepthLM via vLLM from {self.config.depth_model}")
        self.llm = LLM(
            model=self.config.depth_model,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.8,
            enforce_eager=True,  # skip torch.compile → no C compiler needed
        )
        self.sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0,
        )

    def unload(self):
        """Free GPU memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.sampling_params = None
        self._depth_cache = None
        self._cache_image_id = None
        torch.cuda.empty_cache()

    def _make_arrow_image(
        self, img_norm: Image.Image, x: int, y: int,
    ) -> Optional[Image.Image]:
        """Create an image with a red arrow at (x, y)."""
        return render_arrow(img_norm, x, y, self.config.arrow_cross_size)

    @staticmethod
    def _pil_to_data_uri(img: Image.Image) -> str:
        """Convert PIL Image to base64 data URI string for vLLM."""
        import base64
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _build_vllm_request(self, arrow_img: Image.Image) -> Dict:
        """Build a single vLLM chat request for Pixtral."""
        data_uri = self._pil_to_data_uri(arrow_img)
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": DEPTH_PROMPT},
            ],
        }

    def estimate_depth_at_point(
        self,
        image: Image.Image,
        x: int,
        y: int,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Estimate metric depth at pixel (x, y)."""
        fx = (intrinsics or {}).get("fx", self.config.default_focal_length_px)
        img_norm, scale = normalize_focal_length(image, fx, self.config.normalized_focal_length)
        sx, sy = int(x * scale), int(y * scale)

        arrow_img = self._make_arrow_image(img_norm, sx, sy)
        if arrow_img is None:
            return None

        messages = [self._build_vllm_request(arrow_img)]
        outputs = self.llm.chat([messages], sampling_params=self.sampling_params)
        response = outputs[0].outputs[0].text
        return self._parse_depth(response)

    def _estimate_depths_vllm(
        self,
        img_norm: Image.Image,
        points_norm: List[Tuple[int, int]],
    ) -> List[Optional[float]]:
        """Estimate depths at all points using vLLM continuous batching."""
        # Prepare all requests
        conversations = []
        valid_indices = []
        for i, (x, y) in enumerate(points_norm):
            arrow_img = self._make_arrow_image(img_norm, x, y)
            if arrow_img is not None:
                conversations.append([self._build_vllm_request(arrow_img)])
                valid_indices.append(i)

        print(f"    Submitting {len(conversations)} requests to vLLM...")

        # vLLM handles batching internally via continuous batching
        outputs = self.llm.chat(conversations, sampling_params=self.sampling_params)

        # Parse results
        all_depths: Dict[int, Optional[float]] = {}
        for idx, output in zip(valid_indices, outputs):
            response = output.outputs[0].text
            all_depths[idx] = self._parse_depth(response)

        return [all_depths.get(i) for i in range(len(points_norm))]

    def get_depth_map(
        self,
        image: Image.Image,
        intrinsics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate dense depth map via vLLM batch inference + interpolation.

        1. Normalize image to 750px focal length
        2. Sample n_sample points on a grid over the image
        3. vLLM continuous batching inference
        4. Interpolate to full (normalized) resolution
        5. Resize back to original image size

        Returns: (H, W) depth map in meters at original image resolution.
        """
        from scipy.interpolate import griddata

        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return self._depth_cache

        fx = (intrinsics or {}).get("fx", self.config.default_focal_length_px)
        img_norm, scale = normalize_focal_length(image, fx, self.config.normalized_focal_length)
        nw, nh = img_norm.size

        # Generate grid sample points (avoid borders for arrow rendering)
        n_sample = self.config.depthlm_n_sample
        margin = self.config.arrow_cross_size + 1
        n_x = int(np.sqrt(n_sample * nw / nh))
        n_y = int(n_sample / n_x)
        xs_grid = np.linspace(margin, nw - margin - 1, n_x).astype(int)
        ys_grid = np.linspace(margin, nh - margin - 1, n_y).astype(int)
        xx, yy = np.meshgrid(xs_grid, ys_grid)
        sample_points = list(zip(xx.ravel().tolist(), yy.ravel().tolist()))

        print(f"    DepthLM vLLM dense mode: {len(sample_points)} sample points "
              f"({n_x}x{n_y} grid) on {nw}x{nh} image")

        # vLLM batch inference
        depths = self._estimate_depths_vllm(img_norm, sample_points)

        # Filter valid points for interpolation
        valid_pts = []
        valid_depths = []
        for (x, y), d in zip(sample_points, depths):
            if d is not None:
                valid_pts.append([x, y])
                valid_depths.append(d)

        print(f"    Valid depth samples: {len(valid_depths)}/{len(sample_points)}")

        if len(valid_depths) < 4:
            print("    WARNING: too few valid depths, returning zeros")
            self._depth_cache = np.zeros((image.height, image.width))
            self._cache_image_id = img_id
            return self._depth_cache

        # Interpolate to normalized resolution
        valid_pts = np.array(valid_pts)
        valid_depths = np.array(valid_depths)
        grid_x, grid_y = np.meshgrid(np.arange(nw), np.arange(nh))
        depth_norm = griddata(valid_pts, valid_depths, (grid_x, grid_y), method="cubic")

        # Fill NaN with nearest-neighbor
        nan_mask = np.isnan(depth_norm)
        if nan_mask.any():
            depth_nearest = griddata(
                valid_pts, valid_depths, (grid_x, grid_y), method="nearest",
            )
            depth_norm[nan_mask] = depth_nearest[nan_mask]

        # Resize to original image size
        if (nw, nh) != (image.width, image.height):
            depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).float()
            depth_resized = F.interpolate(
                depth_tensor,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            depth_np = depth_resized.numpy()
        else:
            depth_np = depth_norm

        self._depth_cache = depth_np
        self._cache_image_id = img_id
        return depth_np

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        """Estimate depth at multiple points using dense depth map."""
        depth_map = self.get_depth_map(image, intrinsics=intrinsics)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths

    @staticmethod
    def _parse_depth(response: str) -> Optional[float]:
        """Extract depth value from <answer> tags."""
        match = re.search(r'<answer>\s*([\d.]+)\s*</answer>', response)
        if match:
            value = float(match.group(1))
            if 0.01 <= value <= 100.0:
                return value
        return None


# ===================================================================
# Hybrid — DA3 + DepthLM pixel-wise average
# ===================================================================

class HybridDepthEstimator:
    """Combines Depth Anything 3 and DepthLM by averaging their depth maps.

    DA3 runs a single fast forward pass, DepthLM runs batched vLLM inference.
    The final depth map is the per-pixel average of both.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.da3 = DepthAnything3Estimator(config)
        self.depthlm = DepthLMEstimator(config)
        self._depth_cache: Optional[np.ndarray] = None
        self._cache_image_id: Optional[int] = None

    def load(self):
        print("  [Hybrid] Loading DA3...")
        self.da3.load()
        print("  [Hybrid] Loading DepthLM...")
        self.depthlm.load()

    def unload(self):
        self.da3.unload()
        self.depthlm.unload()
        self._depth_cache = None
        self._cache_image_id = None

    def get_depth_map(
        self,
        image: Image.Image,
        intrinsics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Average of DA3 and DepthLM depth maps, per pixel."""
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return self._depth_cache

        print("    [Hybrid] Running DA3...")
        depth_da3 = self.da3.get_depth_map(image)

        print("    [Hybrid] Running DepthLM...")
        depth_lm = self.depthlm.get_depth_map(image, intrinsics=intrinsics)

        # Both are (H, W) at original image resolution
        # Average where both are valid (> 0.01m)
        valid_da3 = depth_da3 > 0.01
        valid_lm = depth_lm > 0.01
        both_valid = valid_da3 & valid_lm

        depth_avg = np.zeros_like(depth_da3)
        depth_avg[both_valid] = (depth_da3[both_valid] + depth_lm[both_valid]) / 2.0
        # Where only one is valid, use that one
        only_da3 = valid_da3 & ~valid_lm
        only_lm = valid_lm & ~valid_da3
        depth_avg[only_da3] = depth_da3[only_da3]
        depth_avg[only_lm] = depth_lm[only_lm]

        n_both = both_valid.sum()
        n_total = (depth_avg > 0.01).sum()
        print(f"    [Hybrid] Merged: {n_both} pixels averaged, {n_total} total valid")

        self._depth_cache = depth_avg
        self._cache_image_id = img_id
        return depth_avg

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        depth_map = self.get_depth_map(image, intrinsics=intrinsics)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths


# ===================================================================
# VGGT — depth + camera intrinsics + 3D points (single-image capable)
# ===================================================================

class VGGTEstimator:
    """Dense metric depth + camera intrinsics + 3D points using VGGT.

    Like UniDepthV2, VGGT jointly estimates depth, camera parameters,
    and 3D world points from a single image, eliminating the need for
    separate intrinsics estimation and backprojection.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self._depth_cache: Optional[np.ndarray] = None
        self._points3d_cache: Optional[np.ndarray] = None
        self._intrinsics_cache: Optional[Dict[str, float]] = None
        self._cache_image_id: Optional[int] = None

    def load(self):
        """Load VGGT model from HuggingFace Hub or a fine-tuned local checkpoint."""
        import sys as _sys

        if self.config.vggt_root not in _sys.path:
            _sys.path.insert(0, self.config.vggt_root)

        from vggt.models.vggt import VGGT

        if self.config.vggt_checkpoint_path:
            ckpt_path = self.config.vggt_checkpoint_path
            print(f"  Loading VGGT from local checkpoint: {ckpt_path}")
            self.model = VGGT()
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file as safetensors_load_file

                state_dict = safetensors_load_file(ckpt_path, device="cpu")
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                state_dict = state_dict["model"] if "model" in state_dict else state_dict
            info = self.model.load_state_dict(state_dict, strict=False)
            missing = len(info.missing_keys) if hasattr(info, "missing_keys") else 0
            unexpected = len(info.unexpected_keys) if hasattr(info, "unexpected_keys") else 0
            print(f"  Loaded fine-tuned VGGT checkpoint (missing={missing}, unexpected={unexpected})")
        else:
            print("  Loading VGGT-1B from facebook/VGGT-1B")
            self.model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.model = self.model.to(self.config.device).eval()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._depth_cache = None
        self._points3d_cache = None
        self._intrinsics_cache = None
        self._cache_image_id = None
        torch.cuda.empty_cache()

    def _infer(self, image: Image.Image):
        """Run VGGT inference and cache depth, 3D points, and intrinsics."""
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return

        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        W, H = image.size

        # Save to temp file for load_and_preprocess (expects file paths)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f, format="JPEG")
            tmp_path = f.name

        try:
            images, coords = load_and_preprocess_images_square([tmp_path], target_size=518)
            images = images.to(self.config.device)
            # coords: (1, 6) → [x1, y1, x2, y2, orig_w, orig_h] in 518×518 space
            self._coords = coords[0].numpy()  # (6,)
        finally:
            os.unlink(tmp_path)

        # Inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = self.model(images)

        # Extract intrinsics from pose encoding
        pose_enc = preds["pose_enc"]  # (B, S, 9)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )
        # extrinsic/intrinsic may be numpy or tensor
        if isinstance(intrinsic, torch.Tensor):
            K = intrinsic[0, 0].cpu().numpy()
        else:
            K = intrinsic[0, 0]

        # Scale intrinsics: VGGT K is for 518×518 padded space
        # Convert to original image coordinates via ROI mapping
        x1c, y1c, x2c, y2c = self._coords[:4]
        roi_w = x2c - x1c
        roi_h = y2c - y1c
        scale_x = W / roi_w
        scale_y = H / roi_h
        self._intrinsics_cache = {
            "fx": float(K[0, 0]) * scale_x,
            "fy": float(K[1, 1]) * scale_y,
            "cx": (float(K[0, 2]) - x1c) * scale_x,
            "cy": (float(K[1, 2]) - y1c) * scale_y,
        }

        # Depth map: preds["depth"] shape is (B, S, H, W, 1) → squeeze to (H, W)
        depth_raw = preds["depth"][0, 0, :, :, 0]  # batch=0, view=0, last dim squeezed → (H, W)
        depth_proc = depth_raw.float().cpu().numpy() if isinstance(depth_raw, torch.Tensor) else depth_raw

        # Extract depth from ROI (exclude padding) and resize to original image size
        x1, y1, x2, y2 = self._coords[:4].astype(int)
        depth_roi = depth_proc[y1:y2, x1:x2]  # crop out padding
        if depth_roi.shape != (H, W):
            depth_tensor = torch.from_numpy(depth_roi).unsqueeze(0).unsqueeze(0)
            depth_resized = F.interpolate(
                depth_tensor, size=(H, W), mode="bilinear", align_corners=False
            )[0, 0]
            self._depth_cache = depth_resized.numpy()
        else:
            self._depth_cache = depth_roi

        # 3D points via depth unprojection (more accurate than point_head)
        # preds["depth"] is (B, S, H, W, 1), unproject expects (S, H, W, 1) or (S, H, W)
        depth_for_unproj = preds["depth"][0]  # (S, H, W, 1)
        ext_for_unproj = extrinsic[0]         # (S, 3, 4)
        int_for_unproj = intrinsic[0]         # (S, 3, 3)
        world_points = unproject_depth_map_to_point_map(
            depth_for_unproj, ext_for_unproj, int_for_unproj,
        )  # (S, H, W, 3) numpy
        self._points3d_cache = world_points[0]  # (proc_H, proc_W, 3)

        self._cache_image_id = img_id

    def get_depth_map(self, image: Image.Image, **kwargs) -> np.ndarray:
        """Compute dense metric depth map (H, W) in meters."""
        self._infer(image)
        return self._depth_cache

    def get_intrinsics(self, image: Image.Image) -> Dict[str, float]:
        """Get model-estimated camera intrinsics."""
        self._infer(image)
        return self._intrinsics_cache

    def get_points3d_for_mask(self, image: Image.Image, mask: np.ndarray) -> np.ndarray:
        """Extract 3D points for masked pixels from model output.

        Handles the coordinate mapping between original image (H, W) and
        VGGT's square-padded processing space (518×518).

        Args:
            image: PIL RGB image.
            mask: (H, W) boolean mask at original image resolution.

        Returns:
            (N, 3) array of 3D points.
        """
        self._infer(image)

        H, W = mask.shape
        pts_h, pts_w = self._points3d_cache.shape[:2]  # 518, 518

        # Map mask from original image space → VGGT 518×518 padded space
        # _coords = [x1, y1, x2, y2, orig_w, orig_h] in 518×518 space
        x1, y1, x2, y2 = self._coords[:4].astype(int)
        roi_w = max(x2 - x1, 1)
        roi_h = max(y2 - y1, 1)

        # Resize mask to the ROI region within 518×518
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_roi = np.array(
            mask_img.resize((roi_w, roi_h), Image.Resampling.NEAREST)
        ).astype(bool)

        # Place into full 518×518 mask (padded areas = False)
        mask_full = np.zeros((pts_h, pts_w), dtype=bool)
        mask_full[y1:y1+roi_h, x1:x1+roi_w] = mask_roi

        # Extract 3D points
        points = self._points3d_cache[mask_full]  # (N, 3)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] <= 100.0)
        return points[valid]

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        depth_map = self.get_depth_map(image)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths


class VGGTNewOneEstimator(VGGTEstimator):
    """Single-image VGGT backend that follows the mp4 inference path closely.

    The goal is to keep image input while matching the mp4 backend's preprocessing,
    inference mode, and depth / point-map extraction behavior as closely as possible.
    """

    def _infer(self, image: Image.Image):
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return

        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        W, H = image.size

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f, format="JPEG")
            tmp_path = f.name

        try:
            images, coords = load_and_preprocess_images_square([tmp_path], target_size=518)
            images = images.to(self.config.device)
            with torch.inference_mode():
                predictions = self.model(images)
        finally:
            os.unlink(tmp_path)

        self._coords = coords[0].detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords[0])

        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        if isinstance(intrinsic, torch.Tensor):
            K = intrinsic[0, 0].detach().cpu().numpy()
        else:
            K = np.asarray(intrinsic[0, 0])

        x1c, y1c, x2c, y2c = self._coords[:4]
        roi_w = max(float(x2c - x1c), 1.0)
        roi_h = max(float(y2c - y1c), 1.0)
        scale_x = W / roi_w
        scale_y = H / roi_h
        self._intrinsics_cache = {
            "fx": float(K[0, 0]) * scale_x,
            "fy": float(K[1, 1]) * scale_y,
            "cx": (float(K[0, 2]) - x1c) * scale_x,
            "cy": (float(K[1, 2]) - y1c) * scale_y,
        }

        depth_predictions = predictions["depth"][0]  # (1, H, W, 1)
        depth_proc = depth_predictions[..., 0]
        if isinstance(depth_proc, torch.Tensor):
            depth_proc = depth_proc.detach().cpu().numpy()
        else:
            depth_proc = np.asarray(depth_proc)
        depth_proc = depth_proc[0]

        x1, y1, x2, y2 = self._coords[:4].astype(int)
        depth_roi = depth_proc[y1:y2, x1:x2]
        if depth_roi.shape != (H, W):
            depth_tensor = torch.from_numpy(depth_roi).unsqueeze(0).unsqueeze(0)
            depth_resized = F.interpolate(
                depth_tensor, size=(H, W), mode="bilinear", align_corners=False
            )[0, 0]
            self._depth_cache = depth_resized.detach().cpu().numpy()
        else:
            self._depth_cache = depth_roi

        ext_for_unproj = extrinsic[0] if isinstance(extrinsic, torch.Tensor) else extrinsic[0]
        int_for_unproj = intrinsic[0] if isinstance(intrinsic, torch.Tensor) else intrinsic[0]
        world_points = unproject_depth_map_to_point_map(
            depth_predictions,
            ext_for_unproj,
            int_for_unproj,
        )
        if isinstance(world_points, torch.Tensor):
            world_points = world_points.detach().cpu().numpy()
        else:
            world_points = np.asarray(world_points)
        self._points3d_cache = world_points[0]
        self._cache_image_id = img_id


class VGGTMp4Estimator(VGGTEstimator):
    """VGGT video estimator using multiple sampled frames from an mp4-like file.

    The estimator runs VGGT on a short sequence sampled from the video and uses the
    middle sampled frame as the reference image for segmentation and volume estimation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config=config)
        self._video_cache_key: Optional[Tuple[str, float]] = None
        self._video_frames: Optional[List[Image.Image]] = None
        self._video_frame_numbers: Optional[List[int]] = None
        self._reference_image_cache: Optional[Image.Image] = None
        self._reference_frame_index: Optional[int] = None
        self._video_coords: Optional[np.ndarray] = None
        self._video_extrinsics: Optional[np.ndarray] = None
        self._video_proc_intrinsics: Optional[np.ndarray] = None
        self._video_orig_intrinsics: Optional[List[Dict[str, float]]] = None
        self._video_proc_depths: Optional[np.ndarray] = None
        self._video_world_points: Optional[np.ndarray] = None
        self._video_depth_maps: Optional[List[Optional[np.ndarray]]] = None

    def unload(self):
        super().unload()
        self._video_cache_key = None
        self._video_frames = None
        self._video_frame_numbers = None
        self._reference_image_cache = None
        self._reference_frame_index = None
        self._video_coords = None
        self._video_extrinsics = None
        self._video_proc_intrinsics = None
        self._video_orig_intrinsics = None
        self._video_proc_depths = None
        self._video_world_points = None
        self._video_depth_maps = None

    def get_reference_image(self, video_path: str) -> Image.Image:
        self._ensure_video_frames(video_path)
        if self._reference_image_cache is None:
            raise RuntimeError(f"Failed to extract a reference frame from video: {video_path}")
        return self._reference_image_cache.copy()

    def get_frame_image(self, video_path: str, frame_index: int) -> Image.Image:
        self._ensure_video_frames(video_path)
        frame_index = self._resolve_frame_index(frame_index)
        assert self._video_frames is not None
        return self._video_frames[frame_index].copy()

    def get_sampled_frame_numbers(self, video_path: str) -> List[int]:
        self._ensure_video_frames(video_path)
        return list(self._video_frame_numbers or [])

    def _ensure_video_frames(self, video_path: str) -> None:
        video_path = os.path.abspath(video_path)
        mtime = os.path.getmtime(video_path)
        cache_key = (video_path, mtime)
        if self._video_cache_key == cache_key and self._video_frames is not None:
            return

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            capture.release()
            raise RuntimeError(f"Video contains no readable frames: {video_path}")

        target_count = min(self.config.vggt_video_num_frames, total_frames)
        sampled_indices = np.linspace(0, total_frames - 1, num=target_count, dtype=int)
        sampled_indices = sorted(set(int(idx) for idx in sampled_indices))

        frames: List[Image.Image] = []
        for frame_idx in sampled_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        capture.release()
        if not frames:
            raise RuntimeError(f"Failed to sample frames from video: {video_path}")

        self._video_cache_key = cache_key
        self._video_frames = frames
        self._video_frame_numbers = sampled_indices[: len(frames)]
        self._reference_frame_index = len(frames) // 2
        self._reference_image_cache = frames[self._reference_frame_index].copy()

    def _infer_video(self, video_path: str):
        self._ensure_video_frames(video_path)
        cache_key = self._video_cache_key
        if self._cache_image_id == cache_key and self._video_world_points is not None:
            return

        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        assert self._video_frames is not None
        assert self._reference_frame_index is not None

        with tempfile.TemporaryDirectory(prefix="vggt_mp4_frames_") as tmp_dir:
            frame_paths = []
            for frame_idx, frame in enumerate(self._video_frames):
                frame_path = os.path.join(tmp_dir, f"{frame_idx:03d}.jpg")
                frame.save(frame_path, format="JPEG")
                frame_paths.append(frame_path)

            images, coords = load_and_preprocess_images_square(frame_paths, target_size=518)
            images = images.to(self.config.device)
            with torch.inference_mode():
                predictions = self.model(images)

        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        self._video_coords = self._to_numpy(coords)
        self._video_extrinsics = self._to_numpy(extrinsic[0] if isinstance(extrinsic, torch.Tensor) else extrinsic[0])
        self._video_proc_intrinsics = self._to_numpy(intrinsic[0] if isinstance(intrinsic, torch.Tensor) else intrinsic[0])
        depth_predictions = predictions["depth"][0]
        self._video_proc_depths = self._to_numpy(depth_predictions[..., 0])
        self._video_world_points = self._to_numpy(
            unproject_depth_map_to_point_map(
                depth_predictions,
                extrinsic[0] if isinstance(extrinsic, torch.Tensor) else extrinsic[0],
                intrinsic[0] if isinstance(intrinsic, torch.Tensor) else intrinsic[0],
            )
        )
        self._video_orig_intrinsics = [
            self._scale_intrinsics_to_original(self._video_proc_intrinsics[idx], idx)
            for idx in range(len(self._video_frames))
        ]
        self._video_depth_maps = [None] * len(self._video_frames)

        self._activate_video_frame(self._resolve_frame_index(None))
        self._cache_image_id = cache_key

    def get_depth_map(self, image: Image.Image, **kwargs) -> np.ndarray:
        video_path = kwargs.get("video_path")
        if not video_path:
            raise ValueError("VGGTMp4Estimator.get_depth_map requires video_path=...")
        self._infer_video(video_path)
        self._activate_video_frame(self._resolve_frame_index(kwargs.get("frame_index")))
        return self._depth_cache

    def get_intrinsics(self, image: Image.Image, **kwargs) -> Dict[str, float]:
        video_path = kwargs.get("video_path")
        if not video_path:
            raise ValueError("VGGTMp4Estimator.get_intrinsics requires video_path=...")
        self._infer_video(video_path)
        self._activate_video_frame(self._resolve_frame_index(kwargs.get("frame_index")))
        return self._intrinsics_cache

    def get_points3d_for_mask(self, image: Image.Image, mask: np.ndarray, **kwargs) -> np.ndarray:
        video_path = kwargs.get("video_path")
        if not video_path:
            raise ValueError("VGGTMp4Estimator.get_points3d_for_mask requires video_path=...")
        self._infer_video(video_path)
        frame_index = self._resolve_frame_index(kwargs.get("frame_index"))
        proc_mask = self._mask_to_proc_mask(mask, frame_index)
        self._activate_video_frame(frame_index)
        return self._extract_points3d_from_proc_mask(frame_index, proc_mask)

    def propagate_mask(
        self,
        mask: np.ndarray,
        video_path: str,
        source_frame_index: Optional[int] = None,
    ) -> Dict:
        self._infer_video(video_path)
        source_frame_index = self._resolve_frame_index(source_frame_index)
        source_proc_mask = self._mask_to_proc_mask(mask, source_frame_index)
        source_points = self._extract_points3d_from_proc_mask(source_frame_index, source_proc_mask)
        if len(source_points) == 0:
            raise RuntimeError("Failed to extract valid 3D points from the source frame mask.")

        propagation_points = self._downsample_for_propagation(source_points)
        frames = []
        total_source_points = max(len(propagation_points), 1)

        assert self._video_frames is not None
        assert self._video_frame_numbers is not None

        for frame_index in range(len(self._video_frames)):
            if frame_index == source_frame_index:
                proc_mask = source_proc_mask.copy()
                visible_points = total_source_points
                visible_ratio = 1.0
            else:
                proc_mask, visible_points, visible_ratio = self._project_world_points_to_proc_mask(
                    propagation_points, frame_index
                )

            mask_orig = self._proc_mask_to_original(proc_mask, frame_index)
            frames.append(
                {
                    "frame_index": frame_index,
                    "video_frame_number": int(self._video_frame_numbers[frame_index]),
                    "mask": mask_orig,
                    "proc_mask": proc_mask,
                    "box": self._mask_to_box(mask_orig),
                    "mask_pixels": int(mask_orig.sum()),
                    "visible_points": int(visible_points),
                    "visible_ratio": float(visible_ratio),
                }
            )

        candidates = [
            frame
            for frame in frames
            if frame["mask_pixels"] >= self.config.vggt_video_min_mask_pixels
            and frame["visible_ratio"] >= self.config.vggt_video_min_visible_ratio
        ]
        if not candidates:
            candidates = [frames[source_frame_index]]

        best = max(candidates, key=lambda item: (item["visible_ratio"], item["mask_pixels"]))

        return {
            "source_frame_index": source_frame_index,
            "source_video_frame_number": int(self._video_frame_numbers[source_frame_index]),
            "source_point_count": int(len(source_points)),
            "frames": frames,
            "best_frame_index": best["frame_index"],
            "best_video_frame_number": best["video_frame_number"],
            "best_mask": best["mask"],
            "best_box": best["box"],
            "best_visible_points": best["visible_points"],
            "best_visible_ratio": best["visible_ratio"],
        }

    def fuse_points3d_from_propagation(self, video_path: str, propagation: Dict) -> Dict:
        self._infer_video(video_path)

        frames = propagation.get("frames", [])
        if not frames:
            return {"points": np.empty((0, 3), dtype=np.float32), "frames": []}

        fusion_mode = getattr(self.config, "vggt_video_fusion_mode", "reference").lower()
        source_frame_index = propagation.get("source_frame_index", propagation.get("best_frame_index", 0))
        best_frame_index = propagation.get("best_frame_index", source_frame_index)

        if fusion_mode == "reference":
            selected_frames = [frame for frame in frames if frame["frame_index"] == source_frame_index]
        elif fusion_mode == "best":
            selected_frames = [frame for frame in frames if frame["frame_index"] == best_frame_index]
        else:
            best_visible_ratio = max(frame["visible_ratio"] for frame in frames)
            min_visible_ratio = max(
                self.config.vggt_video_min_visible_ratio,
                best_visible_ratio * self.config.vggt_video_fusion_relative_visible_ratio,
            )
            candidates = [
                frame
                for frame in frames
                if frame["mask_pixels"] >= self.config.vggt_video_min_mask_pixels
                and frame["visible_ratio"] >= min_visible_ratio
            ]
            if not candidates:
                candidates = [frame for frame in frames if frame["frame_index"] == source_frame_index]

            topk = max(1, int(getattr(self.config, "vggt_video_topk_frames", 3)))
            candidates.sort(key=lambda item: (item["visible_ratio"], item["mask_pixels"]), reverse=True)
            selected_frames = candidates[:topk]

        fused_points = []
        used_frames = []
        for frame in selected_frames:
            points = self._extract_points3d_from_proc_mask(frame["frame_index"], frame["proc_mask"])
            if len(points) == 0:
                continue
            fused_points.append(points)
            used_frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "video_frame_number": frame["video_frame_number"],
                    "n_points": int(len(points)),
                    "visible_ratio": float(frame["visible_ratio"]),
                }
            )

        if not fused_points:
            return {
                "points": np.empty((0, 3), dtype=np.float32),
                "frames": used_frames,
                "fusion_mode": fusion_mode,
            }

        points = np.concatenate(fused_points, axis=0)
        max_fused_points = int(getattr(self.config, "vggt_video_max_fused_points", 0))
        if max_fused_points > 0 and len(points) > max_fused_points:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(points), size=max_fused_points, replace=False)
            points = points[keep]

        return {
            "points": points.astype(np.float32, copy=False),
            "frames": used_frames,
            "fusion_mode": fusion_mode,
        }

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _resolve_frame_index(self, frame_index: Optional[int]) -> int:
        assert self._reference_frame_index is not None
        if frame_index is None:
            return self._reference_frame_index
        assert self._video_frames is not None
        if not (0 <= frame_index < len(self._video_frames)):
            raise IndexError(f"Frame index out of range: {frame_index}")
        return int(frame_index)

    def _scale_intrinsics_to_original(self, intrinsic_proc: np.ndarray, frame_index: int) -> Dict[str, float]:
        assert self._video_frames is not None
        assert self._video_coords is not None

        W, H = self._video_frames[frame_index].size
        x1c, y1c, x2c, y2c = self._video_coords[frame_index][:4]
        roi_w = max(float(x2c - x1c), 1.0)
        roi_h = max(float(y2c - y1c), 1.0)
        scale_x = W / roi_w
        scale_y = H / roi_h
        return {
            "fx": float(intrinsic_proc[0, 0]) * scale_x,
            "fy": float(intrinsic_proc[1, 1]) * scale_y,
            "cx": (float(intrinsic_proc[0, 2]) - x1c) * scale_x,
            "cy": (float(intrinsic_proc[1, 2]) - y1c) * scale_y,
        }

    def _get_or_build_frame_depth_map(self, frame_index: int) -> np.ndarray:
        assert self._video_frames is not None
        assert self._video_proc_depths is not None
        assert self._video_coords is not None
        assert self._video_depth_maps is not None

        cached = self._video_depth_maps[frame_index]
        if cached is not None:
            return cached

        W, H = self._video_frames[frame_index].size
        x1, y1, x2, y2 = self._video_coords[frame_index][:4].astype(int)
        depth_roi = self._video_proc_depths[frame_index][y1:y2, x1:x2]
        if depth_roi.shape != (H, W):
            depth_tensor = torch.from_numpy(depth_roi).unsqueeze(0).unsqueeze(0)
            depth_resized = F.interpolate(
                depth_tensor, size=(H, W), mode="bilinear", align_corners=False
            )[0, 0]
            depth_map = depth_resized.detach().cpu().numpy()
        else:
            depth_map = depth_roi

        self._video_depth_maps[frame_index] = depth_map
        return depth_map

    def _activate_video_frame(self, frame_index: int) -> None:
        assert self._video_coords is not None
        assert self._video_orig_intrinsics is not None
        assert self._video_world_points is not None

        self._coords = self._video_coords[frame_index]
        self._intrinsics_cache = self._video_orig_intrinsics[frame_index]
        self._depth_cache = self._get_or_build_frame_depth_map(frame_index)
        self._points3d_cache = self._video_world_points[frame_index]

    def _mask_to_proc_mask(self, mask: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._video_world_points is not None
        assert self._video_coords is not None

        pts_h, pts_w = self._video_world_points[frame_index].shape[:2]
        x1, y1, x2, y2 = self._video_coords[frame_index][:4].astype(int)
        roi_w = max(x2 - x1, 1)
        roi_h = max(y2 - y1, 1)

        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_roi = np.array(mask_img.resize((roi_w, roi_h), Image.Resampling.NEAREST)) > 0

        mask_full = np.zeros((pts_h, pts_w), dtype=bool)
        mask_full[y1:y1 + roi_h, x1:x1 + roi_w] = mask_roi
        return mask_full

    def _proc_mask_to_original(self, proc_mask: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._video_frames is not None
        assert self._video_coords is not None

        W, H = self._video_frames[frame_index].size
        x1, y1, x2, y2 = self._video_coords[frame_index][:4].astype(int)
        roi = proc_mask[y1:y2, x1:x2].astype(np.uint8) * 255
        mask_img = Image.fromarray(roi, mode="L")
        mask_orig = np.array(mask_img.resize((W, H), Image.Resampling.NEAREST)) > 0
        return mask_orig

    def _extract_points3d_from_proc_mask(self, frame_index: int, proc_mask: np.ndarray) -> np.ndarray:
        assert self._video_world_points is not None

        points = self._video_world_points[frame_index][proc_mask]
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] <= 100.0)
        return points[valid]

    def _downsample_for_propagation(self, points: np.ndarray) -> np.ndarray:
        max_points = int(getattr(self.config, "vggt_video_max_propagation_points", 0))
        if max_points <= 0 or len(points) <= max_points:
            return points
        rng = np.random.default_rng(42)
        keep = rng.choice(len(points), size=max_points, replace=False)
        return points[keep]

    def _project_world_points_to_proc_mask(
        self,
        world_points: np.ndarray,
        frame_index: int,
    ) -> Tuple[np.ndarray, int, float]:
        assert self._video_world_points is not None
        assert self._video_extrinsics is not None
        assert self._video_proc_intrinsics is not None

        proc_h, proc_w = self._video_world_points[frame_index].shape[:2]
        if len(world_points) == 0:
            return np.zeros((proc_h, proc_w), dtype=bool), 0, 0.0

        extrinsic = self._video_extrinsics[frame_index].astype(np.float32)
        intrinsic = self._video_proc_intrinsics[frame_index].astype(np.float32)
        world_points = np.asarray(world_points, dtype=np.float32)

        world_points_h = np.concatenate(
            [world_points, np.ones((len(world_points), 1), dtype=np.float32)],
            axis=1,
        )
        camera_points = (extrinsic @ world_points_h.T).T
        valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-4)
        if not valid.any():
            return np.zeros((proc_h, proc_w), dtype=bool), 0, 0.0

        camera_points = camera_points[valid]
        z = camera_points[:, 2]
        u = intrinsic[0, 0] * (camera_points[:, 0] / z) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (camera_points[:, 1] / z) + intrinsic[1, 2]

        x = np.rint(u).astype(np.int32)
        y = np.rint(v).astype(np.int32)
        in_bounds = (x >= 0) & (x < proc_w) & (y >= 0) & (y < proc_h)
        visible_points = int(in_bounds.sum())
        visible_ratio = visible_points / float(len(world_points))
        if not in_bounds.any():
            return np.zeros((proc_h, proc_w), dtype=bool), 0, visible_ratio

        mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
        mask[y[in_bounds], x[in_bounds]] = 255

        kernel_size = max(1, self.config.vggt_video_mask_dilation_px)
        if kernel_size > 0:
            kernel = np.ones((kernel_size * 2 + 1, kernel_size * 2 + 1), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        mask = self._keep_largest_component(mask)
        return mask > 0, visible_points, visible_ratio

    @staticmethod
    def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(mask)
        out[labels == largest_label] = 255
        return out

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.array(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            dtype=np.float32,
        )


class VGGTWorldPointsEstimator(VGGTEstimator):
    """VGGT variant that uses the model's world_points head directly
    instead of depth unprojection.
    """

    def _infer(self, image: Image.Image):
        """Run VGGT inference, using world_points from point_head."""
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return

        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        W, H = image.size

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            image.save(f, format="JPEG")
            tmp_path = f.name

        try:
            images, coords = load_and_preprocess_images_square([tmp_path], target_size=518)
            images = images.to(self.config.device)
            self._coords = coords[0].numpy()
        finally:
            os.unlink(tmp_path)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = self.model(images)

        # Intrinsics (same as parent)
        pose_enc = preds["pose_enc"]
        _, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        if isinstance(intrinsic, torch.Tensor):
            K = intrinsic[0, 0].cpu().numpy()
        else:
            K = intrinsic[0, 0]

        x1c, y1c, x2c, y2c = self._coords[:4]
        roi_w = x2c - x1c
        roi_h = y2c - y1c
        scale_x = W / roi_w
        scale_y = H / roi_h
        self._intrinsics_cache = {
            "fx": float(K[0, 0]) * scale_x,
            "fy": float(K[1, 1]) * scale_y,
            "cx": (float(K[0, 2]) - x1c) * scale_x,
            "cy": (float(K[1, 2]) - y1c) * scale_y,
        }

        # Depth map (same as parent)
        depth_raw = preds["depth"][0, 0, :, :, 0]
        depth_proc = depth_raw.float().cpu().numpy() if isinstance(depth_raw, torch.Tensor) else depth_raw
        x1, y1, x2, y2 = self._coords[:4].astype(int)
        depth_roi = depth_proc[y1:y2, x1:x2]
        if depth_roi.shape != (H, W):
            depth_tensor = torch.from_numpy(depth_roi).unsqueeze(0).unsqueeze(0)
            depth_resized = F.interpolate(
                depth_tensor, size=(H, W), mode="bilinear", align_corners=False
            )[0, 0]
            self._depth_cache = depth_resized.numpy()
        else:
            self._depth_cache = depth_roi

        # 3D points: use world_points directly from point_head
        # preds["world_points"] shape: (B, S, H, W, 3)
        wp = preds["world_points"][0, 0]  # (518, 518, 3)
        self._points3d_cache = wp.float().cpu().numpy() if isinstance(wp, torch.Tensor) else wp

        self._cache_image_id = img_id


# ===================================================================
# Depth Pro — Apple's sharp monocular metric depth + FOV estimation
# ===================================================================

class DepthProEstimator:
    """Dense metric depth + focal length using Apple's Depth Pro.

    Estimates depth and focal length (FOV head) from a single image.
    Focal length is converted to camera intrinsics (fx=fy, cx=W/2, cy=H/2).
    """

    DEPTH_PRO_ROOT = "/home/irteam/data-vol1/ml-depth-pro"

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.transform = None
        self._depth_cache: Optional[np.ndarray] = None
        self._intrinsics_cache: Optional[Dict[str, float]] = None
        self._cache_image_id: Optional[int] = None

    def load(self):
        """Load Depth Pro model."""
        import sys as _sys

        src_path = f"{self.DEPTH_PRO_ROOT}/src"
        if src_path not in _sys.path:
            _sys.path.insert(0, src_path)

        from depth_pro import create_model_and_transforms
        from depth_pro.depth_pro import DepthProConfig

        checkpoint = f"{self.DEPTH_PRO_ROOT}/checkpoints/depth_pro.pt"
        print(f"  Loading Depth Pro from {checkpoint}")

        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=checkpoint,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        self.model, self.transform = create_model_and_transforms(
            config=config,
            device=torch.device(self.config.device),
            precision=torch.half,
        )
        self.model.eval()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.transform = None
        self._depth_cache = None
        self._intrinsics_cache = None
        self._cache_image_id = None
        torch.cuda.empty_cache()

    def _infer(self, image: Image.Image):
        """Run Depth Pro inference and cache depth + intrinsics."""
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return

        W, H = image.size

        # Preprocess: PIL → numpy → transform → tensor
        image_np = np.array(image)
        image_tensor = self.transform(image_np).to(self.config.device)

        with torch.no_grad():
            out = self.model.infer(image_tensor, f_px=None)

        # depth: (H, W) tensor → numpy
        self._depth_cache = out["depth"].cpu().numpy().astype(np.float32)

        # focallength_px: scalar → intrinsics
        f_px = float(out["focallength_px"].cpu())
        self._intrinsics_cache = {
            "fx": f_px,
            "fy": f_px,
            "cx": W / 2.0,
            "cy": H / 2.0,
        }

        self._cache_image_id = img_id

    def get_depth_map(self, image: Image.Image, **kwargs) -> np.ndarray:
        """Compute dense metric depth map (H, W) in meters."""
        self._infer(image)
        return self._depth_cache

    def get_intrinsics(self, image: Image.Image) -> Dict[str, float]:
        """Get model-estimated camera intrinsics from FOV head."""
        self._infer(image)
        return self._intrinsics_cache

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        depth_map = self.get_depth_map(image)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths


# ===================================================================
# UniDepthV2 — joint depth + intrinsics estimation
# ===================================================================

class UniDepthV2Estimator:
    """Dense metric depth + camera intrinsics using UniDepthV2.

    Unlike other backends, this model jointly estimates depth and camera
    intrinsics from a single image, eliminating the need for EXIF-based
    or heuristic intrinsics estimation.
    """

    UNIDEPTH_ROOT = "/home/irteam/data-vol1/UniDepth"

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self._depth_cache: Optional[np.ndarray] = None
        self._points3d_cache: Optional[np.ndarray] = None
        self._intrinsics_cache: Optional[Dict[str, float]] = None
        self._cache_image_id: Optional[int] = None

    def load(self):
        """Load UniDepthV2 model."""
        import sys as _sys

        if self.UNIDEPTH_ROOT not in _sys.path:
            _sys.path.insert(0, self.UNIDEPTH_ROOT)

        from unidepth.models import UniDepthV2
        import json

        backbone = self.config.unidepth_backbone
        config_path = f"{self.UNIDEPTH_ROOT}/configs/config_v2_{backbone}.json"
        print(f"  Loading UniDepthV2 (backbone={backbone})")

        with open(config_path) as f:
            model_config = json.load(f)

        self.model = UniDepthV2(model_config)

        import huggingface_hub
        weights_path = huggingface_hub.hf_hub_download(
            repo_id=f"lpiccinelli/unidepth-v2-{backbone}",
            filename="pytorch_model.bin",
            repo_type="model",
        )
        info = self.model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
        if info.missing_keys:
            print(f"    Missing keys: {info.missing_keys}")

        self.model = self.model.to(self.config.device).eval()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._depth_cache = None
        self._points3d_cache = None
        self._intrinsics_cache = None
        self._cache_image_id = None
        torch.cuda.empty_cache()

    def _infer(self, image: Image.Image):
        """Run UniDepthV2 inference and cache depth, 3D points, and intrinsics."""
        img_id = id(image)
        if self._cache_image_id == img_id and self._depth_cache is not None:
            return

        # Convert PIL to (3, H, W) uint8 tensor
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # (3, H, W)

        with torch.no_grad():
            out = self.model.infer(rgb)

        # depth: (1, 1, H, W) → (H, W)
        self._depth_cache = out["depth"][0, 0].cpu().numpy()

        # points: (1, 3, H, W) → (3, H, W) dense 3D coordinates
        self._points3d_cache = out["points"][0].cpu().numpy()  # (3, H, W)

        # intrinsics: (1, 3, 3) → extract fx, fy, cx, cy
        K = out["intrinsics"][0].cpu().numpy()  # (3, 3)
        self._intrinsics_cache = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }

        self._cache_image_id = img_id

    def get_depth_map(self, image: Image.Image, **kwargs) -> np.ndarray:
        """Compute dense metric depth map (H, W) in meters."""
        self._infer(image)
        return self._depth_cache

    def get_intrinsics(self, image: Image.Image) -> Dict[str, float]:
        """Get model-estimated camera intrinsics (fx, fy, cx, cy)."""
        self._infer(image)
        return self._intrinsics_cache

    def get_points3d_for_mask(self, image: Image.Image, mask: np.ndarray) -> np.ndarray:
        """Extract 3D points for masked pixels directly from model output.

        Uses UniDepthV2's internally computed 3D points (depth + intrinsics
        jointly estimated), avoiding separate backprojection and its error
        propagation.

        Args:
            image: PIL RGB image (triggers inference if not cached).
            mask: (H, W) boolean mask of the object.

        Returns:
            (N, 3) array of 3D points [X, Y, Z] for masked pixels.
        """
        self._infer(image)
        # _points3d_cache shape: (3, H, W) — channels are X, Y, Z
        ys, xs = np.where(mask)
        points = self._points3d_cache[:, ys, xs]  # (3, N)
        points = points.T  # (N, 3)

        # Filter invalid points
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] <= 100.0)
        return points[valid]

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        depth_map = self.get_depth_map(image)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths


# ===================================================================
# AnyCalib + UniDepthV2 — AnyCalib intrinsics + dense UniDepth depth
# ===================================================================

class AnyCalibUniDepthEstimator(UniDepthV2Estimator):
    """Dense UniDepthV2 metric depth with camera intrinsics from AnyCalib.

    UniDepthV2 still provides the dense metric depth map, but masked 3D points
    are backprojected with AnyCalib's pinhole intrinsics instead of UniDepth's
    internally estimated camera matrix.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config)
        self.anycalib_model = None
        self._anycalib_intrinsics_cache: Optional[Dict[str, float]] = None
        self._anycalib_cache_image_id: Optional[int] = None

    def load(self):
        """Load UniDepthV2 and AnyCalib."""
        import sys as _sys
        from pathlib import Path as _Path

        super().load()

        anycalib_root = _Path(self.config.anycalib_root).expanduser()
        if not anycalib_root.exists():
            raise FileNotFoundError(f"AnyCalib root not found: {anycalib_root}")
        if str(anycalib_root) not in _sys.path:
            _sys.path.insert(0, str(anycalib_root))

        from anycalib import AnyCalib

        print(
            f"  Loading AnyCalib model_id={self.config.anycalib_model_id}, "
            f"cam_id={self.config.anycalib_cam_id}"
        )
        self.anycalib_model = AnyCalib(
            model_id=self.config.anycalib_model_id
        ).to(self.config.device).eval()

    def unload(self):
        """Free GPU memory."""
        super().unload()
        if self.anycalib_model is not None:
            del self.anycalib_model
            self.anycalib_model = None
        self._anycalib_intrinsics_cache = None
        self._anycalib_cache_image_id = None
        torch.cuda.empty_cache()

    def _infer_anycalib_intrinsics(self, image: Image.Image):
        if self.anycalib_model is None:
            raise RuntimeError("AnyCalibUniDepthEstimator.load() must be called first.")
        img_id = id(image)
        if (
            self._anycalib_cache_image_id == img_id
            and self._anycalib_intrinsics_cache is not None
        ):
            return

        arr = np.array(image.convert("RGB"), copy=True)
        image_tensor = (
            torch.tensor(arr, dtype=torch.float32, device=self.config.device)
            .permute(2, 0, 1)
            / 255.0
        )

        with torch.inference_mode():
            output = self.anycalib_model.predict(
                image_tensor, cam_id=self.config.anycalib_cam_id
            )
        intr = output["intrinsics"]
        if isinstance(intr, list):
            intr = intr[0]
        intr_np = intr.detach().float().cpu().numpy()
        if len(intr_np) >= 4:
            fx, fy, cx, cy = intr_np[:4]
        elif len(intr_np) == 3:
            fx, cx, cy = intr_np
            fy = fx
        else:
            raise ValueError(
                f"Unsupported AnyCalib intrinsics shape for "
                f"{self.config.anycalib_cam_id}: {intr_np.shape}"
            )

        self._anycalib_intrinsics_cache = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
        self._anycalib_cache_image_id = img_id

    def get_intrinsics(self, image: Image.Image) -> Dict[str, float]:
        """Get AnyCalib-estimated camera intrinsics (fx, fy, cx, cy)."""
        self._infer_anycalib_intrinsics(image)
        return self._anycalib_intrinsics_cache

    def get_points3d_for_mask(self, image: Image.Image, mask: np.ndarray) -> np.ndarray:
        """Backproject UniDepth depth through AnyCalib intrinsics for masked pixels."""
        depth_map = self.get_depth_map(image)
        intrinsics = self.get_intrinsics(image)
        ys, xs = np.where(mask)
        depths = depth_map[ys, xs].astype(np.float32)
        valid = np.isfinite(depths) & (depths > 0.0) & (depths <= 100.0)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        z = depths[valid]
        fx = max(float(intrinsics["fx"]), 1e-6)
        fy = max(float(intrinsics["fy"]), 1e-6)
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        return np.stack([x, y, z], axis=1).astype(np.float32, copy=False)


# ===================================================================
# AnyCalib + MoCA3D — camera intrinsics + per-object cuboid volume
# ===================================================================

class AnyCalibMoCA3DEstimator:
    """Camera intrinsics from AnyCalib and object cuboids from MoCA3D.

    This backend is object-centric rather than dense-depth-centric:
    AnyCalib estimates ``fx, fy, cx, cy`` once per image, and MoCA3D predicts
    projected 3D cuboid corners plus per-corner depths for each tight 2D box.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.anycalib_model = None
        self.moca_model = None
        self._intrinsics_cache: Optional[Dict[str, float]] = None
        self._cache_image_id: Optional[int] = None
        self._sparse_depth_cache: Optional[np.ndarray] = None
        self._last_image_id: Optional[int] = None

    def load(self):
        """Load AnyCalib and MoCA3D models."""
        import sys as _sys
        from pathlib import Path as _Path

        anycalib_root = _Path(self.config.anycalib_root).expanduser()
        moca_root = _Path(self.config.moca3d_root).expanduser()
        if not anycalib_root.exists():
            raise FileNotFoundError(f"AnyCalib root not found: {anycalib_root}")
        if not moca_root.exists():
            raise FileNotFoundError(f"MoCA3D root not found: {moca_root}")

        for root in (str(anycalib_root), str(moca_root)):
            if root not in _sys.path:
                _sys.path.insert(0, root)

        from anycalib import AnyCalib
        from omegaconf import OmegaConf
        from safetensors.torch import load_file as load_safetensors
        from models.moca_3d import Moca3DModel

        device = torch.device(self.config.device)

        print(
            f"  Loading AnyCalib model_id={self.config.anycalib_model_id}, "
            f"cam_id={self.config.anycalib_cam_id}"
        )
        self.anycalib_model = AnyCalib(
            model_id=self.config.anycalib_model_id
        ).to(device).eval()

        config_path = moca_root / "configs" / "MoCA_config.yaml"
        checkpoint_path = _Path(self.config.moca3d_checkpoint_path).expanduser()
        dinov3_path = _Path(self.config.moca3d_dinov3_checkpoint_path).expanduser()
        if not dinov3_path.exists():
            try:
                from huggingface_hub import hf_hub_download

                dinov3_path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"  DINOv3 checkpoint not found at {dinov3_path}; "
                    f"downloading {self.config.moca3d_dinov3_hf_repo}/"
                    f"{self.config.moca3d_dinov3_hf_filename}"
                )
                downloaded_dinov3_path = _Path(hf_hub_download(
                    repo_id=self.config.moca3d_dinov3_hf_repo,
                    filename=self.config.moca3d_dinov3_hf_filename,
                    repo_type="model",
                    local_dir=str(dinov3_path.parent),
                ))
                if downloaded_dinov3_path.suffix == ".safetensors":
                    from safetensors.torch import load_file as load_safetensors

                    print(
                        f"  Converting DINOv3 safetensors to torch checkpoint: "
                        f"{dinov3_path}"
                    )
                    state_dict = load_safetensors(str(downloaded_dinov3_path), device="cpu")
                    torch.save(state_dict, dinov3_path)
                else:
                    dinov3_path = downloaded_dinov3_path
            except Exception as exc:
                raise FileNotFoundError(
                    "DINOv3 checkpoint not found and auto-download failed: "
                    f"{dinov3_path}. Download "
                    f"{self.config.moca3d_dinov3_hf_repo}/"
                    f"{self.config.moca3d_dinov3_hf_filename} and place it there, "
                    "or set MOCA3D_DINOV3_CHECKPOINT_PATH."
                ) from exc
        if not checkpoint_path.exists():
            try:
                from huggingface_hub import hf_hub_download

                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"  MoCA3D checkpoint not found at {checkpoint_path}; "
                    f"downloading {self.config.moca3d_hf_repo}/moca3d.safetensors"
                )
                checkpoint_path = _Path(hf_hub_download(
                    repo_id=self.config.moca3d_hf_repo,
                    filename="moca3d.safetensors",
                    repo_type="model",
                    local_dir=str(checkpoint_path.parent),
                ))
            except Exception as exc:
                raise FileNotFoundError(
                    "MoCA3D checkpoint not found and auto-download failed: "
                    f"{checkpoint_path}. Download {self.config.moca3d_hf_repo}/moca3d.safetensors "
                    "and place it there, or set MOCA3D_CHECKPOINT_PATH."
                ) from exc

        print(f"  Loading MoCA3D from {checkpoint_path}")
        moca_cfg = OmegaConf.load(config_path)
        moca_cfg.device = str(device)
        moca_cfg.feature_mode = False
        moca_cfg.dinov3_checkpoint_path = str(dinov3_path)
        moca_cfg.input_size = float(self.config.moca3d_input_size)
        if "data" in moca_cfg:
            moca_cfg.data.dino_image_size = int(self.config.moca3d_input_size)

        self.moca_model = Moca3DModel(moca_cfg).to(device).eval()
        if checkpoint_path.suffix == ".safetensors":
            state_dict = load_safetensors(str(checkpoint_path), device=str(device))
        else:
            checkpoint_obj = torch.load(checkpoint_path, map_location=device)
            state_dict = self._extract_state_dict(checkpoint_obj)
        state_dict = self._strip_state_prefixes(state_dict)
        self.moca_model.load_state_dict(state_dict, strict=True)

    def unload(self):
        """Free GPU memory."""
        if self.anycalib_model is not None:
            del self.anycalib_model
            self.anycalib_model = None
        if self.moca_model is not None:
            del self.moca_model
            self.moca_model = None
        self._intrinsics_cache = None
        self._cache_image_id = None
        self._sparse_depth_cache = None
        self._last_image_id = None
        torch.cuda.empty_cache()

    @staticmethod
    def _extract_state_dict(checkpoint_obj):
        if isinstance(checkpoint_obj, dict):
            for key in ("state_dict", "model"):
                if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                    return checkpoint_obj[key]
            return {
                key: value for key, value in checkpoint_obj.items()
                if isinstance(value, torch.Tensor)
            }
        raise TypeError(f"Unsupported MoCA3D checkpoint type: {type(checkpoint_obj)}")

    @staticmethod
    def _strip_state_prefixes(state_dict):
        prefixes = (
            "module.", "_orig_mod.", "model.", "moca_model.", "moca.",
            "joint_model.moca_model.",
        )
        stripped = {}
        for key, value in state_dict.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            stripped[new_key] = value
        return stripped

    def _ensure_loaded(self):
        if self.anycalib_model is None or self.moca_model is None:
            raise RuntimeError("AnyCalibMoCA3DEstimator.load() must be called first.")

    def _infer_intrinsics(self, image: Image.Image):
        self._ensure_loaded()
        img_id = id(image)
        if self._cache_image_id == img_id and self._intrinsics_cache is not None:
            return

        arr = np.array(image.convert("RGB"), copy=True)
        image_tensor = (
            torch.tensor(arr, dtype=torch.float32, device=self.config.device)
            .permute(2, 0, 1)
            / 255.0
        )

        with torch.inference_mode():
            output = self.anycalib_model.predict(
                image_tensor, cam_id=self.config.anycalib_cam_id
            )
        intr = output["intrinsics"]
        if isinstance(intr, list):
            intr = intr[0]
        intr_np = intr.detach().float().cpu().numpy()
        if len(intr_np) >= 4:
            fx, fy, cx, cy = intr_np[:4]
        elif len(intr_np) == 3:
            fx, cx, cy = intr_np
            fy = fx
        else:
            raise ValueError(
                f"Unsupported AnyCalib intrinsics shape for "
                f"{self.config.anycalib_cam_id}: {intr_np.shape}"
            )

        self._intrinsics_cache = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
        self._cache_image_id = img_id
        self._last_image_id = img_id
        self._sparse_depth_cache = np.zeros((image.height, image.width), dtype=np.float32)

    def get_intrinsics(self, image: Image.Image, **kwargs) -> Dict[str, float]:
        """Get AnyCalib-estimated pinhole intrinsics (fx, fy, cx, cy)."""
        self._infer_intrinsics(image)
        return self._intrinsics_cache

    def get_depth_map(self, image: Image.Image, **kwargs) -> np.ndarray:
        """Return a sparse visualization depth map.

        MoCA3D does not predict dense depth. During object inference this cache
        is filled with each object's median cuboid-corner depth over its mask.
        """
        if self._last_image_id == id(image) and self._sparse_depth_cache is not None:
            return self._sparse_depth_cache
        self._last_image_id = id(image)
        self._sparse_depth_cache = np.zeros((image.height, image.width), dtype=np.float32)
        return self._sparse_depth_cache

    def estimate_depths(
        self,
        image: Image.Image,
        points: List[Tuple[int, int]],
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> List[Optional[float]]:
        depth_map = self.get_depth_map(image)
        h, w = depth_map.shape
        depths = []
        for x, y in points:
            if 0 <= y < h and 0 <= x < w:
                val = float(depth_map[y, x])
                depths.append(val if 0.01 <= val <= 100.0 else None)
            else:
                depths.append(None)
        return depths

    def estimate_object_geometry(
        self,
        image: Image.Image,
        box: np.ndarray,
        mask: Optional[np.ndarray] = None,
        intrinsics: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Predict one object's 3D cuboid and volume from image + tight bbox."""
        self._ensure_loaded()
        intrinsics = intrinsics or self.get_intrinsics(image)
        image_tensor, box_tensor, padding_mask, geom = self._prepare_moca_inputs(image, box)

        with torch.inference_mode():
            outputs = self.moca_model(
                images_dino=image_tensor,
                bbx2d_tight=box_tensor,
                mask=padding_mask,
            )

        coords_512 = outputs["corner coords"][0].detach().float().cpu().numpy()
        virtual_depths = outputs["sampled depths"][0].detach().float().cpu().numpy()
        heatmaps = outputs.get("corner heatmaps")
        if heatmaps is not None:
            corner_conf = heatmaps[0].detach().float().flatten(1).amax(dim=1).cpu().numpy()
        else:
            corner_conf = None

        pred_uv = self._moca_coords_to_original(coords_512, geom, image.size)
        real_depths = self._virtual_depths_to_metric(
            virtual_depths, intrinsics=intrinsics, image_height=image.height
        )
        points_3d = self._backproject_corners(pred_uv, real_depths, intrinsics)
        valid = np.isfinite(points_3d).all(axis=1) & (points_3d[:, 2] > 0.0) & (points_3d[:, 2] <= 100.0)
        if int(valid.sum()) < 4:
            raise RuntimeError(f"MoCA3D produced insufficient valid corners ({int(valid.sum())}/8).")

        rectified, dimensions = self._rectify_cuboid_kabsch(points_3d, corner_conf)
        volume_m3 = float(np.prod(dimensions))
        center = rectified.mean(axis=0)

        if mask is not None:
            self._update_sparse_depth_cache(image, mask, float(np.median(real_depths[valid])))

        return {
            "points_3d": points_3d.astype(np.float32, copy=False),
            "obb_geometry": {
                "center": center.astype(np.float32, copy=False),
                "extents": dimensions.astype(np.float32, copy=False),
                "corners": rectified.astype(np.float32, copy=False),
                "points_3d": points_3d.astype(np.float32, copy=False),
            },
            "volume_info": {
                "volume_m3": volume_m3,
                "volume_cm3": volume_m3 * 1e6,
                "dimensions_m": dimensions.astype(float).tolist(),
                "correction_applied": 1.0,
                "estimation_mode": "anycalib_moca3d",
                "moca3d_projected_corners": pred_uv.astype(float).tolist(),
                "moca3d_corner_depths_m": real_depths.astype(float).tolist(),
                "moca3d_virtual_depths": virtual_depths.astype(float).tolist(),
            },
        }

    def scale_object_geometry(self, geometry_result: Dict, scale_factor: float) -> Dict:
        """Apply marker scale to a MoCA3D geometry result."""
        if scale_factor == 1.0:
            return geometry_result
        scaled = dict(geometry_result)
        scaled["points_3d"] = geometry_result["points_3d"] * scale_factor
        obb = dict(geometry_result["obb_geometry"])
        obb["center"] = obb["center"] * scale_factor
        obb["extents"] = obb["extents"] * scale_factor
        obb["corners"] = obb["corners"] * scale_factor
        obb["points_3d"] = obb["points_3d"] * scale_factor
        scaled["obb_geometry"] = obb

        volume_info = dict(geometry_result["volume_info"])
        dims = np.asarray(volume_info["dimensions_m"], dtype=float) * scale_factor
        volume_info["dimensions_m"] = dims.tolist()
        volume_info["volume_m3"] = float(np.prod(dims))
        volume_info["volume_cm3"] = volume_info["volume_m3"] * 1e6
        scaled["volume_info"] = volume_info
        return scaled

    def _prepare_moca_inputs(self, image: Image.Image, box: np.ndarray):
        input_size = int(self.config.moca3d_input_size)
        image_rgb = image.convert("RGB")
        w, h = image_rgb.size
        longest = max(w, h)
        scale = input_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pad_left = (input_size - new_w) // 2
        pad_top = (input_size - new_h) // 2

        arr = np.asarray(image_rgb)
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
        padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        tensor = torch.from_numpy(padded.copy()).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        mean = tensor.new_tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(self.IMAGENET_STD).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.config.device)

        box_arr = np.asarray(box, dtype=np.float32).copy()
        box_arr[[0, 2]] = box_arr[[0, 2]] * scale + pad_left
        box_arr[[1, 3]] = box_arr[[1, 3]] * scale + pad_top
        box_arr[0::2] = np.clip(box_arr[0::2], 0.0, float(input_size))
        box_arr[1::2] = np.clip(box_arr[1::2], 0.0, float(input_size))
        if box_arr[2] <= box_arr[0] + 1.0 or box_arr[3] <= box_arr[1] + 1.0:
            raise ValueError(f"Invalid MoCA3D bbox after letterbox transform: {box_arr.tolist()}")
        box_tensor = torch.from_numpy(box_arr / float(input_size)).float().unsqueeze(0).to(self.config.device)

        padding_mask = torch.ones((1, input_size, input_size), dtype=torch.bool, device=self.config.device)
        padding_mask[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = False

        geom = {
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "input_size": input_size,
        }
        return tensor, box_tensor, padding_mask, geom

    @staticmethod
    def _moca_coords_to_original(coords_512: np.ndarray, geom: Dict, image_size: Tuple[int, int]) -> np.ndarray:
        w, h = image_size
        coords = np.asarray(coords_512, dtype=np.float32).copy()
        coords[:, 0] = (coords[:, 0] - float(geom["pad_left"])) / float(geom["scale"])
        coords[:, 1] = (coords[:, 1] - float(geom["pad_top"])) / float(geom["scale"])
        coords[:, 0] = np.clip(coords[:, 0], 0.0, max(float(w - 1), 0.0))
        coords[:, 1] = np.clip(coords[:, 1], 0.0, max(float(h - 1), 0.0))
        return coords

    @staticmethod
    def _virtual_depths_to_metric(
        virtual_depths: np.ndarray,
        intrinsics: Dict[str, float],
        image_height: int,
    ) -> np.ndarray:
        # MoCA3D trains depth in the virtual camera scale:
        # z_virtual = z_real * (512 / fy) * (H / 512) = z_real * H / fy.
        fy = max(float(intrinsics["fy"]), 1e-6)
        return np.asarray(virtual_depths, dtype=np.float32) * (fy / max(float(image_height), 1.0))

    @staticmethod
    def _backproject_corners(
        uv: np.ndarray,
        depths: np.ndarray,
        intrinsics: Dict[str, float],
    ) -> np.ndarray:
        u = uv[:, 0].astype(np.float32)
        v = uv[:, 1].astype(np.float32)
        z = np.asarray(depths, dtype=np.float32)
        fx = max(float(intrinsics["fx"]), 1e-6)
        fy = max(float(intrinsics["fy"]), 1e-6)
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _rectify_cuboid_kabsch(points_3d: np.ndarray, weights: Optional[np.ndarray] = None):
        """Fit a valid cuboid to MoCA3D's 8 predicted 3D corners."""
        points = np.asarray(points_3d, dtype=np.float64)
        if points.shape != (8, 3):
            raise ValueError(f"Expected 8 MoCA3D corners, got {points.shape}")

        canonical = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ], dtype=np.float64)

        if weights is None:
            w = np.ones(8, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w - np.nanmin(w)
            w = np.maximum(w, 1e-3)
        w = w / max(float(w.sum()), 1e-6)
        w3 = w[:, None]

        mu_p = (w3 * points).sum(axis=0, keepdims=True)
        mu_v = (w3 * canonical).sum(axis=0, keepdims=True)
        x = points - mu_p
        y = canonical - mu_v

        h = (w3 * y).T @ x
        u, _, vh = np.linalg.svd(h)
        r = u @ vh
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1.0
            r = u @ vh

        x_local = x @ r.T
        denom = (w3 * (y ** 2)).sum(axis=0) + 1e-6
        numer = (w3 * y * x_local).sum(axis=0)
        dimensions = np.maximum(np.abs(numer / denom), 0.01)
        rectified = (y * dimensions[None, :]) @ r + mu_p
        return rectified.astype(np.float32), dimensions.astype(np.float32)

    def _update_sparse_depth_cache(self, image: Image.Image, mask: np.ndarray, depth: float):
        if self._last_image_id != id(image) or self._sparse_depth_cache is None:
            self._last_image_id = id(image)
            self._sparse_depth_cache = np.zeros((image.height, image.width), dtype=np.float32)
        if mask.shape == self._sparse_depth_cache.shape and np.isfinite(depth) and depth > 0:
            self._sparse_depth_cache[mask.astype(bool)] = float(depth)
