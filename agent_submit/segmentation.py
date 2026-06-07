"""Segmentation wrappers for text-prompted instance segmentation.

Supports two backends:
  1. SAM3 (facebook/sam3) — text-prompted segmentation (DEFAULT)
  2. OpenWorldSAM2 — open-vocabulary segmentation with cross-attention
"""

import sys
import types
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

from config import PipelineConfig

# Erosion kernel size (pixels) to shrink mask boundaries.
# Removes edge pixels where segmentation bleeds into background,
# preventing depth contamination from neighboring surfaces.
_MASK_EROSION_PX = 5


@dataclass
class SegmentationResult:
    """Result of segmenting a single object class."""
    mask: np.ndarray       # (H, W) bool
    box: np.ndarray        # (4,) [x0, y0, x1, y1]
    score: float
    object_name: str


def _erode_mask(mask: np.ndarray, px: int = _MASK_EROSION_PX) -> np.ndarray:
    """Erode a boolean mask inward by `px` pixels to remove boundary noise.

    Uses OpenCV (SIMD-optimized) when available — orders of magnitude faster
    than PIL MinFilter for large masks. Falls back to PIL otherwise.
    If erosion would eliminate the mask entirely, returns the original.
    """
    kernel_size = px * 2 + 1
    if _cv2 is not None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = _cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        eroded = np.array(pil_mask.filter(ImageFilter.MinFilter(kernel_size))) > 0
    if eroded.sum() < 100:  # too small after erosion — keep original
        return mask
    return eroded


def _patch_pkg_resources():
    """Ensure pkg_resources is available (needed by sam3.model_builder)."""
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        import importlib.resources
        pkg = types.ModuleType("pkg_resources")

        def resource_filename(package: str, resource: str) -> str:
            return str(importlib.resources.files(package).joinpath(resource))

        pkg.resource_filename = resource_filename
        sys.modules["pkg_resources"] = pkg


class SAM3Segmenter:
    """Text-prompted segmentation using SAM3 (PCS mode)."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.processor = None

    def load(self):
        """Load SAM3 model and processor."""
        _patch_pkg_resources()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print(f"  Loading SAM3 from {self.config.sam3_checkpoint}")
        self.model = build_sam3_image_model(
            bpe_path=self.config.sam3_bpe_path,
            device=self.config.device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=self.config.sam3_resolution,
            device=self.config.device,
            confidence_threshold=self.config.sam3_confidence_threshold,
        )

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        return_all: bool = False,
    ) -> Optional[List[SegmentationResult]]:
        """Segment objects matching text_prompt in the image.

        Args:
            image: PIL RGB image
            text_prompt: English object name (e.g., "desk")
            return_all: if True, return all instances; otherwise only the best

        Returns:
            List of SegmentationResult, or None if nothing detected.
        """
        state = self.processor.set_image(image)
        state = self.processor.set_text_prompt(text_prompt, state)

        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

        if masks is None or len(masks) == 0:
            return None

        # Convert to numpy
        masks_np = masks.cpu().numpy().astype(bool)   # (N, H, W) or (N, 1, H, W)
        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(1)
        boxes_np = boxes.cpu().numpy()                 # (N, 4)
        scores_np = scores.cpu().numpy()               # (N,)

        results = []
        for i in range(len(scores_np)):
            results.append(SegmentationResult(
                mask=_erode_mask(masks_np[i]),
                box=boxes_np[i],
                score=float(scores_np[i]),
                object_name=text_prompt,
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        if return_all:
            return results
        return [results[0]]  # Best detection only

    def segment_multiple(
        self,
        image: Image.Image,
        object_names: List[str],
    ) -> List[Optional[List[SegmentationResult]]]:
        """Segment multiple object types. Re-uses the same image encoding."""
        state = self.processor.set_image(image)
        all_results = []

        for name in object_names:
            # Reset prompts and re-use backbone features
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(name, state)

            masks = state.get("masks")
            boxes = state.get("boxes")
            scores = state.get("scores")

            if masks is None or len(masks) == 0:
                all_results.append(None)
                continue

            masks_np = masks.cpu().numpy().astype(bool)
            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)
            boxes_np = boxes.cpu().numpy()
            scores_np = scores.cpu().numpy()

            results = []
            for i in range(len(scores_np)):
                results.append(SegmentationResult(
                    mask=_erode_mask(masks_np[i]),
                    box=boxes_np[i],
                    score=float(scores_np[i]),
                    object_name=name,
                ))
            results.sort(key=lambda r: r.score, reverse=True)
            all_results.append(results)

        return all_results


# ===================================================================
# OpenWorldSAM2 — open-vocabulary segmentation
# ===================================================================

# COCO thing class names (80 classes) for category_id mapping
_COCO_THING_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Mapping from common furniture/object names to COCO category IDs
_COCO_NAME_TO_ID = {name: i for i, name in enumerate(_COCO_THING_CLASSES)}
# Add common aliases
_COCO_NAME_TO_ID.update({
    "sofa": _COCO_NAME_TO_ID["couch"],
    "desk": _COCO_NAME_TO_ID["dining table"],
    "table": _COCO_NAME_TO_ID["dining table"],
    "fridge": _COCO_NAME_TO_ID["refrigerator"],
    "monitor": _COCO_NAME_TO_ID["tv"],
    "television": _COCO_NAME_TO_ID["tv"],
    "armchair": _COCO_NAME_TO_ID["chair"],
    "bookshelf": _COCO_NAME_TO_ID["book"],  # closest match
    "cabinet": _COCO_NAME_TO_ID["refrigerator"],  # approximate
    "wardrobe": _COCO_NAME_TO_ID["refrigerator"],  # approximate
    "washing machine": _COCO_NAME_TO_ID["oven"],  # approximate
    "flower pot": _COCO_NAME_TO_ID["potted plant"],
    "flowerpot": _COCO_NAME_TO_ID["potted plant"],
    "plant": _COCO_NAME_TO_ID["potted plant"],
    "stand": _COCO_NAME_TO_ID["bottle"],  # fallback - no lamp in COCO things
    "lamp": _COCO_NAME_TO_ID["bottle"],   # fallback
    "computer": _COCO_NAME_TO_ID["laptop"],
    "desktop": _COCO_NAME_TO_ID["laptop"],
    "air purifier": _COCO_NAME_TO_ID["bottle"],  # fallback
    "water purifier": _COCO_NAME_TO_ID["bottle"],  # fallback
    "microwave": _COCO_NAME_TO_ID["microwave"],
    "dresser": _COCO_NAME_TO_ID["refrigerator"],  # approximate
    "drawer": _COCO_NAME_TO_ID["refrigerator"],   # approximate
})


class OpenWorldSAMSegmenter:
    """Text-prompted segmentation using OpenWorldSAM2 (instance mode)."""

    OWSAM_ROOT = "/home/irteam/data-vol1/OpenWorldSAM"
    CONFIG_FILE = "configs/coco/instance-segmentation/Open-World-SAM2-CrossAttention.yaml"
    WEIGHTS_FILE = "checkpoints/model_final.pth"

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self._cfg = None

    def load(self):
        """Load OpenWorldSAM2 model."""
        import os

        # Temporarily adjust sys.path and sys.modules to avoid shadowing
        # OpenWorldSAM's own 'utils' and 'datasets' packages with agent's.
        self._orig_cwd = os.getcwd()
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        self._removed_paths = []
        for p in list(sys.path):
            if os.path.abspath(p) == agent_dir:
                sys.path.remove(p)
                self._removed_paths.append(p)

        # Remove cached agent modules that shadow OWSAM packages
        saved_modules = {}
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("utils", "datasets") or mod_name.startswith(("utils.", "datasets.")):
                saved_modules[mod_name] = sys.modules.pop(mod_name)

        if self.OWSAM_ROOT not in sys.path:
            sys.path.insert(0, self.OWSAM_ROOT)

        # Change to OWSAM root so hydra can find config files
        os.chdir(self.OWSAM_ROOT)

        from demo.inference_utils import setup_cfg, load_model

        print(f"  Loading OpenWorldSAM2 from {self.OWSAM_ROOT}")
        self._cfg = setup_cfg(
            config_file=self.CONFIG_FILE,
            weights=self.WEIGHTS_FILE,
            device=self.config.device,
        )
        self._cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = True
        self._cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False
        self._cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = False
        self._cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = False
        self._cfg.MODEL.OpenWorldSAM2.TEST.NMS_THRESHOLD = 0.5
        self._cfg.MODEL.OpenWorldSAM2.TEST.IOU_THRESHOLD = 0.4

        self.model = load_model(self._cfg)

        # Restore paths and cached modules
        os.chdir(self._orig_cwd)
        for p in self._removed_paths:
            if p not in sys.path:
                sys.path.insert(0, p)
        # Restore agent modules that were temporarily removed
        for mod_name, mod in saved_modules.items():
            if mod_name not in sys.modules:
                sys.modules[mod_name] = mod

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._cfg = None
        torch.cuda.empty_cache()

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        return_all: bool = False,
    ) -> Optional[List[SegmentationResult]]:
        """Segment objects matching text_prompt in the image.

        Args:
            image: PIL RGB image
            text_prompt: English object name (e.g., "desk")
            return_all: if True, return all instances; otherwise only the best

        Returns:
            List of SegmentationResult, or None if nothing detected.
        """
        from demo.inference_utils import sam_preprocess, beit3_preprocess, build_inference_inputs

        # Convert PIL to numpy (RGB)
        image_np = np.array(image)

        # Preprocess
        sam_tensor = sam_preprocess(image_np)
        beit_tensor = beit3_preprocess(image_np)
        h, w = image_np.shape[:2]

        # Resolve category ID
        prompt_lower = text_prompt.lower().strip()
        category_id = _COCO_NAME_TO_ID.get(prompt_lower, 0)

        # Build inputs and run
        inputs = build_inference_inputs(
            sam_tensor, beit_tensor, h, w,
            [text_prompt], [category_id],
        )

        with torch.no_grad():
            outputs = self.model(inputs)[0]

        instances = outputs.get("instances")
        if instances is None or len(instances) == 0:
            return None

        # Convert to SegmentationResult
        masks_np = instances.pred_masks.cpu().numpy().astype(bool)  # (N, H, W)
        boxes_np = instances.pred_boxes.tensor.cpu().numpy()        # (N, 4)
        scores_np = instances.scores.cpu().numpy()                  # (N,)

        results = []
        for i in range(len(scores_np)):
            results.append(SegmentationResult(
                mask=_erode_mask(masks_np[i]),
                box=boxes_np[i],
                score=float(scores_np[i]),
                object_name=text_prompt,
            ))

        results.sort(key=lambda r: r.score, reverse=True)

        if return_all:
            return results
        return [results[0]]

    def segment_all(
        self,
        image: Image.Image,
        classes: Optional[List[str]] = None,
        score_threshold: float = 0.3,
        batch_size: int = 8,
        dedup_iou: float = 0.6,
    ) -> List[SegmentationResult]:
        """Detect every instance across a class vocabulary.

        Runs the model in batches of ``batch_size`` classes to keep GPU memory
        bounded (each prompt spawns ``num_tokens`` transformer queries, so 80
        classes in one forward pass can easily exceed tens of GB). Results are
        de-duplicated across batches via mask-IoU NMS since within-batch NMS
        cannot see detections from other batches.

        Args:
            image: PIL RGB image.
            classes: Class vocabulary. Defaults to the 80 COCO thing classes.
            score_threshold: Minimum score to keep a detection.
            batch_size: Number of class prompts per forward pass.
            dedup_iou: Mask-IoU threshold for cross-batch de-duplication.

        Returns:
            List of SegmentationResult sorted by score.
        """
        from demo.inference_utils import sam_preprocess, beit3_preprocess, build_inference_inputs

        target_classes = list(classes) if classes else list(_COCO_THING_CLASSES)
        category_ids = []
        for i, name in enumerate(target_classes):
            cid = _COCO_NAME_TO_ID.get(name.lower().strip())
            category_ids.append(cid if cid is not None else i)

        image_np = np.array(image)
        sam_tensor = sam_preprocess(image_np)
        beit_tensor = beit3_preprocess(image_np)
        h, w = image_np.shape[:2]

        id_to_name = {cid: name for name, cid in zip(target_classes, category_ids)}
        for i, coco_name in enumerate(_COCO_THING_CLASSES):
            id_to_name.setdefault(i, coco_name)

        raw_results: List[SegmentationResult] = []
        num_batches = (len(target_classes) + batch_size - 1) // batch_size
        print(
            f"  OpenWorldSAM scan: {len(target_classes)} classes in "
            f"{num_batches} batch(es) of {batch_size}"
        )
        for b in range(num_batches):
            lo = b * batch_size
            hi = min(lo + batch_size, len(target_classes))
            batch_prompts = target_classes[lo:hi]
            batch_ids = category_ids[lo:hi]

            inputs = build_inference_inputs(
                sam_tensor, beit_tensor, h, w,
                batch_prompts, batch_ids,
            )
            with torch.no_grad():
                outputs = self.model(inputs)[0]
            instances = outputs.get("instances")
            batch_kept = 0
            if instances is None or len(instances) == 0:
                del outputs
                torch.cuda.empty_cache()
                print(f"    [batch {b+1}/{num_batches}] → 0 detections")
                continue

            masks_np = instances.pred_masks.cpu().numpy().astype(bool)
            boxes_np = instances.pred_boxes.tensor.cpu().numpy()
            scores_np = instances.scores.cpu().numpy()
            pred_cls_np = (
                instances.pred_classes.cpu().numpy()
                if instances.has("pred_classes")
                else np.zeros(len(scores_np), dtype=int)
            )
            del outputs, instances
            torch.cuda.empty_cache()

            # Defer expensive mask erosion until after NMS so we don't erode
            # detections that will be dropped as duplicates.
            for i in range(len(scores_np)):
                if scores_np[i] < score_threshold:
                    continue
                cls_id = int(pred_cls_np[i])
                obj_name = id_to_name.get(cls_id, f"class_{cls_id}")
                raw_results.append(SegmentationResult(
                    mask=masks_np[i],
                    box=boxes_np[i],
                    score=float(scores_np[i]),
                    object_name=obj_name,
                ))
                batch_kept += 1
            print(f"    [batch {b+1}/{num_batches}] → {batch_kept} detections above threshold")

        # Cross-batch NMS via bbox IoU. OpenWorldSAM derives boxes from the
        # mask (``bit_masks.get_bounding_boxes()``), so boxes are tight and
        # bbox IoU is a good proxy for mask IoU — O(n²) cheap scalar ops
        # instead of O(n²) full-resolution boolean ops.
        print(f"  NMS across {len(raw_results)} raw detections...")
        raw_results.sort(key=lambda r: r.score, reverse=True)

        def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
            ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
            ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            inter = (ix1 - ix0) * (iy1 - iy0)
            a_area = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
            b_area = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
            union = a_area + b_area - inter
            return float(inter / union) if union > 0 else 0.0

        kept: List[SegmentationResult] = []
        for r in raw_results:
            if any(_bbox_iou(r.box, k.box) >= dedup_iou for k in kept):
                continue
            kept.append(r)
        print(f"  NMS kept {len(kept)}/{len(raw_results)} detections")

        # Now erode only the survivors (cv2-backed when available).
        print(f"  Eroding {len(kept)} mask(s) (engine={'cv2' if _cv2 is not None else 'PIL'})...")
        for r in kept:
            r.mask = _erode_mask(r.mask)
        print(f"  Eroded {len(kept)} mask(s).")
        return kept


class FallbackSegmenter:
    """Try SAM3 first; if it fails to detect, fall back to OpenWorldSAM."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.primary = SAM3Segmenter(self.config)
        self.fallback = OpenWorldSAMSegmenter(self.config)

    def load(self):
        self.primary.load()
        self.fallback.load()

    def unload(self):
        self.primary.unload()
        self.fallback.unload()

    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        return_all: bool = False,
    ) -> Optional[List[SegmentationResult]]:
        result = self.primary.segment(image, text_prompt, return_all=return_all)
        if result is not None:
            return result
        print(f"  SAM3 failed for '{text_prompt}', trying OpenWorldSAM...")
        return self.fallback.segment(image, text_prompt, return_all=return_all)

    def segment_all(
        self,
        image: Image.Image,
        classes: Optional[List[str]] = None,
        score_threshold: float = 0.3,
    ) -> List[SegmentationResult]:
        """Delegate class-agnostic detection to OpenWorldSAM (SAM3 lacks this)."""
        return self.fallback.segment_all(
            image, classes=classes, score_threshold=score_threshold,
        )
