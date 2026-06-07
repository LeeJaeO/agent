"""Main pipeline: Korean text + indoor image → object volume estimation + logistics recommendation."""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from text_parser import KoreanTextParser
from segmentation import SAM3Segmenter, OpenWorldSAMSegmenter, FallbackSegmenter
from depth_estimation import (
    DepthAnythingV2Estimator,
    UniDepthV2Estimator, AnyCalibUniDepthEstimator, AnyCalibMoCA3DEstimator,
)
from shape_priors import AutomaticShapePriorEstimator
from standard_size_agent import StandardSizeAgent
from volume_calculator import VolumeCalculator
from utils import (
    estimate_camera_intrinsics,
    generate_logistics_recommendation,
    visualize_results,
    visualize_3d_obb_on_image,
    visualize_3d_pointcloud,
    visualize_depth_map,
    visualize_per_object_mask,
    visualize_object_depth,
    visualize_objects_on_depth,
    visualize_single_object_3d,
)


def _create_segmenter(config: PipelineConfig):
    """Factory: create the segmenter based on config."""
    if config.segmentation_backend == "openworldsam":
        return OpenWorldSAMSegmenter(config)
    if config.segmentation_backend == "sam3_fallback":
        return FallbackSegmenter(config)
    return SAM3Segmenter(config)


def _create_depth_estimator(config: PipelineConfig):
    """Factory: create the depth estimator based on config."""
    if config.depth_backend == "anycalib_moca3d":
        return AnyCalibMoCA3DEstimator(config)
    if config.depth_backend == "anycalib_unidepth":
        return AnyCalibUniDepthEstimator(config)
    if config.depth_backend == "unidepth":
        return UniDepthV2Estimator(config)
    if config.depth_backend == "depth_anything_v2":
        return DepthAnythingV2Estimator(config)
    return UniDepthV2Estimator(config)


def _mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )


def _marker_mask_metrics(mask: np.ndarray, box: np.ndarray) -> dict:
    h, w = mask.shape
    mask_area = float(mask.sum())
    x0, y0, x1, y1 = box.astype(float)
    bbox_w = max(x1 - x0, 1.0)
    bbox_h = max(y1 - y0, 1.0)
    bbox_area = bbox_w * bbox_h

    oriented_aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1.0)
    ys, xs = np.where(mask)
    if len(xs) >= 8:
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        coords -= coords.mean(axis=0, keepdims=True)
        cov = np.cov(coords.T)
        _, eigvecs = np.linalg.eigh(cov)
        projected = coords @ eigvecs
        extents = np.maximum(projected.max(axis=0) - projected.min(axis=0), 1.0)
        oriented_aspect = float(max(extents) / max(min(extents), 1.0))

    return {
        "mask_area": mask_area,
        "area_ratio": mask_area / max(float(h * w), 1.0),
        "bbox_area_ratio": bbox_area / max(float(h * w), 1.0),
        "rectangularity": mask_area / max(bbox_area, 1.0),
        "oriented_aspect_ratio": oriented_aspect,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
    }


def _make_instance_name(base_name: str, prompt_count: int, total_for_prompt: int) -> str:
    if total_for_prompt == 1 and prompt_count == 0:
        return base_name
    return f"{base_name}_{prompt_count + 1}"


_PRESENCE_ONLY_COUNT_CATEGORIES = {
    # These are pile/bulk categories in moving estimates. VLM count usually
    # means "this category is present", not "keep exactly one SAM instance".
    "clothes",
    "clothing",
    "blanket",
    "blankets",
    "dishes",
    "utensils",
    "miscellaneous",
    "miscellaneous items",
    "small appliance",
    "small appliances",
}


def _limit_segments_to_expected_count(seg_results, object_name: str, expected_counts: dict | None):
    """Keep only the top-scoring instances allowed by the VLM count."""
    if seg_results is None:
        return None
    if not expected_counts:
        return seg_results
    if str(object_name).strip().lower() in _PRESENCE_ONLY_COUNT_CATEGORIES:
        return seg_results
    expected = expected_counts.get(object_name)
    if expected is None:
        return seg_results
    expected = max(0, int(expected))
    if expected == 0:
        return None
    limited = seg_results[:expected]
    return limited if limited else None


# Truck capacity table: name → max CBM the truck can carry.
# Pick the smallest truck whose capacity is ≥ the actual CBM (raw, unfloored)
# so we never undersize. Floor of the CBM is used for display only.
_TRUCK_CAPACITIES = [
    ("1톤", 6),
    ("1.4톤", 8),
    ("2.5톤", 14),
    ("3.5톤", 17),
    ("5톤", 26),
]

# Display truck name → rule-table key (retpdf/map/quote_rules.json).
# Rule table doesn't have 1.4톤 or 3.5톤 keys, so we map them to the closest
# larger rule key (safer to overestimate price than to undersize).
_TRUCK_TO_RULE_KEY = {
    "1톤": "1톤",
    "1.4톤": "2.5톤",
    "2.5톤": "2.5톤",
    "3.5톤": "3-4톤",
    "5톤": "5톤",
}


def _recommend_truck(total_cbm: float) -> tuple[int, str]:
    """Floor CBM for display; pick smallest truck whose capacity ≥ raw CBM."""
    cbm_int = int(total_cbm)  # display floor (10.77 → 10)
    for name, capacity in _TRUCK_CAPACITIES:
        if total_cbm <= capacity:
            return cbm_int, name
    return cbm_int, "5톤"


# English (detected) category → Korean display name. Covers SAM3-detected
# items beyond COCO 80 (vanity, blanket, mattress, ...) plus the COCO classes
# that OpenWorldSAM emits for --detect-all.
_CATEGORY_KO_MAP = {
    # Electronics
    "TV": "TV", "tv": "TV", "monitor": "모니터", "laptop": "노트북",
    "computer": "컴퓨터", "desktop": "데스크탑", "cell phone": "휴대폰",
    "remote": "리모컨", "keyboard": "키보드", "mouse": "마우스",
    # Furniture
    "chair": "의자", "couch": "소파", "sofa": "소파", "armchair": "안락의자",
    "bed": "침대", "mattress": "매트리스", "dining table": "식탁",
    "table": "테이블", "desk": "책상", "dresser": "서랍장", "vanity": "화장대",
    "wardrobe": "옷장", 
    # "storage cabinet": "수납장", "cabinet": "장",
    "drawer": "서랍", "bookshelf": "책장", "shelf": "선반",
    "nightstand": "협탁",
    # Appliances
    "refrigerator": "냉장고", "fridge": "냉장고", "microwave": "전자레인지",
    "oven": "오븐", "toaster": "토스터", "sink": "싱크대",
    "washing machine": "세탁기", "dishwasher": "식기세척기",
    "fan": "선풍기", "air conditioner": "에어컨",
    "air purifier": "공기청정기", "water purifier": "정수기",
    # Bathroom
    "toilet": "변기", "bathtub": "욕조",
    # Decor / textile
    "curtain": "커튼", "blanket": "이불", "pillow": "베개",
    "potted plant": "화분", "plant": "화분", "vase": "꽃병",
    "clock": "시계", "lamp": "스탠드 조명", "stand": "스탠드",
    "mirror": "거울", "rug": "러그",
    # Misc
    "book": "책", "bottle": "병", "wine glass": "와인잔", "cup": "컵",
    "bowl": "그릇", "fork": "포크", "knife": "칼", "spoon": "숟가락",
    "scissors": "가위", "hair drier": "헤어드라이어", "toothbrush": "칫솔",
    "teddy bear": "곰인형", "backpack": "백팩", "handbag": "핸드백",
    "suitcase": "캐리어", "umbrella": "우산", "tie": "넥타이",
    "person": "사람", "bicycle": "자전거", "car": "자동차",
    "motorcycle": "오토바이", "boat": "보트", "bench": "벤치",
    "bird": "새", "cat": "고양이", "dog": "개",
}


def _translate_categories_ko(categories) -> list:
    """Map English detected categories to Korean display names.

    Falls back to the original string when no mapping is registered so the
    user can see which tokens still need translation.
    """
    out = []
    for cat in categories:
        cat_str = str(cat).strip()
        ko = _CATEGORY_KO_MAP.get(cat_str) or _CATEGORY_KO_MAP.get(cat_str.lower())
        out.append(ko if ko else cat_str)
    return out


# Fields prompted from the user. Order matters for UX (most-needed first).
_CUSTOMER_PROMPTS = [
    ("customer_name",        "고객명"),
    ("customer_phone",       "전화번호 (예: 010-1234-5678)"),
    ("origin_address",       "출발지 주소"),
    ("inbound_date",         "입고일 (예: 2026.06.29 또는 '다음 주 월요일')"),
    ("inbound_start",        "입고 시간대 (오전/오후)"),
    ("destination_address",  "도착지 주소"),
    ("outbound_date",        "출고일"),
    ("outbound_start",       "출고 시간대 (오전/오후)"),
    ("inbound_work_type",    "입고 작업 (1층작업/사다리차작업/계단/반지하/E/V작업)"),
    ("outbound_work_type",   "출고 작업"),
    ("request_text",         "특이사항 (없으면 엔터)"),
]


def _collect_customer_info_interactive(parser) -> dict | None:
    """Prompt the user for customer fields, then ask Qwen to normalize them.

    Returns ``None`` if stdin isn't a TTY (e.g., running non-interactively),
    so the caller can fall back to placeholder data without hanging.
    """
    if not sys.stdin.isatty():
        print("⚠ 비대화형 환경이라 customer 정보 입력 스킵 — placeholder 사용.")
        return None

    print("\n=== 고객 정보 입력 (모두 채우세요) ===")
    raw = {}
    for key, label in _CUSTOMER_PROMPTS:
        try:
            raw[key] = input(f"  {label}: ").strip()
        except EOFError:
            print("  (EOF) — 이후 필드는 placeholder 사용")
            return None

    from datetime import date
    today = date.today().strftime("%Y.%m.%d")
    prompt = (
        "다음 견적서 고객 정보를 표준 양식으로 정규화하세요. JSON 객체 하나로만 응답하세요.\n\n"
        f"입력:\n{json.dumps(raw, ensure_ascii=False, indent=2)}\n\n"
        "규칙:\n"
        "- customer_phone: '010-1234-5678' 형식 (하이픈 두 개, 숫자만)\n"
        f"- inbound_date / outbound_date: 'YYYY.MM.DD' 형식. 상대 표현(다음 주 월요일 등)은 오늘({today}) 기준으로 환산\n"
        "- inbound_start / outbound_start: 정확히 '오전' 또는 '오후'\n"
        "- inbound_work_type / outbound_work_type: 정확히 '1층작업', '사다리차작업', '계단/반지하', 'E/V작업' 중 하나\n"
        "- request_text: 빈 값이면 '없음'으로\n"
        "- customer_name, origin_address, destination_address: 그대로 유지하되 공백 정리\n\n"
        "JSON 응답 (다른 텍스트 금지, 모든 키 포함):\n"
    )

    try:
        response = parser.run_prompt(prompt, max_new_tokens=512)
    except Exception as exc:
        print(f"⚠ Qwen 정규화 실패, 원본 입력 사용: {exc}")
        return raw

    import re as _re
    match = _re.search(r"\{.*\}", response, _re.DOTALL)
    if not match:
        print(f"⚠ Qwen이 JSON을 반환하지 않음, 원본 사용. 응답 일부: {response[:200]}")
        return raw
    try:
        normalized = json.loads(match.group())
    except json.JSONDecodeError as exc:
        print(f"⚠ JSON 파싱 실패, 원본 사용: {exc}")
        return raw

    # Backfill any missing keys from raw so callers can rely on the contract.
    for key, _ in _CUSTOMER_PROMPTS:
        normalized.setdefault(key, raw.get(key, ""))

    print("\n=== 정규화 결과 ===")
    for key, _ in _CUSTOMER_PROMPTS:
        print(f"  {key}: {normalized.get(key, '')}")
    return normalized


def _load_customer_info_json(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"⚠ PDF 고객 정보 JSON 로드 실패, placeholder 사용: {exc}")
        return None
    if not isinstance(data, dict):
        print("⚠ PDF 고객 정보 JSON이 객체가 아님, placeholder 사용")
        return None

    customer = {}
    for key, _ in _CUSTOMER_PROMPTS:
        value = data.get(key, "")
        customer[key] = str(value).strip() if value is not None else ""
    if not customer.get("request_text"):
        customer["request_text"] = "없음"
    return customer


def _generate_quote_pdf(
    output_dir: Path,
    truck_type: str,
    total_cbm: float,
    stored_items_ko: list | None = None,
    customer_overrides: dict | None = None,
) -> Path | None:
    """Render a quote PDF via ``retpdf/return_pdf.py`` with truck_type patched.

    Uses the canonical customer/format JSONs in ``retpdf/json/`` and writes
    a patched company JSON (with the computed ``truck_type``) into
    ``output_dir`` before invoking the renderer. Returns the PDF path on
    success, ``None`` on failure (with a warning printed).
    """
    retpdf_dir = Path(__file__).resolve().parent / "retpdf"
    if not retpdf_dir.exists():
        print(f"⚠ retpdf 디렉토리를 찾지 못해 PDF 생성을 건너뜁니다: {retpdf_dir}")
        return None
    if str(retpdf_dir) not in sys.path:
        sys.path.insert(0, str(retpdf_dir))
    try:
        import return_pdf as rpdf
    except Exception as exc:
        print(f"⚠ retpdf 모듈 import 실패, PDF 생성을 건너뜁니다: {exc}")
        return None

    try:
        from scripts.generate_company_json import update_company_payload
    except Exception as exc:
        print(f"⚠ 가격 규칙 모듈 import 실패, 가격 자동 계산 스킵: {exc}")
        update_company_payload = None

    try:
        format_path = retpdf_dir / "json" / "quote_format.json"
        customer_path = retpdf_dir / "json" / "quote_customer.json"
        company_path = retpdf_dir / "json" / "quote_company.json"
        rules_path = retpdf_dir / "map" / "quote_rules.json"
        template_path = retpdf_dir / "html" / "quote_template.html"
        for p in (format_path, customer_path, company_path, rules_path, template_path):
            if not p.exists():
                print(f"⚠ retpdf 입력 누락: {p} — PDF 생성을 건너뜁니다.")
                return None

        customer_data = json.loads(customer_path.read_text(encoding="utf-8"))
        company_data = json.loads(company_path.read_text(encoding="utf-8"))
        rules_data = json.loads(rules_path.read_text(encoding="utf-8"))

        # Patch customer fields from interactive input + auto-generated lists.
        if customer_overrides:
            for key, value in customer_overrides.items():
                if value not in (None, ""):
                    customer_data[key] = value
        if stored_items_ko is not None:
            customer_data["stored_items"] = list(stored_items_ko)
        # Per spec: discarded list stays empty for this auto-generated quote.
        customer_data["discarded_items"] = []

        # Map the user-facing truck name to the rule-table key for pricing.
        # Display name (truck_type) stays as the recommendation; the rule key
        # determines which row of the price table to apply.
        rule_key = _TRUCK_TO_RULE_KEY.get(truck_type)
        company_data["truck_type"] = rule_key or truck_type

        if update_company_payload is not None and rule_key is not None:
            try:
                company_data = update_company_payload(
                    customer_data, company_data, rules_data,
                )
                print(
                    f"  가격 자동 계산: 입고 {company_data['inbound_total']}, "
                    f"출고 {company_data['outbound_total']}, "
                    f"계약금 {company_data['deposit_total']}, "
                    f"총액 {company_data['grand_total']}"
                )
            except (KeyError, ValueError) as exc:
                print(
                    f"⚠ '{truck_type}' 가격 규칙 적용 실패 ({exc}); "
                    "기존 placeholder 가격 유지."
                )
        elif rule_key is None:
            print(
                f"⚠ '{truck_type}'은 quote_rules.json에 없음; "
                "가격 자동 계산 스킵 (placeholder 유지)."
            )

        # Restore display name AFTER pricing computation so PDF shows what
        # was actually recommended (e.g., "1.4톤") even if priced as "2.5톤".
        company_data["truck_type"] = truck_type

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        patched_company = output_dir / "quote_company_patched.json"
        patched_company.write_text(
            json.dumps(company_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        patched_customer = output_dir / "quote_customer_patched.json"
        patched_customer.write_text(
            json.dumps(customer_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        data_paths = [format_path, patched_customer, patched_company]
        html_text = rpdf.render_quote_html(template_path, data_paths)
        html_out = output_dir / "quote.html"
        html_out.write_text(html_text, encoding="utf-8")

        pdf_out = output_dir / "quote.pdf"
        rpdf.render_pdf(html_out, pdf_out, rpdf.DEFAULT_CONDA_ENV)
        return pdf_out
    except Exception as exc:
        print(f"⚠ PDF 생성 중 오류: {exc}")
        return None


def _select_marker_candidate(candidates, marker_name: str, marker_def: dict):
    if not candidates:
        return None, []

    # Build candidate face aspects. For flat markers (2D) there's only one pair;
    # for 3D markers (with depth_mm), any of the three faces could be visible,
    # so we score against whichever face aspect best matches.
    dims = [marker_def["width_mm"], marker_def["height_mm"]]
    if "depth_mm" in marker_def:
        dims.append(marker_def["depth_mm"])
    face_aspects = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            a, b = dims[i], dims[j]
            face_aspects.append(max(a, b) / max(min(a, b), 1e-6))
    if not face_aspects:
        face_aspects = [1.0]

    if marker_name == "CreditCard":
        min_area_ratio = 5e-5
        max_area_ratio = 0.03
        max_bbox_area_ratio = 0.05
        min_rectangularity = 0.45
        max_aspect_error = 0.55
    else:
        min_area_ratio = 2e-4
        max_area_ratio = 0.45
        max_bbox_area_ratio = 0.55
        min_rectangularity = 0.50
        max_aspect_error = 0.60

    plausible = []
    debug_rows = []

    for cand in candidates:
        metrics = _marker_mask_metrics(cand.mask, cand.box)
        aspect = max(metrics["oriented_aspect_ratio"], 1e-6)
        aspect_error = min(
            abs(float(np.log(aspect / fa))) for fa in face_aspects
        )

        is_plausible = (
            min_area_ratio <= metrics["area_ratio"] <= max_area_ratio
            and metrics["bbox_area_ratio"] <= max_bbox_area_ratio
            and metrics["rectangularity"] >= min_rectangularity
            and aspect_error <= max_aspect_error
        )

        # Favor segmentation confidence, rectangularity, and correct aspect ratio.
        rank_score = (
            cand.score
            + 0.35 * metrics["rectangularity"]
            - 0.50 * aspect_error
            - 1.50 * metrics["area_ratio"]
        )
        debug_rows.append({
            "score": cand.score,
            "rank_score": rank_score,
            "area_ratio": metrics["area_ratio"],
            "bbox_area_ratio": metrics["bbox_area_ratio"],
            "rectangularity": metrics["rectangularity"],
            "aspect_ratio": aspect,
            "aspect_error": aspect_error,
            "plausible": is_plausible,
        })

        if is_plausible:
            plausible.append((rank_score, cand))

    if not plausible:
        return None, debug_rows

    plausible.sort(key=lambda item: item[0], reverse=True)
    return plausible[0][1], debug_rows


class VolumeEstimationPipeline:
    """End-to-end pipeline: image + Korean text → per-object volumes + logistics."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.parser = KoreanTextParser(self.config)
        self.segmenter = _create_segmenter(self.config)
        self.depth_estimator = _create_depth_estimator(self.config)
        self.shape_prior_estimator = AutomaticShapePriorEstimator(self.config)
        self.standard_size_agent = StandardSizeAgent()
        self.volume_calculator = VolumeCalculator(self.config)
        self._models_loaded = False

    def load_models(self):
        """Load all models once (for batch processing)."""
        if not self._models_loaded:
            print("[*] Loading text parser...")
            self.parser.load()
            print(f"[*] Loading SAM3 and {self.config.depth_backend}...")
            self.segmenter.load()
            self.depth_estimator.load()
            self._models_loaded = True

    def unload_models(self):
        """Unload all models."""
        if self._models_loaded:
            self.parser.unload()
            self.segmenter.unload()
            self.depth_estimator.unload()
            self._models_loaded = False

    def run(
        self,
        image_path: str,
        korean_text: str,
        output_dir: str = None,
        marker: str = None,
        detect_all: bool = False,
        json_only: bool = False,
    ) -> dict:
        """Execute the full pipeline.

        Args:
            image_path: Path to indoor room image.
            korean_text: Korean text describing objects to move.
            output_dir: If provided, save visualization results here.
            marker: Reference marker type (e.g., "CreditCard") for scale calibration.
            detect_all: If True, ignore ``korean_text`` and auto-detect every
                COCO-class object via ``segmenter.segment_all``. Requires a
                segmenter that exposes ``segment_all`` (OpenWorldSAM or the
                SAM3→OpenWorldSAM fallback). Image input only.
            json_only: If True, save only result.json and skip visualizations.

        Returns:
            Dict with per-object volumes, total volume, and logistics recommendation.
        """
        t0 = time.time()
        _owns_models = not self._models_loaded

        # ------------------------------------------------------------------
        # Step 1: Load image and estimate camera intrinsics
        # ------------------------------------------------------------------
        print("[1/6] Loading image...")
        image = Image.open(image_path).convert("RGB")

        # UniDepth/AnyCalib estimate intrinsics from the image.
        if isinstance(self.depth_estimator, (UniDepthV2Estimator, AnyCalibMoCA3DEstimator)):
            if not self._models_loaded:
                self.load_models()
            intrinsics = self.depth_estimator.get_intrinsics(image)
            backend_names = {
                UniDepthV2Estimator: "UniDepth",
                AnyCalibUniDepthEstimator: "AnyCalib+UniDepth",
                AnyCalibMoCA3DEstimator: "AnyCalib",
            }
            backend_name = backend_names.get(type(self.depth_estimator), "Unknown")
            print(f"  Image size: {image.size}, Intrinsics ({backend_name}): "
                  f"fx={intrinsics['fx']:.0f}, fy={intrinsics['fy']:.0f}")
        else:
            intrinsics = estimate_camera_intrinsics(image)
            print(f"  Image size: {image.size}, Intrinsics (EXIF/heuristic): "
                  f"fx={intrinsics['fx']:.0f}")

        # ------------------------------------------------------------------
        # Step 2: Identify objects — parse Korean text OR auto-detect
        # ------------------------------------------------------------------
        if not self._models_loaded:
            self.load_models()

        detected_segments = None  # unique_name → SegmentationResult
        object_categories = {}
        vlm_expected_counts = None
        if detect_all:
            print("[2/6] Auto-detecting all objects...")
            segment_all_fn = getattr(self.segmenter, "segment_all", None)
            if segment_all_fn is None:
                raise ValueError(
                    "--detect-all requires a segmenter with segment_all support "
                    "(use --seg-backend openworldsam or sam3_fallback)."
                )
            detections = segment_all_fn(image)
            object_names = []
            detected_segments = {}
            name_counts = {}
            for det in detections:
                base = det.object_name
                idx = name_counts.get(base, 0)
                name_counts[base] = idx + 1
                unique_name = base if idx == 0 else f"{base}_{idx}"
                det.object_name = unique_name
                object_names.append(unique_name)
                detected_segments[unique_name] = det
                object_categories[unique_name] = base
            print(f"  Auto-detected {len(object_names)} object(s): {object_names}")
        else:
            print("[2/6] Parsing Korean text...")
            object_names = self.parser.parse(korean_text)
            object_categories = {name: name for name in object_names}
            print(f"  Objects: {object_names}")

        marker_def = self.config.marker_definitions.get(marker) if marker else None
        # If the marker prompt is in Korean (or contains non-ASCII), route it
        # through the same Qwen-based parser used for object names so users can
        # write "냉장고" and have it translated to "refrigerator" for SAM3.
        if marker_def is not None and marker_def.get("prompt"):
            original_prompt = marker_def["prompt"]
            if not original_prompt.isascii():
                translated = self.parser.parse(original_prompt)
                if translated:
                    marker_def = dict(marker_def)  # don't mutate shared config
                    marker_def["prompt"] = translated[0]
                    print(
                        f"  Translated marker prompt: '{original_prompt}' → "
                        f"'{marker_def['prompt']}'"
                    )
                else:
                    print(
                        f"  ⚠ Could not translate marker prompt '{original_prompt}', "
                        "using original"
                    )
        if not detect_all:
            print("  Segmenting all instances for each object prompt...")
            expanded_object_names = []
            expanded_segments = {}
            expanded_categories = {}
            name_counts = {}
            for object_name in object_names:
                seg_results = self.segmenter.segment(
                    image,
                    object_name,
                    return_all=True,
                )
                seg_results = _limit_segments_to_expected_count(
                    seg_results,
                    object_name,
                    vlm_expected_counts,
                )
                if seg_results is None:
                    expanded_object_names.append(object_name)
                    expanded_segments[object_name] = None
                    expanded_categories[object_name] = object_name
                    print(f"    {object_name}: 0 instance(s)")
                    continue

                expected_text = ""
                if vlm_expected_counts and object_name in vlm_expected_counts:
                    expected_text = f" (VLM count cap={vlm_expected_counts[object_name]})"
                print(f"    {object_name}: {len(seg_results)} instance(s){expected_text}")
                for seg in seg_results:
                    prompt_count = name_counts.get(object_name, 0)
                    unique_name = _make_instance_name(
                        object_name,
                        prompt_count,
                        len(seg_results),
                    )
                    name_counts[object_name] = prompt_count + 1
                    seg.object_name = unique_name
                    expanded_object_names.append(unique_name)
                    expanded_segments[unique_name] = seg
                    expanded_categories[unique_name] = object_name

            object_names = expanded_object_names
            detected_segments = expanded_segments
            object_categories = expanded_categories
            print(f"  Expanded objects: {object_names}")

        # ------------------------------------------------------------------
        # Step 3: Marker-based scale calibration (optional)
        # ------------------------------------------------------------------
        scale_factor = 1.0
        marker_result = None
        if marker:
            if marker_def:
                print(f"[3/6] Detecting marker '{marker}' ({marker_def['prompt']})...")
                marker_seg = None
                marker_debug = []
                marker_seg_all = self.segmenter.segment(
                    image,
                    marker_def["prompt"],
                    return_all=True,
                )
                if marker_seg_all is not None:
                    marker_seg, marker_debug = _select_marker_candidate(
                        marker_seg_all, marker, marker_def
                    )
                if marker_debug:
                    debug_str = ", ".join(
                        (
                            f"score={row['score']:.2f}/rank={row['rank_score']:.2f}/"
                            f"area={row['area_ratio']:.4f}/rect={row['rectangularity']:.2f}/"
                            f"aspect={row['aspect_ratio']:.2f}/ok={row['plausible']}"
                        )
                        for row in marker_debug[:5]
                    )
                    print(f"  Marker candidates: {debug_str}")

                if marker_seg is not None:
                    marker_mask = marker_seg.mask
                    marker_frame_image = image
                    marker_frame_index = None
                    marker_box = _mask_to_box(marker_mask)
                    marker_result = {
                        "name": marker,
                        "prompt": marker_def["prompt"],
                        "mask": marker_mask,
                        "box": marker_box,
                        "score": marker_seg.score,
                        "frame_image": marker_frame_image,
                        "frame_index": marker_frame_index,
                    }
                    print(f"  Marker detected (score={marker_seg.score:.2f}, "
                          f"mask={int(marker_mask.sum())}px)")

                    # Get marker 3D points
                    if isinstance(self.depth_estimator, AnyCalibMoCA3DEstimator):
                        marker_geometry = self.depth_estimator.estimate_object_geometry(
                            marker_frame_image,
                            marker_box,
                            mask=marker_mask,
                            intrinsics=intrinsics,
                        )
                        marker_pts = marker_geometry["obb_geometry"]["corners"]
                        marker_result["moca3d_dimensions_m"] = marker_geometry["volume_info"]["dimensions_m"]
                    elif isinstance(self.depth_estimator, UniDepthV2Estimator):
                        marker_pts = self.depth_estimator.get_points3d_for_mask(
                            image, marker_mask
                        )
                    else:
                        depth_map = self.depth_estimator.get_depth_map(
                            image, intrinsics=intrinsics
                        )
                        ys, xs = np.where(marker_mask)
                        pts_2d = np.stack([xs, ys], axis=1)
                        depths = depth_map[ys, xs].astype(float)
                        marker_pts = self.volume_calculator.backproject_to_3d(
                            pts_2d, depths,
                            intrinsics["fx"], intrinsics["fy"],
                            intrinsics["cx"], intrinsics["cy"],
                        )

                    if len(marker_pts) > 0 and not isinstance(self.depth_estimator, AnyCalibMoCA3DEstimator):
                        raw_marker_count = len(marker_pts)
                        marker_pts = self.volume_calculator.filter_points_by_depth(marker_pts)
                        if len(marker_pts) != raw_marker_count:
                            print(f"  Marker depth filter: {raw_marker_count} -> {len(marker_pts)} points")

                    scale_factor = self.volume_calculator.compute_marker_scale_factor(
                        marker_pts,
                        marker_def["width_mm"],
                        marker_def["height_mm"],
                        marker_def.get("depth_mm"),
                    )

                    if len(marker_pts) >= 4:
                        est_extents = np.sort(
                            self.volume_calculator._pca_extents(marker_pts)
                        )[::-1]
                        if "depth_mm" in marker_def:
                            known_sorted = sorted(
                                [marker_def["width_mm"], marker_def["height_mm"],
                                 marker_def["depth_mm"]],
                                reverse=True,
                            )
                            print(f"  Marker estimated size: "
                                  f"{est_extents[0]*1000:.1f} x {est_extents[1]*1000:.1f} x "
                                  f"{est_extents[2]*1000:.1f} mm")
                            print(f"  Marker actual size:    "
                                  f"{known_sorted[0]:.1f} x {known_sorted[1]:.1f} x "
                                  f"{known_sorted[2]:.1f} mm")
                        else:
                            print(f"  Marker estimated size: "
                                  f"{est_extents[0]*1000:.1f} x {est_extents[1]*1000:.1f} mm")
                            print(f"  Marker actual size:    "
                                  f"{marker_def['width_mm']:.1f} x {marker_def['height_mm']:.1f} mm")
                    else:
                        print("  Marker 3D points were insufficient for a robust size estimate.")
                    print(f"  Scale factor: {scale_factor:.3f}")
                else:
                    print(
                        f"  ⚠ Marker '{marker}' not detected, "
                        "proceeding without calibration"
                    )
            else:
                print(f"  ⚠ Unknown marker type '{marker}', skipping calibration")

        # ------------------------------------------------------------------
        # Step 4: Per-object processing
        # ------------------------------------------------------------------
        results = []
        for i, obj_name in enumerate(object_names):
            obj_category = object_categories.get(obj_name, obj_name)
            moca_geometry_result = None
            print(f"[4/6] Processing '{obj_name}' ({i+1}/{len(object_names)})...")

            # 4a. Segment
            if detected_segments is not None:
                seg = detected_segments.get(obj_name)
                if seg is None:
                    print(f"  ⚠ '{obj_name}' not detected, skipping")
                    results.append({"object": obj_name, "object_category": obj_category, "error": "not detected"})
                    continue
            else:
                seg_results = self.segmenter.segment(image, obj_name)
                if seg_results is None:
                    print(f"  ⚠ '{obj_name}' not detected, skipping")
                    results.append({"object": obj_name, "object_category": obj_category, "error": "not detected"})
                    continue
                seg = seg_results[0]  # Best detection on the image

            best_frame_index = None
            best_frame_image = image
            best_mask = seg.mask
            best_box = seg.box
            mask_pixels = int(best_mask.sum())
            print(f"  Detected (score={seg.score:.2f}, mask={mask_pixels}px)")

            standard_estimate = self.standard_size_agent.estimate(obj_name, obj_category)
            if standard_estimate is not None:
                volume_info = {
                    "volume_m3": standard_estimate.volume_m3,
                    "volume_cm3": standard_estimate.volume_cm3,
                    "dimensions_m": standard_estimate.dimensions_m,
                    "correction_applied": True,
                    "estimation_mode": "standard_size_agent",
                    "standard_size_metadata": standard_estimate.metadata,
                    "object": obj_name,
                    "object_category": obj_category,
                    "mask_pixels_original": mask_pixels,
                    "reference_mask_pixels": int(seg.mask.sum()),
                    "detection_score": seg.score,
                    "n_3d_points": 0,
                    "mask": best_mask,
                    "box": best_box,
                    "frame_image": best_frame_image,
                    "frame_index": best_frame_index,
                    "obb_geometry": None,
                }
                results.append(volume_info)
                dims = standard_estimate.dimensions_m
                print(
                    "  StandardSizeAgent: using fixed dimensions "
                    f"{dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}m"
                )
                print(f"  Volume: {volume_info['volume_m3']:.4f} m³ "
                      f"({volume_info['volume_cm3']:,.0f} cm³)")
                continue

            # 4b+4c. Get 3D points for masked pixels
            if isinstance(self.depth_estimator, AnyCalibMoCA3DEstimator):
                try:
                    moca_geometry_result = self.depth_estimator.estimate_object_geometry(
                        image,
                        best_box,
                        mask=best_mask,
                        intrinsics=intrinsics,
                    )
                except Exception as exc:
                    print(f"  ⚠ MoCA3D cuboid estimation failed: {exc}")
                    results.append({
                        "object": obj_name,
                        "object_category": obj_category,
                        "error": f"moca3d estimation failed: {exc}",
                    })
                    continue
                points_3d = moca_geometry_result["points_3d"]
                dims = moca_geometry_result["volume_info"]["dimensions_m"]
                print(
                    "  Using AnyCalib+MoCA3D cuboid: "
                    f"{len(points_3d)} corners, "
                    f"dims={dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}m"
                )
                if len(points_3d) > 0:
                    z_vals = points_3d[:, 2]
                    print(f"  Corner depth range: {z_vals.min():.2f}m - {z_vals.max():.2f}m")
            elif isinstance(self.depth_estimator, UniDepthV2Estimator):
                points_3d = self.depth_estimator.get_points3d_for_mask(image, best_mask)
                print(f"  Using model 3D points: {len(points_3d)} valid points "
                      f"(mask={mask_pixels}px at original res)")
                if len(points_3d) > 0:
                    z_vals = points_3d[:, 2]
                    print(f"  Depth range: {z_vals.min():.2f}m - {z_vals.max():.2f}m")
            else:
                # Other backends: dense depth map + manual backprojection
                depth_map = self.depth_estimator.get_depth_map(image, intrinsics=intrinsics)
                ys, xs = np.where(best_mask)
                points = np.stack([xs, ys], axis=1)  # (N, 2)
                depths = depth_map[ys, xs].astype(float)
                print(f"  Using dense depth map: {len(points)} mask points")

                valid_depths = [d for d in depths if d is not None]
                print(f"  Valid depths: {len(valid_depths)}/{len(depths)}")
                if valid_depths:
                    print(f"  Depth range: {min(valid_depths):.2f}m - {max(valid_depths):.2f}m")

                points_3d = self.volume_calculator.backproject_to_3d(
                    points, depths,
                    intrinsics["fx"], intrinsics["fy"],
                    intrinsics["cx"], intrinsics["cy"],
                )

            if len(points_3d) > 0 and not isinstance(self.depth_estimator, AnyCalibMoCA3DEstimator):
                raw_count = len(points_3d)
                points_3d = self.volume_calculator.filter_points_by_depth(points_3d)
                filter_name = "Depth outlier"
                if len(points_3d) != raw_count:
                    print(f"  {filter_name} filter: {raw_count} -> {len(points_3d)} points")

            # Apply marker scale calibration
            if scale_factor != 1.0 and len(points_3d) > 0:
                if moca_geometry_result is not None:
                    moca_geometry_result = self.depth_estimator.scale_object_geometry(
                        moca_geometry_result, scale_factor
                    )
                    points_3d = moca_geometry_result["points_3d"]
                else:
                    points_3d = points_3d * scale_factor
                print(f"  Applied marker scale: ×{scale_factor:.3f}")

            obj_intrinsics = intrinsics
            obj_depth_map = None

            # 4d. Compute volume from 3D points
            volume_mode = "obb"
            shape_prior = None
            if moca_geometry_result is not None:
                volume_info = dict(moca_geometry_result["volume_info"])
                obb_geometry = moca_geometry_result["obb_geometry"]
            else:
                shape_prior = self.shape_prior_estimator.estimate(
                    object_category=obj_category,
                    mask=best_mask,
                    box=best_box,
                    points_3d=points_3d,
                    intrinsics=obj_intrinsics,
                    depth_map=obj_depth_map,
                    video_mode=False,
                )
                if shape_prior is not None:
                    volume_info = {
                        "volume_m3": shape_prior.volume_m3,
                        "volume_cm3": shape_prior.volume_cm3,
                        "dimensions_m": shape_prior.dimensions_m,
                        "correction_applied": shape_prior.correction_applied,
                        "estimation_mode": shape_prior.mode,
                    }
                    obb_geometry = shape_prior.obb_geometry
                    print(
                        f"  Applied automatic box-geometry refinement "
                        f"(rectangularity={shape_prior.metadata['rectangularity']:.2f})"
                    )
                else:
                    volume_info = self.volume_calculator.compute_obb_volume(
                        points_3d, obj_category, mode=volume_mode
                    )
                    obb_geometry = self.volume_calculator.compute_obb_geometry(
                        points_3d,
                        obj_category,
                        robust=bool(volume_mode in ("camera_aligned", "hybrid")),
                    )
            volume_info["object"] = obj_name
            volume_info["object_category"] = obj_category
            volume_info["mask_pixels_original"] = mask_pixels
            volume_info["reference_mask_pixels"] = int(seg.mask.sum())
            volume_info["detection_score"] = seg.score
            volume_info["n_3d_points"] = len(points_3d)
            volume_info["mask"] = best_mask
            volume_info["box"] = best_box
            volume_info["frame_image"] = best_frame_image
            volume_info["frame_index"] = best_frame_index
            # Compute full OBB geometry for 3D visualization
            volume_info["obb_geometry"] = obb_geometry
            if shape_prior is not None:
                volume_info["shape_prior_metadata"] = shape_prior.metadata
            results.append(volume_info)
            print(f"  Volume: {volume_info['volume_m3']:.4f} m³ "
                  f"({volume_info['volume_cm3']:,.0f} cm³)")

        # ------------------------------------------------------------------
        # Step 5: Logistics recommendation
        # ------------------------------------------------------------------
        print("[5/6] Generating recommendation...")
        valid_results = [r for r in results if "error" not in r]
        raw_total_m3 = sum(
            r.get("volume_m3", 0) for r in results
            if "error" not in r
        )
        total_m3 = sum(r.get("volume_m3", 0) for r in valid_results)
        recommendation = generate_logistics_recommendation(
            valid_results, self.config.logistics_thresholds,
        )
        print(recommendation)

        # ------------------------------------------------------------------
        # Step 6: Save results
        # ------------------------------------------------------------------
        t_save_start = time.time()
        if output_dir:
            print("[6/6] Saving results...")
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            def _clear_existing_outputs(directory: Path, patterns) -> None:
                for pattern in patterns:
                    for old_path in directory.glob(pattern):
                        if old_path.is_file():
                            old_path.unlink()

            # Clear prior artifacts in the target output folder so stale files from
            # an older run do not get mixed with the current result set.
            _clear_existing_outputs(
                out,
                ("result_vis.jpg", "result_3d_obb.jpg", "result_3d_pointcloud.png",
                 "result.json", "depth_with_objects.png"),
            )
            if json_only:
                for stale_dir_name in ("01_depth", "02_segmentation", "03_object_depth", "04_pointcloud"):
                    stale_dir = out / stale_dir_name
                    if stale_dir.exists():
                        _clear_existing_outputs(stale_dir, ("*.png", "*.jpg", "*.jpeg"))

            # Compute depth map once (used by combined viz + Modules 1/3 below).
            if isinstance(self.depth_estimator, UniDepthV2Estimator):
                depth_map = self.depth_estimator.get_depth_map(image)
            else:
                depth_map = self.depth_estimator.get_depth_map(image, intrinsics=intrinsics)

            # Combined depth + all segmented objects — saved unconditionally,
            # including in --json-only mode (this is the single visual the user
            # always wants).
            visualize_objects_on_depth(
                image, results, depth_map,
                str(out / "depth_with_objects.png"),
            )
            print(f"  Saved combined viz: {out / 'depth_with_objects.png'}")

            if not json_only:
                # --- Module 1: Depth map visualization ---
                depth_dir = out / "01_depth"
                depth_dir.mkdir(exist_ok=True)
                _clear_existing_outputs(depth_dir, ("*.png", "*.jpg", "*.jpeg"))
                visualize_depth_map(image, depth_map, str(depth_dir / "depth_map.png"))

                # --- Module 2: Per-object segmentation masks ---
                seg_dir = out / "02_segmentation"
                seg_dir.mkdir(exist_ok=True)
                _clear_existing_outputs(seg_dir, ("*.png", "*.jpg", "*.jpeg"))
                if marker_result is not None:
                    marker_safe_name = marker_result["name"].replace(" ", "_").replace("/", "_")
                    visualize_per_object_mask(
                        marker_result.get("frame_image", image),
                        marker_result["mask"],
                        f"marker:{marker_result['prompt']}",
                        marker_result["score"],
                        save_path=str(seg_dir / f"00_marker_{marker_safe_name}_mask.png"),
                    )
                for i, res in enumerate(results):
                    if "error" in res:
                        continue
                    obj_name = res.get("object", f"obj{i}")
                    safe_name = obj_name.replace(" ", "_").replace("/", "_")
                    visualize_per_object_mask(
                        res.get("frame_image", image), res["mask"], obj_name, res.get("detection_score", 0),
                        save_path=str(seg_dir / f"{i:02d}_{safe_name}_mask.png"),
                    )

                # --- Module 3: Per-object depth ---
                obj_depth_dir = out / "03_object_depth"
                obj_depth_dir.mkdir(exist_ok=True)
                _clear_existing_outputs(obj_depth_dir, ("*.png", "*.jpg", "*.jpeg"))
                for i, res in enumerate(results):
                    if "error" in res:
                        continue
                    obj_name = res.get("object", f"obj{i}")
                    safe_name = obj_name.replace(" ", "_").replace("/", "_")
                    obj_frame_image = res.get("frame_image", image)
                    visualize_object_depth(
                        obj_frame_image, res["mask"], depth_map, obj_name,
                        save_path=str(obj_depth_dir / f"{i:02d}_{safe_name}_depth.png"),
                    )

                # --- Combined visualizations (existing) ---
                visualize_results(image, results, str(out / "result_vis.jpg"))

                visualize_3d_obb_on_image(
                    image, results, intrinsics,
                    save_path=str(out / "result_3d_obb.jpg"),
                )
            else:
                print("  JSON-only mode: skipping visualization outputs.")

            # JSON output (exclude non-serializable fields)
            import numpy as _np
            def _to_json(v):
                if isinstance(v, _np.ndarray):
                    return v.tolist()
                if isinstance(v, _np.floating):
                    return float(v)
                if isinstance(v, _np.integer):
                    return int(v)
                return v

            json_results = []
            for r in results:
                jr = {}
                for k, v in r.items():
                    if k in ("mask", "obb_geometry", "frame_image"):  # skip non-serializable blobs
                        continue
                    jr[k] = _to_json(v)
                json_results.append(jr)

            output_data = {
                "image": str(image_path),
                "input_text": korean_text,
                "marker": marker,
                "scale_factor": scale_factor,
                "objects": json_results,
                "raw_total_volume_m3": raw_total_m3,
                "raw_total_volume_cm3": raw_total_m3 * 1e6,
                "total_volume_m3": total_m3,
                "total_volume_cm3": total_m3 * 1e6,
                "recommendation": recommendation,
                "processing_seconds": round(t_save_start - t0, 1),
                "save_seconds": round(time.time() - t_save_start, 1),
                "elapsed_seconds": round(time.time() - t0, 1),
            }
            with open(out / "result.json", "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"  Saved to {out}")
        else:
            print("[6/6] Done (no output dir specified).")

        # Cleanup only if this run loaded the models itself
        if _owns_models:
            self.unload_models()

        t_end = time.time()
        elapsed = t_end - t0
        processing_seconds = t_save_start - t0
        save_seconds = t_end - t_save_start
        print(
            f"\nTotal time: {elapsed:.1f}s "
            f"(processing {processing_seconds:.1f}s + save {save_seconds:.1f}s)"
        )

        return {
            "objects": results,
            "total_volume_m3": total_m3,
            "raw_total_volume_m3": raw_total_m3,
            "scale_factor": scale_factor,
            "recommendation": recommendation,
            "elapsed_seconds": round(elapsed, 1),
            "processing_seconds": round(processing_seconds, 1),
            "save_seconds": round(save_seconds, 1),
        }


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _print_summary(result: dict):
    """Print a JSON summary of one run."""
    import numpy as _np
    def _to_json(v):
        if isinstance(v, _np.ndarray): return v.tolist()
        if isinstance(v, _np.floating): return float(v)
        if isinstance(v, _np.integer): return int(v)
        return v
    summary = {
        "objects": [
            {k: _to_json(v) for k, v in obj.items() if k not in ("mask", "box", "obb_geometry", "frame_image")}
            for obj in result["objects"]
        ],
        "total_volume_m3": result["total_volume_m3"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Indoor object volume estimation pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        help="Path to a single indoor room image",
    )
    group.add_argument(
        "--image-dir",
        help="Path to a directory of images (batch mode, models loaded once)",
    )
    parser.add_argument(
        "--text", default=None,
        help='Korean text describing objects (e.g., "책상, 의자, 소파 옮겨줘"). '
             'With --image-dir, omit to use each filename as object name.',
    )
    parser.add_argument(
        "--text-file", default=None,
        help="Path to a UTF-8 text file containing Korean object names. "
             "If provided, this overrides --text.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for results (optional)",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="When --output is set, save only result.json and skip all image visualizations.",
    )
    parser.add_argument(
        "--depth-backend", default="unidepth",
        choices=["depth_anything_v2", "unidepth", "anycalib_unidepth", "anycalib_moca3d"],
        help="Depth estimation backend (default: unidepth)",
    )
    parser.add_argument(
        "--seg-backend", default="sam3",
        choices=["sam3", "openworldsam", "sam3_fallback"],
        help="Segmentation backend (default: sam3, sam3_fallback = SAM3 → OpenWorldSAM)",
    )
    parser.add_argument(
        "--marker", default=None,
        help="Reference marker name for scale calibration. Built-in presets: "
             "CreditCard (85.60x53.98mm), A4 (297x210mm), "
             "Microwave (435x395x255mm). For a custom marker, pass any name and "
             "supply --marker-prompt, --marker-width-mm, --marker-height-mm "
             "(and optionally --marker-depth-mm for 3D markers).",
    )
    parser.add_argument(
        "--marker-prompt", default=None,
        help="Segmentation prompt for a custom marker. English (e.g., "
             "'microwave oven') is used directly; Korean (e.g., '냉장고') is "
             "translated to English via the same Qwen parser that processes "
             "object names.",
    )
    parser.add_argument(
        "--marker-width-mm", type=float, default=None,
        help="Custom marker real-world width in mm.",
    )
    parser.add_argument(
        "--marker-height-mm", type=float, default=None,
        help="Custom marker real-world height in mm.",
    )
    parser.add_argument(
        "--marker-depth-mm", type=float, default=None,
        help="Custom marker real-world depth in mm (third axis). "
             "Use for 3D box-shaped markers like a microwave.",
    )
    parser.add_argument(
        "--moca3d-ckpt", default=None,
        help="Optional MoCA3D checkpoint path for --depth-backend anycalib_moca3d "
             "(default: MOCA3D_CHECKPOINT_PATH or MoCA3D/checkpoints/moca3d.safetensors).",
    )
    parser.add_argument(
        "--moca3d-dinov3-ckpt", default=None,
        help="Optional DINOv3 checkpoint path used by MoCA3D "
             "(default: MOCA3D_DINOV3_CHECKPOINT_PATH or MoCA3D/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth).",
    )
    parser.add_argument(
        "--anycalib-model-id", default=None,
        help="AnyCalib model_id for --depth-backend anycalib_unidepth/anycalib_moca3d "
             "(default: ANYCALIB_MODEL_ID or anycalib_pinhole).",
    )
    parser.add_argument(
        "--anycalib-cam-id", default=None,
        help="AnyCalib cam_id for --depth-backend anycalib_unidepth/anycalib_moca3d "
             "(default: ANYCALIB_CAM_ID or pinhole).",
    )
    parser.add_argument(
        "--detect-all", action="store_true",
        help="Auto-detect every COCO-class object in the image and ignore --text. "
             "Requires --seg-backend openworldsam or sam3_fallback. Image input only.",
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="After the batch finishes, render a quote PDF via retpdf/return_pdf.py "
             "with truck_type patched from the computed CBM → truck mapping. "
             "Requires --output.",
    )
    parser.add_argument(
        "--pdf-customer-json", default=None,
        help="Optional JSON file with PDF customer fields. Skips interactive customer prompts.",
    )
    args = parser.parse_args()

    if args.text_file:
        args.text = Path(args.text_file).read_text(encoding="utf-8").strip()

    if args.detect_all:
        if args.seg_backend not in ("openworldsam", "sam3_fallback"):
            parser.error(
                "--detect-all requires --seg-backend openworldsam or sam3_fallback"
            )

    config = PipelineConfig(
        depth_backend=args.depth_backend,
        segmentation_backend=args.seg_backend,
    )
    if args.moca3d_ckpt:
        config.moca3d_checkpoint_path = args.moca3d_ckpt
    if args.moca3d_dinov3_ckpt:
        config.moca3d_dinov3_checkpoint_path = args.moca3d_dinov3_ckpt
    if args.anycalib_model_id:
        config.anycalib_model_id = args.anycalib_model_id
    if args.anycalib_cam_id:
        config.anycalib_cam_id = args.anycalib_cam_id

    # Register or override a marker definition from CLI inputs.
    marker_type = args.marker
    _marker_overrides_provided = any(
        v is not None for v in (
            args.marker_prompt, args.marker_width_mm,
            args.marker_height_mm, args.marker_depth_mm,
        )
    )
    if marker_type is not None and _marker_overrides_provided:
        if marker_type in config.marker_definitions:
            marker_def = dict(config.marker_definitions[marker_type])
            if args.marker_prompt is not None:
                marker_def["prompt"] = args.marker_prompt
            if args.marker_width_mm is not None:
                marker_def["width_mm"] = args.marker_width_mm
            if args.marker_height_mm is not None:
                marker_def["height_mm"] = args.marker_height_mm
            if args.marker_depth_mm is not None:
                marker_def["depth_mm"] = args.marker_depth_mm
            config.marker_definitions[marker_type] = marker_def
        else:
            missing = [
                name for name, val in (
                    ("--marker-prompt", args.marker_prompt),
                    ("--marker-width-mm", args.marker_width_mm),
                    ("--marker-height-mm", args.marker_height_mm),
                ) if val is None
            ]
            if missing:
                raise ValueError(
                    f"Unknown marker '{marker_type}'. To register a custom marker "
                    f"you must also pass: {', '.join(missing)}"
                )
            custom_def = {
                "prompt": args.marker_prompt,
                "width_mm": args.marker_width_mm,
                "height_mm": args.marker_height_mm,
            }
            if args.marker_depth_mm is not None:
                custom_def["depth_mm"] = args.marker_depth_mm
            config.marker_definitions[marker_type] = custom_def
    elif marker_type is not None and marker_type not in config.marker_definitions:
        raise ValueError(
            f"Unknown marker '{marker_type}'. Either pick a preset "
            f"({', '.join(sorted(config.marker_definitions.keys()))}) or supply "
            "--marker-prompt, --marker-width-mm, --marker-height-mm to register it."
        )

    pipeline = VolumeEstimationPipeline(config)

    # --- Single image mode ---
    if args.image:
        if not args.text and not args.detect_all:
            # Use filename as object name
            args.text = Path(args.image).stem + " 옮겨줘"
        result = pipeline.run(
            args.image, args.text or "", args.output,
            marker=marker_type, detect_all=args.detect_all,
            json_only=args.json_only,
        )
        print("\n" + "=" * 50)
        print("Summary (JSON):")
        _print_summary(result)
        return

    batch_dir = Path(args.image_dir)
    batch_files = sorted(
        p for p in batch_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    batch_label = "images"
    item_label = "Image"

    if not batch_files:
        print(f"No {batch_label} found in {batch_dir}")
        return

    use_filename_as_text = args.text is None and not args.detect_all
    print(f"=== Batch mode: {len(batch_files)} {batch_label} in {batch_dir} ===")
    if args.detect_all:
        print("  Text: (ignored — --detect-all is enabled)")
    elif use_filename_as_text:
        print("  Text: (using filename as object name)")
    else:
        print(f"  Text: {args.text}")
    print(f"  Backend: {args.depth_backend}\n")

    pipeline.load_models()

    # Collect customer info up front so the operator can fill it out while
    # the (long) batch runs, instead of being forced to wait at the end.
    customer_overrides = None
    if args.pdf:
        customer_overrides = _load_customer_info_json(args.pdf_customer_json)
        if customer_overrides is None:
            customer_overrides = _collect_customer_info_interactive(pipeline.parser)

    batch_total_m3 = 0.0
    batch_raw_total_m3 = 0.0
    batch_total_elapsed = 0.0
    batch_total_processing = 0.0
    batch_total_save = 0.0
    batch_category_counts = Counter()
    batch_files_summary = []
    batch_processed = 0
    batch_failed = 0

    for idx, input_path in enumerate(batch_files):
        if args.detect_all:
            text = ""
        elif use_filename_as_text:
            text = f"{input_path.stem} 옮겨줘"
        else:
            text = args.text

        print(f"\n{'='*60}", flush=True)
        label = "(auto-detect)" if args.detect_all else f'"{text}"'
        print(f"[{item_label} {idx+1}/{len(batch_files)}] {input_path.name} → {label}", flush=True)
        print(f"{'='*60}", flush=True)

        if args.output:
            out_dir = str(Path(args.output) / input_path.stem)
        else:
            out_dir = None

        try:
            result = pipeline.run(
                str(input_path), text, out_dir,
                marker=marker_type, detect_all=args.detect_all,
                json_only=args.json_only,
            )
            _print_summary(result)
            batch_processed += 1
            file_total_m3 = float(result.get("total_volume_m3", 0.0))
            file_raw_total_m3 = float(result.get("raw_total_volume_m3", file_total_m3))
            batch_total_m3 += file_total_m3
            batch_raw_total_m3 += file_raw_total_m3

            valid_objects = [
                obj for obj in result.get("objects", [])
                if "error" not in obj
            ]
            file_category_counts = Counter(
                obj.get("object_category") or obj.get("object")
                for obj in valid_objects
            )
            batch_category_counts.update(file_category_counts)
            file_elapsed = float(result.get("elapsed_seconds", 0.0))
            file_processing = float(result.get("processing_seconds", 0.0))
            file_save = float(result.get("save_seconds", 0.0))
            batch_total_elapsed += file_elapsed
            batch_total_processing += file_processing
            batch_total_save += file_save
            print(
                f"  ⏱  {input_path.name}: {file_elapsed:.1f}s "
                f"(processing {file_processing:.1f}s + save {file_save:.1f}s)"
            )
            batch_files_summary.append({
                "file": str(input_path),
                "output_dir": out_dir,
                "elapsed_seconds": file_elapsed,
                "processing_seconds": file_processing,
                "save_seconds": file_save,
                "raw_total_volume_m3": file_raw_total_m3,
                "raw_total_volume_cm3": file_raw_total_m3 * 1e6,
                "total_volume_m3": file_total_m3,
                "total_volume_cm3": file_total_m3 * 1e6,
                "num_detected_objects": len(valid_objects),
                "detected_categories": sorted(file_category_counts.keys()),
                "category_counts": dict(sorted(file_category_counts.items())),
            })
        except Exception as e:
            batch_failed += 1
            print(f"  ERROR processing {input_path.name}: {e}")
            continue

    pipeline.unload_models()
    n = batch_processed if batch_processed else 1
    avg_elapsed = batch_total_elapsed / n
    avg_processing = batch_total_processing / n
    avg_save = batch_total_save / n

    # CBM (= m³) and truck recommendation from the aggregate batch volume.
    cbm_int, recommended_truck = _recommend_truck(batch_total_m3)

    batch_summary = {
        "input_dir": str(batch_dir),
        "num_files": len(batch_files),
        "num_processed": batch_processed,
        "num_failed": batch_failed,
        "total_elapsed_seconds": round(batch_total_elapsed, 1),
        "total_processing_seconds": round(batch_total_processing, 1),
        "total_save_seconds": round(batch_total_save, 1),
        "avg_elapsed_seconds_per_file": round(avg_elapsed, 1),
        "avg_processing_seconds_per_file": round(avg_processing, 1),
        "avg_save_seconds_per_file": round(avg_save, 1),
        "raw_total_volume_m3": batch_raw_total_m3,
        "raw_total_volume_cm3": batch_raw_total_m3 * 1e6,
        "total_volume_m3": batch_total_m3,
        "total_volume_cm3": batch_total_m3 * 1e6,
        "total_cbm_floor": cbm_int,
        "recommended_truck": recommended_truck,
        "detected_categories": sorted(batch_category_counts.keys()),
        "category_counts": dict(sorted(batch_category_counts.items())),
        "files": batch_files_summary,
    }

    print(
        f"\n=== 부피/트럭 추천 ===\n"
        f"  총 부피: {batch_total_m3:.2f} m³ → {cbm_int} CBM\n"
        f"  추천 트럭: {recommended_truck}"
    )

    if args.output:
        summary_path = Path(args.output) / "batch_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved batch summary: {summary_path}")

        if args.pdf:
            stored_items_ko = _translate_categories_ko(
                batch_summary.get("detected_categories", [])
            )
            pdf_out = _generate_quote_pdf(
                Path(args.output), recommended_truck, batch_total_m3,
                stored_items_ko=stored_items_ko,
                customer_overrides=customer_overrides,
            )
            if pdf_out is not None:
                print(f"Saved quote PDF: {pdf_out}")
    elif args.pdf:
        print("⚠ --pdf 옵션은 --output과 함께 사용해야 합니다 (출력 경로 없음).")

    print(f"\n=== Batch complete: {batch_processed}/{len(batch_files)} {batch_label} processed ===")
    print("Final batch summary:")
    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
