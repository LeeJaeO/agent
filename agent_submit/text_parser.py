"""Korean text → English object names using Qwen3.5-4B (with dictionary fallback)."""

import json
import re
from typing import List, Optional

import torch
from config import PipelineConfig


# ---------------------------------------------------------------------------
# Fallback dictionary: common Korean furniture / household item names
# ---------------------------------------------------------------------------
KO_EN_DICT = {
    "책상": "desk", "의자": "chair", "소파": "sofa", "테이블": "table",
    "침대": "bed", "옷짐": "clothes", "옷장": "wardrobe", "서랍장": "dresser", "책장": "bookshelf",
    "냉장고": "refrigerator", "세탁기": "washing machine", "워시타워": "washing machine", "건조기": "dryer",
    "TV": "TV", "티비": "TV", "모니터": "monitor", "컴퓨터": "computer",
    "에어컨": "air conditioner", "선풍기": "fan", "전자레인지": "microwave",
    "오븐": "oven", "식탁": "dining table", "화장대": "vanity",
    "거울": "mirror", "장식장": "display case", "신발장": "shoe rack",
    "수납장": "storage box", "행거": "clothes rack", "스탠드": "lamp",
    "협탁": "nightstand", "서재": "study desk", "탁자": "table",
    "쿠션": "cushion", "매트리스": "bed", "이불": "blanket",
    "러그": "rug", "카펫": "carpet", "커튼": "curtain",
    "피아노": "piano", "기타": "guitar", "스피커": "speaker",
    "프린터": "printer", "복합기": "multifunction printer",
    "돌침대": "stone bed", "실내자전거": "exercise bike",
    "스텐드 에어컨": "standing air conditioner",
    "스탠드 에어컨": "standing air conditioner",
    "거실장": "TV stand", "김치냉장고": "kimchi refrigerator",
    "렌지다이": "microwave stand", "식기류": "dishes",
    "잔짐": "miscellaneous", "잡동사니": "miscellaneous", "기타짐": "miscellaneous",
}


KO_EXCLUDE_TERMS = (
    "붙박이장", "붙박이 장", "붙박이옷장", "붙박이 옷장",
    "빌트인장", "빌트인 장", "빌트인옷장", "빌트인 옷장",
)


def _remove_excluded_korean_terms(text: str) -> str:
    cleaned = str(text or "")
    for term in KO_EXCLUDE_TERMS:
        cleaned = cleaned.replace(term, " ")
    return cleaned


EN_NORMALIZATION_MAP = {
    # Remove sizes/counts and unify harmless spelling variants, while keeping
    # subtype descriptors such as 4-door, kimchi, standing, wall-mounted, etc.
    "four-door refrigerator": "4-door refrigerator",
    "display cabinet": "display case",
    "storage cabinet": "storage box",
    "drawer cabinet": "storage box",
    "filing cabinet": "file storage unit",
    "kitchen cabinet": "",
    "built-in cabinet": "",
    "built in cabinet": "",
    "built-in wardrobe": "",
    "built in wardrobe": "",
    "built-in closet": "",
    "built in closet": "",
    "wall cabinet": "",
    "drawer": "dresser",
    "drawer chest": "dresser",
    "chest of drawers": "dresser",
    "book shelf": "",
    "bookshelf": "",
    "storage shelf": "",
    "wall shelf": "",
    "shelf": "",
    "shelves": "",
    "cabinet": "",
    "40-inch tv": "tv",
    "40 inch tv": "tv",
    "chair 3 pcs": "chair",
    "chair 3 pieces": "chair",
    "storage box 9 pcs": "storage box",
    "storage box 9 pieces": "storage box",
    "mattress": "bed",
    "queen bed": "bed",
    "king bed": "bed",
    "single bed": "bed",
    "blankets": "blanket",
    "curtains": "curtain",
    "utensils": "dishes",
    "clothing": "clothes",
    "clothing storage": "clothes",
    "clothes storage": "clothes",
    "garment storage": "clothes",
    "linen": "miscellaneous",
    "linens": "miscellaneous",
    "small items": "miscellaneous",
    "miscellaneous items": "miscellaneous",
}


def normalize_english_item_name(name: str) -> str:
    """Normalize translated names into simple SAM-friendly prompts."""
    item = str(name).strip().lower()
    item = re.sub(r"\([^)]*\)", " ", item)
    item = re.sub(r"\b(\d+)\s*-\s*door\s+refrigerator\b", r"\1-door refrigerator", item)
    item = re.sub(r"\b\d+\s*(inch|inches|in)\s*[- ]*", "", item)
    item = re.sub(r"\b\d+\s*(pcs?|pieces?|units?|items?)\b", " ", item)
    item = re.sub(r"\s+", " ", item).strip(" ,-")
    if re.search(
        r"\b(built[- ]?in|wall[- ]?fixed|fitted)\b.*\b(closets?|cabinets?|wardrobes?)\b",
        item,
    ):
        return ""
    item = EN_NORMALIZATION_MAP.get(item, item)
    if re.search(r"\bcabinets?\b", item):
        return ""
    if re.search(r"\b(book\s*)?shel(f|ves)\b", item):
        return ""
    item = re.sub(r"\s+", " ", item).strip(" ,-")
    return item


def _unique_normalized(items) -> List[str]:
    seen = set()
    out = []
    for item in items:
        normalized = normalize_english_item_name(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _parse_with_dictionary(text: str) -> List[str]:
    """Simple dictionary-based Korean → English extraction."""
    results = []
    # Pull out longer dictionary phrases first so terms like "스탠드 에어컨"
    # don't get split into unrelated smaller tokens.
    remaining = text
    for ko, en in sorted(KO_EN_DICT.items(), key=lambda item: len(item[0]), reverse=True):
        if ko in remaining:
            results.append(en)
            remaining = remaining.replace(ko, " ")
    # Remove common Korean action words
    text = re.sub(r'(옮겨줘|옮겨|치워줘|치워|보내줘|빼줘|빼|좀|이사|할|거야|것들|하고|이랑|그리고)', '', remaining)
    # Split by common delimiters
    parts = re.split(r'[,，\s、/]+', text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in KO_EN_DICT:
            results.append(KO_EN_DICT[part])
        elif part.isascii():
            results.append(part.lower())
    return _unique_normalized(results)


class KoreanTextParser:
    """Parse Korean text to extract English object names using Qwen3.5-4B."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the Qwen3.5-4B model for translation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading translation model: {self.config.translation_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.translation_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.translation_model,
            torch_dtype=self.config.dtype,
            device_map="auto",
        )
        self.model.eval()

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()

    def run_prompt(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Run an arbitrary prompt through the loaded Qwen model and return raw text.

        Used by callers that need structured-output normalization beyond the
        ``parse()`` JSON-array contract (e.g., quote customer info cleanup).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Parser model not loaded; call load() first.")
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def parse(self, korean_text: str) -> List[str]:
        """Extract English object names from Korean text.

        Tries LLM first, falls back to dictionary if LLM fails.
        """
        korean_text = _remove_excluded_korean_terms(korean_text)

        # Try LLM-based extraction
        if self.model is not None:
            try:
                return self._parse_with_llm(korean_text)
            except Exception as e:
                print(f"  LLM parsing failed ({e}), falling back to dictionary")

        # Fallback: dictionary-based
        result = _parse_with_dictionary(korean_text)
        if result:
            return result

        # Last resort: return the raw text split
        return [korean_text.strip()]

    def _parse_with_llm(self, korean_text: str) -> List[str]:
        """Use Qwen3.5-4B to extract and translate object names."""
        prompt = (
            "Translate Korean moving item names into specific English object prompts "
            "for image segmentation. Return ONLY a JSON array like "
            "[\"desk\", \"chair\"]. No explanation.\n"
            "Rules:\n"
            "- Preserve the specific item type as much as possible; do not over-simplify "
            "distinct objects into a generic category.\n"
            "- Use concise segmentation-friendly noun phrases, usually 1-4 words.\n"
            "- Remove quantities, physical sizes, counts, colors, brands, and request/action words.\n"
            "- Keep subtype/style descriptors that identify a different movable object type, such as "
            "4-door, kimchi, standing, wall-mounted, freestanding, or foldable.\n"
            "- Translate 수납장 as storage box, but translate 서랍장 as dresser. "
            "Do not include built-in, wall-fixed, or fitted closets/cabinets/wardrobes.\n"
            "- Translate 옷짐 as clothes, not clothing storage, closet, wardrobe, or furniture.\n"
            "- Do not output shelf, shelves, bookshelf, or book shelf.\n"
            "- Translate 매트리스 as bed so bed and mattress are not segmented twice.\n"
            "- Translate 잔짐/잡동사니/기타짐 as miscellaneous, not linen.\n"
            "- Translate only the item type: '의자3개' -> 'chair', "
            "'옷짐' -> 'clothes', "
            "'잔짐' -> 'miscellaneous', "
            "'퀸침대' -> 'bed', '매트리스' -> 'bed', "
            "'4도어 냉장고' -> '4-door refrigerator', "
            "'김치냉장고' -> 'kimchi refrigerator', "
            "'워시타워' -> 'washing machine', "
            "'장식장' -> 'display case', "
            "'수납장' -> 'storage box', '서랍장' -> 'dresser', "
            "'신발장' -> 'shoe rack', "
            "'40인치 TV' -> 'tv', '스탠드형에어컨' -> 'standing air conditioner'.\n"
            "Examples:\n"
            '"책상, 의자, 소파 옮겨줘" -> ["desk", "chair", "sofa"]\n'
            '"퀸침대, 매트리스" -> ["bed"]\n'
            '"옷짐, 옷장" -> ["clothes", "wardrobe"]\n'
            '"잔짐, 잡동사니" -> ["miscellaneous"]\n'
            '"붙박이장, 수납장, 서랍장" -> ["storage box", "dresser"]\n'
            '"냉장고랑 세탁기 빼줘" -> ["refrigerator", "washing machine"]\n'
            '"워시타워 옮겨줘" -> ["washing machine"]\n'
            '"장식장, 신발장, 김치냉장고" -> ["display case", "shoe rack", "kimchi refrigerator"]\n'
            '"스탠드, 화분 옮겨줘" -> ["floor lamp", "potted plant"]\n'
            f'Korean: "{korean_text}"'
        )

        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False: disable thinking mode for Qwen3.x (ignored by other models)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Parse JSON from response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            items = json.loads(match.group())
            return _unique_normalized(items)

        raise ValueError(f"Could not parse JSON from LLM response: {response}")
