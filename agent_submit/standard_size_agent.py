"""Fixed-size estimates for objects that should not use depth volume."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StandardSizeEstimate:
    dimensions_m: list[float]
    volume_m3: float
    metadata: dict

    @property
    def volume_cm3(self) -> float:
        return self.volume_m3 * 1e6


class StandardSizeAgent:
    """Provides catalog-style dimensions for special-case moving items."""

    _STANDARD_DIMENSIONS_M = {
        # Typical full-length mirror packing envelope: width x thickness x height.
        "mirror": [0.60, 0.05, 1.60],
    }

    @staticmethod
    def _norm_name(name: str) -> str:
        text = str(name or "").strip().lower()
        text = re.sub(r"_\d+$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def estimate(self, object_name: str, object_category: str = "") -> StandardSizeEstimate | None:
        names = [self._norm_name(object_category), self._norm_name(object_name)]
        matched_key = None
        dims = None
        for name in names:
            if not name:
                continue
            dims = self._STANDARD_DIMENSIONS_M.get(name)
            if dims is not None:
                matched_key = name
                break
            for key, candidate_dims in self._STANDARD_DIMENSIONS_M.items():
                if re.search(rf"\b{re.escape(key)}\b", name):
                    matched_key = key
                    dims = candidate_dims
                    break
            if dims is not None:
                break
        if dims is None:
            return None

        volume = float(dims[0] * dims[1] * dims[2])
        return StandardSizeEstimate(
            dimensions_m=list(dims),
            volume_m3=volume,
            metadata={
                "agent": "StandardSizeAgent",
                "category": matched_key,
                "matched_object_name": self._norm_name(object_name),
                "matched_object_category": self._norm_name(object_category),
                "reason": "fixed standard size used instead of depth-based volume",
            },
        )
