from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.steps.step4_ocr_pages import (
    LayoutRegion,
    PageOCRContent,
    Step4Error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EASYOCR_CACHE_DIR = PROJECT_ROOT / "cache" / "easyocr"
EASYOCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_LANGUAGES = {
    "vi": "vi",
    "zh-Hant": "ch_tra",
    "fr": "fr",
    "en": "en",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_easyocr_result(
    results: Any,
    image_width: int,
    image_height: int,
) -> PageOCRContent:
    if not isinstance(results, (list, tuple)):
        raise Step4Error("EasyOCR returned an invalid result list.")
    regions: list[LayoutRegion] = []
    for index, result in enumerate(results):
        if (
            not isinstance(result, (list, tuple))
            or len(result) != 3
            or not isinstance(result[1], str)
        ):
            raise Step4Error(f"EasyOCR result {index} is invalid.")
        polygon, text, confidence = result
        if not text.strip():
            continue
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise Step4Error(
                f"EasyOCR result {index} has an invalid confidence score."
            ) from exc
        if not math.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
            raise Step4Error(
                f"EasyOCR result {index} has an invalid confidence score."
            )
        if (
            not isinstance(polygon, (list, tuple))
            or len(polygon) < 4
            or any(
                not isinstance(point, (list, tuple))
                or len(point) < 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in point[:2]
                )
                for point in polygon
            )
        ):
            raise Step4Error(f"EasyOCR result {index} has an invalid polygon.")
        x0 = min(float(point[0]) for point in polygon)
        y0 = min(float(point[1]) for point in polygon)
        x1 = max(float(point[0]) for point in polygon)
        y1 = max(float(point[1]) for point in polygon)
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(image_width), x1)
        y1 = min(float(image_height), y1)
        if x0 >= x1 or y0 >= y1:
            raise Step4Error(
                f"EasyOCR result {index} does not intersect the page."
            )
        regions.append(
            LayoutRegion(
                type="text",
                text=text.strip(),
                bbox=[
                    x0 / image_width,
                    y0 / image_height,
                    x1 / image_width,
                    y1 / image_height,
                ],
            )
        )
    return PageOCRContent(
        markdown="\n".join(region.text for region in regions),
        regions=regions,
    )


class EasyOCRAdapter:
    """EasyOCR adapter; emits text regions in EasyOCR reading order."""

    model_name = "EasyOCR"
    model_version = "EasyOCR"

    def __init__(self, language: str = "vi") -> None:
        if language not in _LANGUAGES:
            raise Step4Error(
                "EasyOCR language must be one of: "
                + ", ".join(_LANGUAGES)
                + "."
            )
        try:
            import easyocr
        except ImportError as exc:
            raise Step4Error(
                "EasyOCR adapter requires the optional dependency. "
                "Install it with `python -m pip install -e \".[easyocr]\"`."
            ) from exc
        self.language = language
        self.easyocr_language = _LANGUAGES[language]
        try:
            self._reader = easyocr.Reader(
                [self.easyocr_language],
                gpu=False,
                model_storage_directory=str(EASYOCR_CACHE_DIR),
                verbose=False,
            )
            version = getattr(easyocr, "__version__", "unknown")
        except Exception as exc:
            raise Step4Error(
                f"Cannot initialize EasyOCR ({self.easyocr_language}): {exc}"
            ) from exc
        self.model_version = f"EasyOCR {version} ({self.easyocr_language})"

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        try:
            import numpy as np

            with Image.open(image_path) as image:
                image_width, image_height = image.size
                image_array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            results = _jsonable(
                self._reader.readtext(
                    image_array,
                    detail=1,
                    paragraph=False,
                )
            )
            raw_response: object = _jsonable({
                "engine": self.model_name,
                "version": self.model_version,
                "language": self.easyocr_language,
                "results": results,
            })
            content = _normalize_easyocr_result(
                results,
                image_width,
                image_height,
            )
            return content, raw_response
        except Step4Error as exc:
            if getattr(exc, "raw_response", None) is not None:
                raise
            raise Step4Error(str(exc), locals().get("raw_response")) from exc
        except (OSError, UnidentifiedImageError) as exc:
            raise Step4Error(
                f"Cannot read OCR page image {image_path}: {exc}"
            ) from exc
        except Exception as exc:
            raise Step4Error(
                f"EasyOCR inference failed for {image_path}: {exc}"
            ) from exc
