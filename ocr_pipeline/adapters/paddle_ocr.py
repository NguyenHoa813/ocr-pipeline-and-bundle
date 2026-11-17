from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLEX_CACHE_DIR = PROJECT_ROOT / "cache" / "paddlex"
PADDLEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(PADDLEX_CACHE_DIR)
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.steps.step4_ocr_pages import (
    ImageKind,
    LayoutRegion,
    PageOCRContent,
    RegionType,
    Step4Error,
)


_LANGUAGES = {
    "vi": "vi",
    "zh-Hant": "chinese_cht",
    "fr": "fr",
    "en": "en",
}
_RECOGNITION_MODELS = {
    "vi": "latin_PP-OCRv5_mobile_rec",
    "zh-Hant": "PP-OCRv6_medium_rec",
    "fr": "latin_PP-OCRv5_mobile_rec",
    "en": "PP-OCRv6_medium_rec",
}
TEXT_DET_LIMIT_SIDE_LEN = 1600
_LABEL_TYPES: dict[str, RegionType] = {
    "abstract": "text",
    "aside_text": "text",
    "caption": "text",
    "chart": "figure",
    "content": "text",
    "diagram": "figure",
    "doc_title": "heading",
    "equation": "formula",
    "equation_block": "formula",
    "figure": "figure",
    "figure_title": "text",
    "footer": "page_footer",
    "footnote": "text",
    "formula": "formula",
    "header": "page_header",
    "heading": "heading",
    "image": "figure",
    "number": "page_number",
    "other": "text",
    "page-footer": "page_footer",
    "page-header": "page_header",
    "page-number": "page_number",
    "page_footer": "page_footer",
    "page_header": "page_header",
    "page_number": "page_number",
    "paragraph": "text",
    "paragraph_title": "heading",
    "photo": "figure",
    "seal": "stamp",
    "stamp": "stamp",
    "table": "table",
    "table_title": "text",
    "text": "text",
    "title": "heading",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _paddle_result_payload(result: Any) -> dict[str, Any]:
    result_json = getattr(result, "json", None)
    if callable(result_json):
        result_json = result_json()
    payload = _jsonable(result_json)
    if not isinstance(payload, dict):
        raise Step4Error("PaddleOCR returned an invalid result object.")
    return payload


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("markdown", "text", "html", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value.strip()
        return ""
    if isinstance(content, list):
        return "\n".join(
            text for item in content if (text := _block_text(item))
        )
    return ""


def _normalized_bbox(
    bbox: Any,
    width: int,
    height: int,
) -> list[float] | None:
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in bbox
        )
    ):
        return None
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if (
        x0 < 0
        or y0 < 0
        or x1 > width
        or y1 > height
        or x0 >= x1
        or y0 >= y1
    ):
        return None
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _parse_structure_result(
    payload: dict[str, Any],
    width: int,
    height: int,
) -> PageOCRContent:
    result_payload = payload.get("res", payload)
    if not isinstance(result_payload, dict):
        raise Step4Error("PaddleOCR returned an invalid layout result.")
    blocks = result_payload.get("parsing_res_list")
    if not isinstance(blocks, list):
        raise Step4Error("PaddleOCR layout output has no parsing_res_list.")

    regions: list[LayoutRegion] = []
    markdown_blocks: list[str] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise Step4Error(f"PaddleOCR layout block {index} is invalid.")
        label = block.get("block_label")
        if not isinstance(label, str):
            raise Step4Error(f"PaddleOCR layout block {index} has no label.")
        bbox = _normalized_bbox(block.get("block_bbox"), width, height)
        if bbox is None:
            raise Step4Error(
                f"PaddleOCR layout block {index} has invalid or out-of-page "
                "coordinates."
            )
        text = _block_text(block.get("block_content"))
        region_type = _LABEL_TYPES.get(label.lower().replace(" ", "_"), "text")
        image_kind: ImageKind | None = None
        if region_type == "figure":
            if label.lower() in {"chart", "diagram", "drawing"}:
                image_kind = "drawing"
            elif label.lower() in {"image", "photo"}:
                image_kind = "photo"
        regions.append(
            LayoutRegion(
                type=region_type,
                text=text,
                bbox=bbox,
                image_kind=image_kind,
            )
        )
        if text:
            if region_type == "heading":
                markdown_blocks.append(f"## {text}")
            elif region_type == "formula":
                markdown_blocks.append(f"$${text}$$")
            else:
                markdown_blocks.append(text)

    return PageOCRContent(
        markdown="\n\n".join(markdown_blocks),
        regions=regions,
    )


class PaddleOCRAdapter:
    """PP-OCRv6 detection with language-appropriate recognition and layout."""

    model_name = "PaddleOCR PP-OCRv6 + PP-StructureV3"
    model_version = "PP-OCRv6_medium_det + PP-OCRv6_medium_rec + PP-StructureV3"
    _engine: Any

    def __init__(self, language: str = "vi") -> None:
        if language not in _LANGUAGES:
            raise Step4Error(
                "PaddleOCR language must be one of: "
                + ", ".join(_LANGUAGES)
                + "."
            )
        self.language = language
        self.paddle_language = _LANGUAGES[language]
        self.text_recognition_model_name = _RECOGNITION_MODELS[language]
        self.model_version = (
            f"PP-OCRv6_medium_det + {self.text_recognition_model_name} "
            "+ PP-StructureV3"
        )
        try:
            from paddleocr import PPStructureV3

            self._engine = PPStructureV3(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name=self.text_recognition_model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_seal_recognition=True,
                use_table_recognition=True,
                use_formula_recognition=True,
            )
        except Exception as exc:
            raise Step4Error(
                "Cannot initialize PaddleOCR PP-OCRv6/PP-StructureV3: "
                f"{exc}"
            ) from exc

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        raw_response: object | None = None
        try:
            with Image.open(image_path) as image:
                width, height = image.size
            results = self._engine.predict(
                input=str(image_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=TEXT_DET_LIMIT_SIDE_LEN,
                text_det_limit_type="max",
            )
            result_list = list(results)
            if not result_list:
                raise Step4Error("PaddleOCR returned no page result.")
            raw_response = _paddle_result_payload(result_list[0])
            content = _parse_structure_result(raw_response, width, height)
            return content, raw_response
        except Step4Error as exc:
            if exc.raw_response is not None:
                raise
            if raw_response is not None:
                raise Step4Error(str(exc), raw_response) from exc
            raise Step4Error(str(exc)) from exc
        except (OSError, UnidentifiedImageError) as exc:
            raise Step4Error(
                f"Cannot read OCR page image {image_path}: {exc}"
            ) from exc
        except Exception as exc:
            raise Step4Error(
                f"PaddleOCR inference failed for {image_path}: {exc}"
            ) from exc
