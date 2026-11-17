from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ocr_pipeline.adapters.preprocess import (
    PreparedPage,
    prepare_page,
)
from ocr_pipeline.adapters.text_layout import (
    TextBox,
    TextLine,
    is_noise_text,
    join_text_boxes,
    markdown_from_paragraphs,
    normalize_box,
    normalize_text,
    union_bbox,
)
from ocr_pipeline.steps.step4_ocr_pages import (
    LayoutRegion,
    PageOCRContent,
    Step4Error,
)

_LANGUAGES = {
    "vi": "vie",
    "vi+zh-Hant": "vie+chi_tra",
    "zh-Hant": "chi_tra",
    "fr": "fra",
    "en": "eng",
}
TESSERACT_CONFIG = (
    "--oem 1 --psm 3 "
    "-c preserve_interword_spaces=1 "
    "-c user_defined_dpi=300"
)
TESSERACT_MIN_TEXT_SIDE = 2400


def _integer(value: object, field: str, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise Step4Error(
            f"Tesseract returned an invalid {field} for word {index}."
        )
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise Step4Error(
            f"Tesseract returned an invalid {field} for word {index}."
        ) from exc


def _word_bbox(
    left: object,
    top: object,
    width: object,
    height: object,
    image_width: int,
    image_height: int,
    index: int,
) -> tuple[float, float, float, float]:
    x0 = _integer(left, "left coordinate", index)
    y0 = _integer(top, "top coordinate", index)
    word_width = _integer(width, "width", index)
    word_height = _integer(height, "height", index)
    x1 = x0 + word_width
    y1 = y0 + word_height
    if (
        x0 < 0
        or y0 < 0
        or x1 > image_width
        or y1 > image_height
        or x0 >= x1
        or y0 >= y1
    ):
        raise Step4Error(
            f"Tesseract returned out-of-page coordinates for word {index}."
        )
    return float(x0), float(y0), float(x1), float(y1)


def _normalize_tesseract_result(
    data: dict[str, Any],
    image_width: int,
    image_height: int,
    language: str = "vi",
) -> PageOCRContent:
    required_fields = (
        "text",
        "left",
        "top",
        "width",
        "height",
        "block_num",
        "par_num",
        "line_num",
    )
    if any(not isinstance(data.get(field), (list, tuple)) for field in required_fields):
        raise Step4Error("Tesseract returned an invalid TSV result.")
    lengths = {len(data[field]) for field in required_fields}
    if len(lengths) != 1:
        raise Step4Error("Tesseract returned mismatched TSV columns.")

    has_confidence = isinstance(data.get("conf"), (list, tuple)) and len(
        data["conf"]
    ) == len(data["text"])

    line_order: list[tuple[int, int, int]] = []
    line_boxes: dict[tuple[int, int, int], list[TextBox]] = {}
    paragraph_of_line: dict[tuple[int, int, int], tuple[int, int]] = {}
    for index, value in enumerate(data["text"]):
        text = normalize_text(value) if isinstance(value, str) else ""
        if not text:
            continue
        if has_confidence and is_noise_text(text):
            try:
                confidence = float(data["conf"][index])
            except (OverflowError, TypeError, ValueError):
                confidence = None
            if confidence is not None and confidence < 50:
                continue
        line_key = (
            _integer(data["block_num"][index], "block number", index),
            _integer(data["par_num"][index], "paragraph number", index),
            _integer(data["line_num"][index], "line number", index),
        )
        if line_key not in line_boxes:
            line_order.append(line_key)
            line_boxes[line_key] = []
            paragraph_of_line[line_key] = line_key[:2]
        line_boxes[line_key].append(
            TextBox(
                text=text,
                bbox=_word_bbox(
                    data["left"][index],
                    data["top"][index],
                    data["width"][index],
                    data["height"][index],
                    image_width,
                    image_height,
                    index,
                ),
            )
        )

    lines: list[TextLine] = []
    paragraphs: list[list[TextLine]] = []
    previous_paragraph: tuple[int, int] | None = None
    for line_key in line_order:
        boxes = line_boxes[line_key]
        line = TextLine(
            text=join_text_boxes(boxes, language),
            bbox=union_bbox(box.bbox for box in boxes),
        )
        if not line.text:
            continue
        lines.append(line)
        paragraph = paragraph_of_line[line_key]
        if paragraph != previous_paragraph:
            paragraphs.append([])
            previous_paragraph = paragraph
        paragraphs[-1].append(line)

    regions = [
        LayoutRegion(
            type="text",
            text=line.text,
            bbox=normalize_box(line.bbox, image_width, image_height),
        )
        for line in lines
    ]
    return PageOCRContent(
        markdown=markdown_from_paragraphs(paragraphs, language),
        regions=regions,
    )


class TesseractOCRAdapter:
    """Tesseract 5 OCR adapter; emits line-level text regions."""

    model_name = "Tesseract OCR"
    model_version = "Tesseract 5"

    def __init__(
        self,
        language: str = "vi",
        tesseract_cmd: str | None = None,
        preprocess: bool = True,
        min_text_side: int = TESSERACT_MIN_TEXT_SIDE,
    ) -> None:
        if language not in _LANGUAGES:
            raise Step4Error(
                "Tesseract language must be one of: "
                + ", ".join(_LANGUAGES)
                + "."
            )
        try:
            import pytesseract
        except ImportError as exc:
            raise Step4Error(
                "Tesseract adapter requires the optional dependency. "
                "Install it with `python -m pip install -e \".[tesseract]\"`."
            ) from exc

        executable = tesseract_cmd or os.environ.get("TESSERACT_CMD")
        if executable:
            pytesseract.pytesseract.tesseract_cmd = executable
        self._pytesseract = pytesseract
        self.language = language
        self.tesseract_language = _LANGUAGES[language]
        self.preprocess = preprocess
        self.min_text_side = min_text_side
        try:
            version = pytesseract.get_tesseract_version()
        except Exception as exc:
            raise Step4Error(
                "Cannot run the Tesseract executable. Install Tesseract 5 "
                "and its language data, or set TESSERACT_CMD. "
                f"Details: {exc}"
            ) from exc
        self.model_version = f"Tesseract {version}"
        self.configuration = {
            "preprocess": self.preprocess,
            "min_text_side": self.min_text_side,
            "tesseract_config": TESSERACT_CONFIG,
        }

    def _load_image(self, image_path: Path) -> tuple[Image.Image, dict[str, object]]:
        if not self.preprocess:
            with Image.open(image_path) as image:
                return image.convert("L"), {"mode": "disabled"}
        prepared: PreparedPage = prepare_page(
            image_path,
            min_text_side=self.min_text_side,
        )
        return prepared.image, prepared.metadata()

    def recognize(self, image_path: Path) -> tuple[PageOCRContent, object]:
        try:
            image, preprocess_metadata = self._load_image(image_path)
            image_width, image_height = image.size
            data = self._pytesseract.image_to_data(
                image,
                lang=self.tesseract_language,
                config=TESSERACT_CONFIG,
                output_type=self._pytesseract.Output.DICT,
            )
            raw_response: object = {
                "engine": self.model_name,
                "version": self.model_version,
                "language": self.tesseract_language,
                "config": TESSERACT_CONFIG,
                "preprocess": preprocess_metadata,
                "data": data,
            }
            if not isinstance(data, dict):
                raise Step4Error("Tesseract returned a non-object TSV result.")
            content = _normalize_tesseract_result(
                data,
                image_width,
                image_height,
                self.language,
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
                f"Tesseract inference failed for {image_path}: {exc}"
            ) from exc
